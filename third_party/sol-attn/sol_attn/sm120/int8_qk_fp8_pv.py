# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Experimental SM120 attention: BF16 pooled QK, INT8 exact QK, E4M3 PV.

The warp-MMA/TMA execution skeleton and online-softmax helpers are adapted
from NVIDIA cuDNN Frontend's SM120 block-sparse-attention kernel.  Sol-specific
routing, CTA-local exact-index compaction, approximate block mass, and the
mixed approximate/exact mainloop are implemented here.
"""

from __future__ import annotations

# ruff: noqa: SIM102, N803
import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
import triton
import triton.language as tl
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from sol_attn.common import layout_utils
from sol_attn.common.selector import (
    sol_attn_popc_b32,
    sol_attn_route_is_exact,
)

# Reuse numerical/routing helpers, while keeping the experimental mainloop separate.
from sol_attn.sm120.mainloop import (
    apply_route_mask,
    finalize_softmax,
    gemm_rs_smem,
    gemm_smem_zero_acc,
    mask_exact_scores,
    online_softmax,
    online_softmax_route,
    reduce_route_columns,
    rescale_o_for_next_acc,
)

M = 64
N = 64
D = 128
DV = 128
THREADS = 128
STAGES = 1


P_SCALE = 448.0


@triton.jit
def _precompute_q_correction(
    QMean,
    KCentered,
    Correction,
    T: tl.constexpr,
    H: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_T: tl.constexpr,
):
    """FP32 mean(Q) @ centered-K, computed once outside the attention kernel."""
    bh = tl.program_id(2)
    batch, head = bh // H, bh % H
    groups = tl.program_id(0) * 16 + tl.arange(0, 16)
    keys = tl.program_id(1) * 64 + tl.arange(0, 64)
    d = tl.arange(0, 128)
    qm = tl.load(
        QMean + (bh * GROUPS + groups[:, None]) * 128 + d[None, :],
        groups[:, None] < GROUPS,
        0.0,
    )
    k = tl.load(
        KCentered + ((batch * T + keys[None, :]) * H + head) * 128 + d[:, None],
        keys[None, :] < T,
        0.0,
    )
    correction = tl.dot(qm, k, input_precision="tf32x3")
    tl.store(
        Correction + (bh * GROUPS + groups[:, None]) * PAD_T + keys[None, :],
        correction,
        (groups[:, None] < GROUPS) & (keys[None, :] < PAD_T),
    )


@dsl_user_op
def mma_i8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """Native signed INT8 16x8x32 MMA with four INT32 accumulators per lane."""
    args = [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
    args += [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        args,
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return tuple(cutlass.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@cute.jit
def gemm_qk_i8(acc, a, b, smem_b, copy_b):
    acc.fill(0)
    b_copy = copy_b.retile(b)
    a32 = cute.recast_tensor(a, cutlass.Uint32)
    b32 = cute.recast_tensor(b, cutlass.Uint32)
    for kb in cutlass.range_constexpr(cute.size(a, mode=[2])):
        cute.copy(copy_b, smem_b[None, None, kb], b_copy[None, None, kb])
        for m in cutlass.range_constexpr(cute.size(acc, mode=[1])):
            for n in cutlass.range_constexpr(cute.size(acc, mode=[2])):
                c0, c1, c2, c3 = mma_i8(
                    a32[0, m, kb],
                    a32[1, m, kb],
                    a32[2, m, kb],
                    a32[3, m, kb],
                    b32[0, n, kb],
                    b32[1, n, kb],
                    acc[0, m, n],
                    acc[1, m, n],
                    acc[2, m, n],
                    acc[3, m, n],
                )
                acc[0, m, n], acc[1, m, n], acc[2, m, n], acc[3, m, n] = c0, c1, c2, c3


class SolAttnForwardSm120INT8QKFP8PV:
    """M64/N64/D128, BF16 Q/K, prequantized E4M3 V/VC, BF16 output.

    v_scale[B,H] is shared by V and VC. P is multiplied by 448 before
    conversion, so O remains in units of 448/v_scale until final normalization.
    Pooled QK stays BF16. Exact QK uses signed INT8 MMA / INT32 accumulation
    with per-thread FP32 Q/K scales. K and pooled KC are centered; the same
    FP32 row correction is restored in both score paths. Exact INT8 Q is also
    centered globally per batch/head over original video queries; a precomputed FP32 key-dependent
    correction restores that mean in exact logits only. PV is unchanged.
    """

    def __init__(
        self,
        *,
        debug_route_trace: bool = False,
        debug_qk: bool = False,
        prefetch_first_exact_k: bool = True,
        prefetch_next_route_k: bool = True,
    ):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.tile_shape_qk = (M, N, D)
        self.tile_shape_pv = (M, DV, N)
        self.num_threads = THREADS
        self.q_stage = 1
        self.kv_stage = STAGES
        self.debug_route_trace = debug_route_trace
        self.debug_qk = debug_qk
        if debug_qk and debug_route_trace:
            raise ValueError("QK and routing debug modes use different LSE formats")
        self.prefetch_first_exact_k = prefetch_first_exact_k
        self.prefetch_next_route_k = prefetch_next_route_k

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mThreshold: cute.Tensor,
        mLSE: cute.Tensor,
        mVScale: cute.Tensor,
        mQ8: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mQBias: cute.Tensor,
        mQCorrection: cute.Tensor,
        mQGroup: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_KC: cute.CopyAtom,
        tma_atom_VC: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_i8: cute.TiledMma,
        QI_smem_layout: cute.ComposedLayout,
        KI_smem_layout: cute.ComposedLayout,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
    ):
        cute.experimental.iket.mark("kernel_start")
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_tile_idx, head_idx, batch_idx = cute.arch.block_idx()
        q_tile_idx = cute.arch.make_warp_uniform(q_tile_idx)
        head_idx = cute.arch.make_warp_uniform(head_idx)
        batch_idx = cute.arch.make_warp_uniform(batch_idx)

        value_scale = cutlass.Float32(mVScale[batch_idx, head_idx])
        token_count = mK.shape[0]
        num_blocks = mKC.shape[0]
        num_route_groups = cute.ceil_div(num_blocks, N)
        q_start = q_tile_idx * M
        q_len = token_count - q_start
        if q_len > M:
            q_len = cutlass.Int32(M)
        threshold = cutlass.Float32(mThreshold[batch_idx, q_tile_idx, head_idx])

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)
        if warp == 0 and lane == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_KC)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_VC)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32)
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])),
            barrier_storage=storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])),
            barrier_storage=storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        # KC (BF16, 16 KiB) and exact K (INT8, 8 KiB) share the allocation,
        # but use independent transaction barriers with the correct byte counts.
        KI_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=M * D,
            barrier_storage=storage.KI_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        KI_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        KI_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])),
            barrier_storage=storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        Q_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
        Q_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stage)
        K_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        K_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        V_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        V_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)

        sQ = storage.Q_smem.get_tensor(Q_smem_layout.outer, swizzle=Q_smem_layout.inner)
        sK = storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sQI = cute.make_tensor(
            cute.recast_ptr(
                storage.Q_smem.data_ptr(),
                swizzle_=QI_smem_layout.inner,
                dtype=cutlass.Int8,
            ),
            QI_smem_layout.outer,
        )
        sKI = cute.make_tensor(
            cute.recast_ptr(
                storage.K_smem.data_ptr(),
                swizzle_=KI_smem_layout.inner,
                dtype=cutlass.Int8,
            ),
            KI_smem_layout.outer,
        )
        sV = storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)
        # Reserve bytes [2048,6144) for FP8 P. Routing uses bytes [0,1544).
        # Both live inside the existing Q/O allocation after Q reaches registers.
        sP = cute.make_tensor(
            cute.recast_ptr(storage.Q_smem.data_ptr(), dtype=self.V_dtype) + 2048,
            # XOR byte-address bits [7,8] into [4,5]. This preserves
            # adjacent FP8 pairs and 16-byte ldmatrix row alignment while
            # distributing the eight producer/ldmatrix rows across banks.
            cute.make_composed_layout(
                cute.make_swizzle(2, 4, 3),
                0,
                cute.make_layout((M, N), stride=(N, 1)),
            ),
        )
        # Q is register-resident after the prologue.  Reuse its 16 KiB SMEM
        # allocation for route scratch until the same allocation becomes sO
        # in the epilogue.  This drops the CTA below the 2-block/SM threshold
        # on SM120 without changing any route reduction or synchronization.
        route_f32_ptr = cute.recast_ptr(storage.Q_smem.data_ptr(), dtype=cutlass.Float32)
        route_i32_ptr = cute.recast_ptr(storage.Q_smem.data_ptr(), dtype=cutlass.Int32)
        route_sums = cute.make_tensor(route_f32_ptr, cute.make_layout((4, N)))
        column_masks = cute.make_tensor(route_f32_ptr + 4 * N, cute.make_layout(N))
        route_indices = cute.make_tensor(route_i32_ptr + 5 * N, cute.make_layout(N))
        route_meta = cute.make_tensor(route_i32_ptr + 6 * N, cute.make_layout(2))

        mQ_slice = mQ[None, None, head_idx, batch_idx]
        mK_slice = mK[None, None, head_idx, batch_idx]
        mV_slice = mV[None, None, head_idx, batch_idx]
        mO_slice = mO[None, None, head_idx, batch_idx]
        mKC_slice = mKC[None, None, head_idx, batch_idx]
        mVC_slice = mVC[None, None, head_idx, batch_idx]
        if cutlass.const_expr(not self.debug_route_trace):
            mLSE_slice = mLSE[None, head_idx, batch_idx]

        gQ = cute.local_tile(mQ_slice, (M, D), coord=(q_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (N, D), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (DV, N), coord=(0, None))
        gKC = cute.local_tile(mKC_slice, (N, D), coord=(None, 0))
        gVC = cute.local_tile(mVC_slice, (DV, N), coord=(0, None))
        gO = cute.local_tile(mO_slice, (M, DV), coord=(q_tile_idx, 0))

        cta_coord_layout = (0, cute.make_layout(1))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sKI, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        tKCsK, tKCgKC = cute.nvgpu.cpasync.tma_partition(
            tma_atom_KC,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gKC, 0, 2),
        )
        tVCsV, tVCgVC = cute.nvgpu.cpasync.tma_partition(
            tma_atom_VC,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gVC, 0, 2),
        )

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSsQ = thr_mma_qk.partition_A(sQ)
        tSsK = thr_mma_qk.partition_B(sK)
        tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
        tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((M, N)), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)

        # FP8 m16n8k32 supplies the byte-fragment layout only. Computation
        # below explicitly emits signed INT8 MMA, never FP8 QK arithmetic.
        thr_i8 = tiled_mma_i8.get_slice(tidx)
        tQIsQ = thr_i8.partition_A(sQI)
        tKIsK = thr_i8.partition_B(sKI)
        tQIrQ = cute.make_rmem_tensor(
            tiled_mma_i8.make_fragment_A(tQIsQ[None, None, None, 0]).layout,
            cutlass.Int8,
        )
        tKIrK = cute.make_rmem_tensor(
            tiled_mma_i8.make_fragment_B(tKIsK[None, None, None, 0]).layout,
            cutlass.Int8,
        )
        tSI = cute.make_rmem_tensor(tSrS.layout, cutlass.Int32)
        copy_i8_atom = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Int8)
        copy_QI = cute.make_tiled_copy_A(copy_i8_atom, tiled_mma_i8)
        copy_KI = cute.make_tiled_copy_B(copy_i8_atom, tiled_mma_i8)
        tQIsQ_copy = copy_QI.get_slice(tidx).partition_S(sQI)
        tQIrQ_copy = copy_QI.get_slice(tidx).retile(tQIrQ)
        tKIsK_copy = copy_KI.get_slice(tidx).partition_S(sKI)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tPsP = thr_mma_pv.partition_A(sP)
        tPrP = tiled_mma_pv.make_fragment_A(tPsP)
        tOsV = thr_mma_pv.partition_B(sV)
        tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((M, DV)), self.acc_dtype)

        atom_copy_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.Q_layout.is_m_major_a(), 4),
            self.Q_dtype,
        )
        atom_copy_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.K_layout.is_n_major_b(), 4),
            self.K_dtype,
        )
        # atom_copy_V_old = cute.make_copy_atom(
        #     cute.nvgpu.CopyUniversalOp(),
        #     self.V_dtype,
        # )
        atom_copy_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.V_dtype,
        )
        smem_copy_Q = cute.make_tiled_copy_A(atom_copy_Q, tiled_mma_qk)
        smem_copy_K = cute.make_tiled_copy_B(atom_copy_K, tiled_mma_qk)
        smem_copy_V = cute.make_tiled_copy_B(atom_copy_V, tiled_mma_pv)
        thr_copy_Q = smem_copy_Q.get_slice(tidx)
        thr_copy_K = smem_copy_K.get_slice(tidx)
        thr_copy_V = smem_copy_V.get_slice(tidx)
        tSsQ_copy = thr_copy_Q.partition_S(sQ)
        tSrQ_copy = thr_copy_Q.retile(tSrQ)
        tSsK_copy = thr_copy_K.partition_S(sK)
        tOsV_copy = thr_copy_V.partition_S(sV)

        max_m_layout = cute.make_layout(
            cute.size(
                layout_utils.reshape_acc_to_mn(tOrO).layout,
                mode=[0],
            )
        )
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if warp == 0:
            iket_1 = cute.experimental.iket.range_start("wait_load_q")
            Q_pipeline.producer_acquire(Q_producer)
            cute.experimental.iket.range_end(iket_1)
            iket_2 = cute.experimental.iket.range_start("issue_load_q")
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, Q_producer.index],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer),
            )
            cute.experimental.iket.range_end(iket_2)
            Q_pipeline.producer_commit(Q_producer)
            Q_producer.advance()
        iket_3 = cute.experimental.iket.range_start("wait_setup_cta")
        cute.arch.sync_threads()
        cute.experimental.iket.range_end(iket_3)
        iket_4 = cute.experimental.iket.range_start("wait_setup_q")
        q_wait = Q_pipeline.consumer_try_wait(Q_consumer)
        Q_pipeline.consumer_wait(Q_consumer, q_wait)
        cute.experimental.iket.range_end(iket_4)
        q_stage = Q_consumer.index
        for k_block in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            cute.copy(
                smem_copy_Q,
                tSsQ_copy[None, None, k_block, q_stage],
                tSrQ_copy[None, None, k_block],
            )
        iket_5 = cute.experimental.iket.range_start("release_setup_q")
        Q_pipeline.consumer_release(Q_consumer)
        cute.experimental.iket.range_end(iket_5)
        Q_consumer.advance()

        # BF16 Q is now register-resident. Temporarily reuse Q shared memory
        # to load the prequantized Q bytes; routing/P will reuse it afterwards.
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(M * D // THREADS):
            linear = tidx + i * THREADS
            row, col = linear // D, linear % D
            value = cutlass.Int8(0)
            if q_start + row < token_count:
                value = mQ8[q_start + row, col, head_idx, batch_idx]
            sQI[row, col, 0] = value
        cute.arch.sync_threads()
        for kb in cutlass.range_constexpr(cute.size(tQIrQ, mode=[2])):
            cute.copy(copy_QI, tQIsQ_copy[None, None, kb, 0], tQIrQ_copy[None, None, kb])
        cute.arch.sync_threads()

        # All accumulator coordinates in a lane share one scale pair:
        # Q rows warp*16+r and warp*16+r+8, K columns (2*g,2*g+1)+8*j.
        query_scale = cutlass.Float32(mQScale[batch_idx, head_idx, q_tile_idx * 32 + warp * 8 + lane // 4])
        coordinates_mn = layout_utils.reshape_acc_to_mn(tScS)
        row_bias = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Float32)
        query_group = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        for r in cutlass.range_constexpr(2):
            row = coordinates_mn[r, 0][0]
            bias = cutlass.Float32(0.0)
            if q_start + row < token_count:
                bias = cutlass.Float32(mQBias[batch_idx, q_start + row, head_idx])
            row_bias[r] = bias
            group = cutlass.Int32(-1)
            if q_start + row < token_count:
                group = mQGroup[q_start + row]
            query_group[r] = group

        for route_group in cutlass.range(0, num_route_groups, 1, unroll=1):
            group_start = route_group * cutlass.Int32(N)
            valid_blocks = num_blocks - group_start
            if valid_blocks > N:
                valid_blocks = cutlass.Int32(N)

            if warp == 0:
                if cutlass.const_expr(self.prefetch_next_route_k):
                    # P19-style terminal handoff: when the previous route
                    # group had an exact block, its final exact QK already
                    # refilled this K stage with the current group's KC.
                    if route_group == 0:
                        iket_6 = cute.experimental.iket.range_start("wait_load_k")
                        K_pipeline.producer_acquire(K_producer)
                        cute.experimental.iket.range_end(iket_6)
                        iket_7 = cute.experimental.iket.range_start("issue_load_kc")
                        cute.copy(
                            tma_atom_KC,
                            tKCgKC[None, route_group],
                            tKCsK[None, K_producer.index],
                            tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer),
                        )
                        cute.experimental.iket.range_end(iket_7)
                        K_pipeline.producer_commit(K_producer)
                        K_producer.advance()
                    else:
                        previous_group_exact_count = cutlass.Int32(route_meta[0])
                        if previous_group_exact_count == 0:
                            iket_8 = cute.experimental.iket.range_start("wait_load_k")
                            K_pipeline.producer_acquire(K_producer)
                            cute.experimental.iket.range_end(iket_8)
                            iket_9 = cute.experimental.iket.range_start("issue_load_kc")
                            cute.copy(
                                tma_atom_KC,
                                tKCgKC[None, route_group],
                                tKCsK[None, K_producer.index],
                                tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer),
                            )
                            cute.experimental.iket.range_end(iket_9)
                            K_pipeline.producer_commit(K_producer)
                            K_producer.advance()
                else:
                    iket_10 = cute.experimental.iket.range_start("wait_load_k")
                    K_pipeline.producer_acquire(K_producer)
                    cute.experimental.iket.range_end(iket_10)
                    iket_11 = cute.experimental.iket.range_start("issue_load_kc")
                    cute.copy(
                        tma_atom_KC,
                        tKCgKC[None, route_group],
                        tKCsK[None, K_producer.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer),
                    )
                    cute.experimental.iket.range_end(iket_11)
                    K_pipeline.producer_commit(K_producer)
                    K_producer.advance()
                iket_12 = cute.experimental.iket.range_start("wait_load_v")
                V_pipeline.producer_acquire(V_producer)
                cute.experimental.iket.range_end(iket_12)
                iket_13 = cute.experimental.iket.range_start("issue_load_vc")
                cute.copy(
                    tma_atom_VC,
                    tVCgVC[None, route_group],
                    tVCsV[None, V_producer.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer),
                )
                cute.experimental.iket.range_end(iket_13)
                V_pipeline.producer_commit(V_producer)
                V_producer.advance()

            iket_14 = cute.experimental.iket.range_start("wait_route_k")
            k_wait = K_pipeline.consumer_try_wait(K_consumer)
            K_pipeline.consumer_wait(K_consumer, k_wait)
            cute.experimental.iket.range_end(iket_14)
            iket_15 = cute.experimental.iket.range_start("mma_route_qk")
            gemm_smem_zero_acc(
                tiled_mma_qk,
                tSrS,
                tSrQ,
                tSrK,
                tSsK_copy[None, None, None, K_consumer.index],
                smem_copy_K,
            )
            # Restore the original logit origin before routing and softmax.
            route_scores_mn = layout_utils.reshape_acc_to_mn(tSrS)
            for r in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(cute.size(route_scores_mn, mode=[1])):
                    route_scores_mn[r, c] += row_bias[r]
            cute.experimental.iket.range_end(iket_15)
            iket_16 = cute.experimental.iket.range_start("release_route_k")
            K_pipeline.consumer_release(K_consumer)
            cute.experimental.iket.range_end(iket_16)
            K_consumer.advance()

            iket_17 = cute.experimental.iket.range_start("route_reduce")
            reduce_route_columns(
                tSrS,
                tScS,
                route_sums,
                warp,
                lane,
                q_len,
            )
            cute.experimental.iket.range_end(iket_17)
            cute.arch.fence_view_async_shared()
            iket_18 = cute.experimental.iket.range_start("wait_route_cta")
            cute.arch.sync_threads()
            cute.experimental.iket.range_end(iket_18)

            if warp == 0:
                iket_19 = cute.experimental.iket.range_start("route_select")
                preceding = cutlass.Int32(0)
                lane_mask_lt = cutlass.Int32(0x7FFFFFFF) >> (cutlass.Int32(31) - lane)
                for word in cutlass.range_constexpr(2):
                    off = cutlass.Int32(word * 32) + lane
                    valid = off < valid_blocks
                    exact = False
                    if valid:
                        col_sum = (
                            cutlass.Float32(route_sums[0, off])
                            + cutlass.Float32(route_sums[1, off])
                            + cutlass.Float32(route_sums[2, off])
                            + cutlass.Float32(route_sums[3, off])
                        )
                        col_mean = col_sum * scale_softmax_log2e / cutlass.Float32(q_len)
                        kv_block = group_start + off
                        exact = sol_attn_route_is_exact(
                            q_tile_idx,
                            kv_block,
                            col_mean,
                            threshold,
                            valid,
                        )
                        exact = exact or (kv_block >= sink_start_block and kv_block < sink_end_block)
                    ballot = cutlass.Int32(cute.arch.vote_ballot_sync(exact))
                    column_masks[off] = -cutlass.Float32.inf if (exact or not valid) else cutlass.Float32(0.0)
                    rank = preceding + sol_attn_popc_b32(ballot & lane_mask_lt)
                    if exact:
                        route_indices[rank] = group_start + off
                    preceding += sol_attn_popc_b32(ballot)
                    if cutlass.const_expr(self.debug_route_trace):
                        if lane == 0:
                            mLSE[
                                batch_idx,
                                q_tile_idx,
                                head_idx,
                                route_group,
                                word,
                            ] = ballot
                if lane == 0:
                    route_meta[0] = preceding
                    route_meta[1] = valid_blocks
                    cute.arch.fence_view_async_shared()
                cute.experimental.iket.range_end(iket_19)
            iket_20 = cute.experimental.iket.range_start("wait_route_cta")
            cute.arch.sync_threads()
            cute.experimental.iket.range_end(iket_20)

            exact_count = cutlass.Int32(route_meta[0])
            has_approx = exact_count < valid_blocks
            if cutlass.const_expr(self.prefetch_first_exact_k):
                # Once routing identifies the first exact block, the route KC
                # stage is free.  Refill it before the approximate softmax/PV
                # so the first exact K transfer overlaps that work.
                if warp == 0 and exact_count > 0:
                    first_exact = cutlass.Int32(route_indices[0])
                    iket_21 = cute.experimental.iket.range_start("wait_load_k")
                    KI_pipeline.producer_acquire(KI_producer)
                    cute.experimental.iket.range_end(iket_21)
                    iket_22 = cute.experimental.iket.range_start("issue_load_k")
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, first_exact],
                        tKsK[None, KI_producer.index],
                        tma_bar_ptr=KI_pipeline.producer_get_barrier(KI_producer),
                    )
                    cute.experimental.iket.range_end(iket_22)
                    KI_pipeline.producer_commit(KI_producer)
                    KI_producer.advance()
            iket_23 = cute.experimental.iket.range_start("wait_route_v")
            v_wait = V_pipeline.consumer_try_wait(V_consumer)
            V_pipeline.consumer_wait(V_consumer, v_wait)
            cute.experimental.iket.range_end(iket_23)
            if has_approx:
                iket_24 = cute.experimental.iket.range_start("route_mask")
                apply_route_mask(tSrS, tScS, column_masks, q_len)
                cute.experimental.iket.range_end(iket_24)
                iket_25 = cute.experimental.iket.range_start("softmax_route")
                row_scale = online_softmax_route(
                    tSrS,
                    tScS,
                    max_m,
                    sum_m,
                    scale_softmax_log2e,
                    group_start,
                    token_count,
                )
                cute.experimental.iket.range_end(iket_25)
                iket_26 = cute.experimental.iket.range_start("rescale_route_o")
                rescale_o_for_next_acc(tOrO, row_scale)
                cute.experimental.iket.range_end(iket_26)
                iket_27 = cute.experimental.iket.range_start("mma_route_pv")
                gemm_pv_fp8(
                    tiled_mma_pv,
                    tOrO,
                    tSrS,
                    tScS,
                    sP,
                    tPrP,
                    tPsP,
                    tOrV,
                    tOsV_copy[None, None, None, V_consumer.index],
                    smem_copy_V,
                )
                cute.experimental.iket.range_end(iket_27)
            iket_28 = cute.experimental.iket.range_start("release_route_v")
            V_pipeline.consumer_release(V_consumer)
            cute.experimental.iket.range_end(iket_28)
            V_consumer.advance()

            if warp == 0 and exact_count > 0:
                first_exact = cutlass.Int32(route_indices[0])
                if cutlass.const_expr(not self.prefetch_first_exact_k):
                    iket_29 = cute.experimental.iket.range_start("wait_load_k")
                    KI_pipeline.producer_acquire(KI_producer)
                    cute.experimental.iket.range_end(iket_29)
                    iket_30 = cute.experimental.iket.range_start("issue_load_k")
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, first_exact],
                        tKsK[None, KI_producer.index],
                        tma_bar_ptr=KI_pipeline.producer_get_barrier(KI_producer),
                    )
                    cute.experimental.iket.range_end(iket_30)
                    KI_pipeline.producer_commit(KI_producer)
                    KI_producer.advance()
                iket_31 = cute.experimental.iket.range_start("wait_load_v")
                V_pipeline.producer_acquire(V_producer)
                cute.experimental.iket.range_end(iket_31)
                iket_32 = cute.experimental.iket.range_start("issue_load_v")
                cute.copy(
                    tma_atom_V,
                    tVgV[None, first_exact],
                    tVsV[None, V_producer.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer),
                )
                cute.experimental.iket.range_end(iket_32)
                V_pipeline.producer_commit(V_producer)
                V_producer.advance()

            for ordinal in cutlass.range(0, exact_count, 1, unroll=1):
                exact_block = cutlass.Int32(route_indices[ordinal])
                iket_33 = cute.experimental.iket.range_start("wait_exact_k")
                k_wait = KI_pipeline.consumer_try_wait(KI_consumer)
                KI_pipeline.consumer_wait(KI_consumer, k_wait)
                cute.experimental.iket.range_end(iket_33)
                iket_34 = cute.experimental.iket.range_start("mma_exact_qk")
                gemm_qk_i8(
                    tSI,
                    tQIrQ,
                    tKIrK,
                    tKIsK_copy[None, None, None, KI_consumer.index],
                    copy_KI,
                )
                combined_scale = query_scale * cutlass.Float32(mKScale[batch_idx, head_idx, exact_block * 4 + lane % 4])
                integer_scores_mn = layout_utils.reshape_acc_to_mn(tSI)
                exact_scores_mn = layout_utils.reshape_acc_to_mn(tSrS)
                iket_q_correction = cute.experimental.iket.range_start("load_q_mean_correction")
                # Both accumulator rows owned by a lane share the same columns.
                # Load each global-mean correction once and reuse across rows.
                # INT8 MMA accumulates INT32, so restore it after dequantization.
                for c in cutlass.range_constexpr(cute.size(exact_scores_mn, mode=[1])):
                    _, col = coordinates_mn[0, c]
                    key_index = exact_block * cutlass.Int32(N) + col
                    correction = cutlass.Float32(0.0)
                    if key_index < token_count:
                        correction = mQCorrection[batch_idx, head_idx, 0, key_index]
                    for r in cutlass.range_constexpr(2):
                        row_correction = cutlass.Float32(0.0)
                        if query_group[r] >= 0:
                            row_correction = correction
                        exact_scores_mn[r, c] = (
                            cutlass.Float32(integer_scores_mn[r, c]) * combined_scale + row_bias[r] + row_correction
                        )
                cute.experimental.iket.range_end(iket_q_correction)
                if cutlass.const_expr(self.debug_qk):
                    for i in cutlass.range_constexpr(cute.size(tSrS)):
                        row, col = tScS[i]
                        if ordinal == 0 and col == 0 and row < q_len:
                            mLSE_slice[q_start + row] = cutlass.Float32(tSI[i])
                cute.experimental.iket.range_end(iket_34)
                iket_35 = cute.experimental.iket.range_start("release_exact_k")
                KI_pipeline.consumer_release(KI_consumer)
                cute.experimental.iket.range_end(iket_35)
                KI_consumer.advance()
                # KC uses a different transaction barrier on the same bytes.
                # All warps must finish reading INT8 K before the KC prefetch.
                cute.arch.sync_threads()
                next_ordinal = ordinal + cutlass.Int32(1)
                if warp == 0:
                    if next_ordinal < exact_count:
                        next_exact = cutlass.Int32(route_indices[next_ordinal])
                        iket_36 = cute.experimental.iket.range_start("wait_load_k")
                        KI_pipeline.producer_acquire(KI_producer)
                        cute.experimental.iket.range_end(iket_36)
                        iket_37 = cute.experimental.iket.range_start("issue_load_k")
                        cute.copy(
                            tma_atom_K,
                            tKgK[None, next_exact],
                            tKsK[None, KI_producer.index],
                            tma_bar_ptr=KI_pipeline.producer_get_barrier(KI_producer),
                        )
                        cute.experimental.iket.range_end(iket_37)
                        KI_pipeline.producer_commit(KI_producer)
                        KI_producer.advance()
                    else:
                        if cutlass.const_expr(self.prefetch_next_route_k):
                            next_route_group = route_group + cutlass.Int32(1)
                            if next_route_group < num_route_groups:
                                # Reuse the K stage released by the final
                                # exact QK.  The next outer prologue supplies
                                # VC, matching the SM90 P19 partial handoff.
                                iket_38 = cute.experimental.iket.range_start("wait_load_k")
                                K_pipeline.producer_acquire(K_producer)
                                cute.experimental.iket.range_end(iket_38)
                                iket_39 = cute.experimental.iket.range_start("issue_load_kc")
                                cute.copy(
                                    tma_atom_KC,
                                    tKCgKC[None, next_route_group],
                                    tKCsK[None, K_producer.index],
                                    tma_bar_ptr=(K_pipeline.producer_get_barrier(K_producer)),
                                )
                                cute.experimental.iket.range_end(iket_39)
                                K_pipeline.producer_commit(K_producer)
                                K_producer.advance()
                # Interior tiles have no invalid logits. Only the final partial
                # KV tile needs column masking; a partial query CTA still needs
                # row masking for every exact tile that it processes.
                if q_len < M or (exact_block + cutlass.Int32(1)) * N > token_count:
                    block_len = token_count - exact_block * cutlass.Int32(N)
                    if block_len > N:
                        block_len = cutlass.Int32(N)
                    iket_40 = cute.experimental.iket.range_start("exact_mask")
                    mask_exact_scores(tSrS, tScS, block_len, q_len)
                    cute.experimental.iket.range_end(iket_40)
                iket_41 = cute.experimental.iket.range_start("softmax_exact")
                row_scale = online_softmax(tSrS, max_m, sum_m, scale_softmax_log2e)
                cute.experimental.iket.range_end(iket_41)
                iket_42 = cute.experimental.iket.range_start("rescale_exact_o")
                rescale_o_for_next_acc(tOrO, row_scale)
                cute.experimental.iket.range_end(iket_42)

                iket_43 = cute.experimental.iket.range_start("wait_exact_v")
                v_wait = V_pipeline.consumer_try_wait(V_consumer)
                V_pipeline.consumer_wait(V_consumer, v_wait)
                cute.experimental.iket.range_end(iket_43)
                iket_44 = cute.experimental.iket.range_start("mma_exact_pv")
                gemm_pv_fp8(
                    tiled_mma_pv,
                    tOrO,
                    tSrS,
                    tScS,
                    sP,
                    tPrP,
                    tPsP,
                    tOrV,
                    tOsV_copy[None, None, None, V_consumer.index],
                    smem_copy_V,
                )
                cute.experimental.iket.range_end(iket_44)
                iket_45 = cute.experimental.iket.range_start("release_exact_v")
                V_pipeline.consumer_release(V_consumer)
                cute.experimental.iket.range_end(iket_45)
                V_consumer.advance()
                if warp == 0 and next_ordinal < exact_count:
                    next_exact = cutlass.Int32(route_indices[next_ordinal])
                    iket_46 = cute.experimental.iket.range_start("wait_load_v")
                    V_pipeline.producer_acquire(V_producer)
                    cute.experimental.iket.range_end(iket_46)
                    iket_47 = cute.experimental.iket.range_start("issue_load_v")
                    cute.copy(
                        tma_atom_V,
                        tVgV[None, next_exact],
                        tVsV[None, V_producer.index],
                        tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer),
                    )
                    cute.experimental.iket.range_end(iket_47)
                    V_pipeline.producer_commit(V_producer)
                    V_producer.advance()

        iket_48 = cute.experimental.iket.range_start("softmax_finalize")
        final_ratio, lse = finalize_softmax(max_m, sum_m, scale_softmax_log2e)
        cute.experimental.iket.range_end(iket_48)
        iket_49 = cute.experimental.iket.range_start("rescale_setup_o")
        final_ratio.store(final_ratio.load() * (value_scale / P_SCALE))
        rescale_o_for_next_acc(tOrO, final_ratio)
        cute.experimental.iket.range_end(iket_49)
        if cutlass.const_expr(not self.debug_route_trace and not self.debug_qk):
            tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
            for m in cutlass.range_constexpr(cute.size(lse)):
                row = tScS_mn[m, 0][0]
                if tScS_mn[m, 0][1] == 0 and row < q_len:
                    mLSE_slice[q_start + row] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))
        sO = storage.Q_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
        tiled_copy_O = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.O_layout.is_m_major_c(), 4),
                self.O_dtype,
            ),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_O.retile(tOrO_cvt)
        tOsO = tiled_copy_O.get_slice(tidx).partition_D(sO)
        cute.copy(tiled_copy_O, tOrO_cv, tOsO)
        cute.arch.fence_view_async_shared()
        iket_50 = cute.experimental.iket.range_start("wait_setup_cta")
        cute.arch.sync_threads()
        cute.experimental.iket.range_end(iket_50)
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            *cta_coord_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        if warp == 0:
            iket_51 = cute.experimental.iket.range_start("issue_store_o")
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.experimental.iket.range_end(iket_51)
            cute.arch.cp_async_bulk_commit_group()
            iket_52 = cute.experimental.iket.range_start("wait_store_o")
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.experimental.iket.range_end(iket_52)

        cute.experimental.iket.mark("kernel_end")

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        threshold: cute.Tensor,
        lse: cute.Tensor,
        v_scale: cute.Tensor,
        q_int8: cute.Tensor,
        k_int8: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        q_bias: cute.Tensor,
        q_correction: cute.Tensor,
        q_group: cute.Tensor,
        softmax_scale: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        q_mkl, k_nkl, kc_nkl = [layout_utils.select(t, [1, 3, 2, 0]) for t in (q, k, kc)]
        qi_mkl, ki_nkl = [layout_utils.select(t, [1, 3, 2, 0]) for t in (q_int8, k_int8)]
        v_nkl, vc_nkl = [layout_utils.select(t, [3, 1, 2, 0]) for t in (v, vc)]
        o_mkl = layout_utils.select(o, [1, 3, 2, 0])
        if cutlass.const_expr(self.debug_route_trace):
            lse_target = lse
        else:
            lse_target = layout_utils.select(lse, [1, 2, 0])

        self.Q_dtype = q_mkl.element_type
        self.K_dtype = k_nkl.element_type
        self.V_dtype = v_nkl.element_type
        self.O_dtype = o_mkl.element_type
        self.Q_layout = utils.LayoutEnum.from_tensor(q_mkl)
        self.K_layout = utils.LayoutEnum.from_tensor(k_nkl)
        self.V_layout = utils.LayoutEnum.from_tensor(v_nkl)
        self.O_layout = utils.LayoutEnum.from_tensor(o_mkl)
        assert self.Q_dtype == cutlass.BFloat16
        assert self.K_dtype == cutlass.BFloat16
        assert self.V_dtype == cutlass.Float8E4M3FN
        assert vc_nkl.element_type == self.V_dtype
        assert self.O_dtype == cutlass.BFloat16
        # V/VC are logically BTHD but physically B,H,D,T_padded, so the PV
        # reduction dimension is contiguous. TMA and ldmatrix use K_SW64.
        assert self.V_layout.is_k_major_b()
        assert utils.LayoutEnum.from_tensor(vc_nkl).is_k_major_b()

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            self.Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            self.K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.kv_stage,
        )
        self.QI_smem_layout = sm90_utils.make_smem_layout_a(
            self.Q_layout, self.tile_shape_qk, cutlass.Int8, self.q_stage
        )
        self.KI_smem_layout = sm90_utils.make_smem_layout_b(
            self.K_layout, self.tile_shape_qk, cutlass.Int8, self.kv_stage
        )
        self.V_smem_layout = sm90_utils.make_smem_layout_b(
            self.V_layout,
            self.tile_shape_pv,
            self.V_dtype,
            self.kv_stage,
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            self.O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(O_smem_layout_staged, mode=[0, 1])

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            KI_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            Q_smem: cute.struct.Align[
                cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)],
                128,
            ]
            K_smem: cute.struct.Align[
                cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)],
                128,
            ]
            V_smem: cute.struct.Align[
                cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)],
                128,
            ]

        self.shared_storage_t = SharedStorage

        tiled_mma_qk = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.Q_dtype,
                self.acc_dtype,
                (16, 8, 16),
            ),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(64, 16, 16),
        )
        tiled_mma_i8 = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (16, 8, 32)),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(64, 16, 32),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaFP8Op(
                self.V_dtype,
                self.acc_dtype,
                (16, 8, 32),
            ),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(64, 16, 32),
        )

        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            q_mkl,
            self.Q_smem_layout,
            (M, D),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            ki_nkl,
            self.KI_smem_layout,
            (N, D),
            num_multicast=1,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            v_nkl,
            self.V_smem_layout,
            (DV, N),
            num_multicast=1,
        )
        tma_atom_KC, tma_tensor_KC = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            kc_nkl,
            self.K_smem_layout,
            (N, D),
            num_multicast=1,
        )
        tma_atom_VC, tma_tensor_VC = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            vc_nkl,
            self.V_smem_layout,
            (DV, N),
            num_multicast=1,
        )
        s2g_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            s2g_op,
            o_mkl,
            self.O_smem_layout,
            (M, DV),
            num_multicast=1,
        )

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            tma_tensor_KC,
            tma_tensor_VC,
            threshold,
            lse_target,
            v_scale,
            qi_mkl,
            q_scale,
            k_scale,
            q_bias,
            q_correction,
            q_group,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            tma_atom_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_i8,
            self.QI_smem_layout,
            self.KI_smem_layout,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.O_smem_layout,
            softmax_scale * 1.4426950408889634,
            sink_start_block,
            sink_end_block,
        ).launch(
            grid=(cute.ceil_div(q_mkl.shape[0], M), q_mkl.shape[2], q_mkl.shape[3]),
            block=(self.num_threads, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage_t.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )


@cute.jit
# def gemm_pv_fp8_KEEP_DONT_REMOVE(
#     tiled_mma: cute.TiledMma,
#     acc: cute.Tensor,
#     probabilities: cute.Tensor,
#     coordinates: cute.Tensor,
#     sP: cute.Tensor,
#     tPrP: cute.Tensor,
#     tPsP: cute.Tensor,
#     tOrV: cute.Tensor,
#     tOsV: cute.Tensor,
#     smem_copy_V: cute.TiledCopy,
# ):
#     # Deliberately simple layout conversion: FP32 accumulator -> shared FP8 ->
#     # FP8 MMA A registers. This initial implementation is not tuned.
#     for i in cutlass.range_constexpr(cute.size(probabilities)):
#         row, col = coordinates[i]
#         sP[row, col] = (cutlass.Float32(probabilities[i]) * P_SCALE).to(
#             cutlass.Float8E4M3FN
#         )
#     cute.arch.sync_threads()
#     cute.copy(
#         cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN),
#         tPsP,
#         tPrP,
#     )
#     cute.arch.sync_threads()
#     gemm_rs_smem(tiled_mma, acc, tPrP, tOrV, tOsV, smem_copy_V)

# def gemm_pv_fp8_PACKED_DONT_REMOVE(
#     tiled_mma: cute.TiledMma,
#     acc: cute.Tensor,
#     probabilities: cute.Tensor,
#     coordinates: cute.Tensor,
#     sP: cute.Tensor,
#     tPrP: cute.Tensor,
#     tPsP: cute.Tensor,
#     tOrV: cute.Tensor,
#     tOsV: cute.Tensor,
#     smem_copy_V: cute.TiledCopy,
# ):
#     from cutlass._mlir.dialects import llvm
#     from cutlass.cutlass_dsl import T

#     # Adjacent accumulator entries own the same row and consecutive columns
#     # (an even column followed by the odd one). Convert both real values with
#     # one instruction and store their bits together; keep the shared handoff.
#     assert cute.size(probabilities) % 2 == 0
#     sP_pairs = cute.recast_tensor(sP, cutlass.Uint16)
#     for i in cutlass.range_constexpr(cute.size(probabilities) // 2):
#         row, col = coordinates[2 * i]
#         p0 = cutlass.Float32(probabilities[2 * i]) * P_SCALE
#         p1 = cutlass.Float32(probabilities[2 * i + 1]) * P_SCALE
#         packed = llvm.inline_asm(
#             T.i16(),
#             [p0.ir_value(), p1.ir_value()],
#             "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
#             "=h,f,f",
#             has_side_effects=False,
#             is_align_stack=False,
#         )
#         sP_pairs[row, col // 2] = cutlass.Uint16(packed)
#     cute.arch.sync_threads()
#     cute.copy(
#         cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN),
#         tPsP,
#         tPrP,
#     )
#     cute.arch.sync_threads()
#     gemm_rs_smem(tiled_mma, acc, tPrP, tOrV, tOsV, smem_copy_V)

def gemm_pv_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    probabilities: cute.Tensor,
    coordinates: cute.Tensor,
    sP: cute.Tensor,
    tPrP: cute.Tensor,
    tPsP: cute.Tensor,
    tOrV: cute.Tensor,
    tOsV: cute.Tensor,
    smem_copy_V: cute.TiledCopy,
):
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    # Adjacent accumulator entries own the same row and consecutive columns
    # (an even column followed by the odd one). Convert both real values with
    # one instruction and store their bits together; keep the shared handoff.
    assert cute.size(probabilities) % 2 == 0
    sP_pairs = cute.recast_tensor(sP, cutlass.Uint16)
    for i in cutlass.range_constexpr(cute.size(probabilities) // 2):
        row, col = coordinates[2 * i]
        p0 = cutlass.Float32(probabilities[2 * i]) * P_SCALE
        p1 = cutlass.Float32(probabilities[2 * i + 1]) * P_SCALE
        packed = llvm.inline_asm(
            T.i16(),
            [p0.ir_value(), p1.ir_value()],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
        sP_pairs[row, col // 2] = cutlass.Uint16(packed)
    cute.arch.sync_threads()
    # P is K-contiguous in shared memory. Load packed FP8 byte pairs directly
    # into the MMA A layout using a warp-collective, non-transposed matrix load.
    copy_P = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float8E4M3FN,
        ),
        tiled_mma,
    )
    thread_copy_P = copy_P.get_slice(cute.arch.thread_idx()[0])
    cute.copy(copy_P, thread_copy_P.partition_S(sP), thread_copy_P.retile(tPrP))
    cute.arch.sync_threads()
    gemm_rs_smem(tiled_mma, acc, tPrP, tOrV, tOsV, smem_copy_V)


def quantize_values(v, vc):
    """Return token-contiguous E4M3 V/VC and common FP32 scale[B,H].

    Kept outside the attention kernel and capture range. Summaries must be
    computed from the original BF16 V, then quantized with the same head scale.
    Returned BTHD views use physical B,H,D,T_padded storage, with T padded to
    a multiple of 16 for TMA stride alignment. Use to_cute_value_tensor for
    these noncontiguous views; do not call contiguous() on them.
    """
    import torch

    if v.dtype != torch.bfloat16 or vc.dtype != torch.bfloat16:
        raise TypeError("V and pooled VC must be BF16 before quantization")
    if v.ndim != 4 or vc.ndim != 4 or v.shape[0:1] + v.shape[2:] != vc.shape[0:1] + vc.shape[2:]:
        raise ValueError("V/VC must have matching B,H,D dimensions")
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = (
        torch.maximum(
            v.abs().amax(dim=(1, 3)).float(),
            vc.abs().amax(dim=(1, 3)).float(),
        ).clamp_min(1e-12)
        / limit
    )
    broadcast = scale[:, None, :, None]

    def pack_values(values):
        quantized = (values.float() / broadcast).clamp(-limit, limit).to(torch.float8_e4m3fn)
        batch, tokens, heads, channels = values.shape
        storage = torch.zeros(
            (batch, heads, channels, (tokens + 15) // 16 * 16),
            device=values.device,
            dtype=torch.float8_e4m3fn,
        )
        storage[:, :, :, :tokens].copy_(quantized.permute(0, 2, 3, 1))
        return storage[:, :, :, :tokens].permute(0, 3, 1, 2)

    return (
        pack_values(v),
        pack_values(vc),
        scale.contiguous(),
    )


def to_cute_value_tensor(value):
    """Preserve the token-contiguous FP8 V/VC layout in the CuTe descriptor."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    if value.ndim != 4 or value.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected a BTHD E4M3 V/VC tensor")
    if value.stride(1) != 1 or any(value.stride(i) % 16 for i in (0, 2, 3)):
        raise ValueError("Expected token stride 1 and 16-byte-aligned outer strides")
    return from_dlpack(value, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)


def quantize_qk(q, k, kc, *, text_tokens=0, smooth_q=True):
    """Return centered KC and (QI,KI,QS,KS,K-bias,Q-correction,Q-group).

    Q scales group rows r/r+8 within each 16-row warp tile. K scales group
    (2*g,2*g+1)+8*j within each 64-row tile. Scales are compact [B,H,groups],
    with 32 Q scales and 4 K scales per 64-token tile. Padded groups use zeros.
    K mean and row correction are FP32, using all real K tokens (text + video).
    Original KC must be used to compute routing thresholds before this call.
    Q means are FP32 per batch/head over ALL original queries, excluding
    text-prefix query zeros. The FP32 correction [B,H,1,T_padded64] is
    computed once, outside the
    attention kernel. smooth_q=False is a K-smoothing-only numerical control.
    """
    import torch
    import torch.nn.functional as F

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or kc.dtype != torch.bfloat16:
        raise TypeError("Q/K/KC must be BF16")
    if q.ndim != 4 or q.shape != k.shape or q.shape[-1] != D:
        raise ValueError("Q/K must have matching BTHD shapes with D128")
    batch, tokens, heads, _ = q.shape
    if not isinstance(text_tokens, int) or not 0 <= text_tokens < tokens:
        raise ValueError("text_tokens must be an integer in [0,T)")
    if kc.shape != (batch, (tokens + 63) // 64, heads, D):
        raise ValueError("KC must contain one mean vector per 64-token K block")
    mean = k.float().mean(1, keepdim=True)
    bias = (q.float() * mean).sum(-1).contiguous()
    centered_kc = (kc.float() - mean).to(torch.bfloat16).contiguous()
    padded = (tokens + 63) // 64 * 64
    centered_k = (k.float() - mean).contiguous()
    indices = torch.arange(tokens, device=q.device)
    real_query = indices >= text_tokens
    if smooth_q:
        groups = 1
        query = q[:, text_tokens:].permute(0, 2, 1, 3).float()
        q_mean = query.mean(2, keepdim=True).contiguous()
        group_ids = torch.zeros_like(indices)
        mean_rows = q_mean.permute(0, 2, 1, 3)
        centered_q = q.float() - torch.where(real_query[None, :, None, None], mean_rows, 0.0)
        correction = torch.empty((batch, heads, groups, padded), device=q.device, dtype=torch.float32)
        _precompute_q_correction[(triton.cdiv(groups, 16), triton.cdiv(padded, 64), batch * heads)](
            q_mean,
            centered_k,
            correction,
            tokens,
            heads,
            groups,
            padded,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        centered_q = q.float()
        group_ids = torch.zeros_like(indices)
        correction = torch.zeros((batch, heads, 1, padded), device=q.device, dtype=torch.float32)
    q_group = torch.where(real_query, group_ids, -1).to(torch.int32).contiguous()

    def quantize(values, is_query):
        x = F.pad(values.permute(0, 2, 1, 3), (0, 0, 0, padded - tokens))
        if is_query:
            groups = x.reshape(batch, heads, padded // 16, 2, 8, D)
            scale = groups.abs().amax(dim=(3, 5)).clamp_min(1e-30) / 127
            scaled = groups / scale[:, :, :, None, :, None]
        else:
            groups = x.reshape(batch, heads, padded // 64, 8, 4, 2, D)
            scale = groups.abs().amax(dim=(3, 5, 6)).clamp_min(1e-30) / 127
            scaled = groups / scale[:, :, :, None, :, None, None]
        quantized = scaled.round().clamp(-127, 127).to(torch.int8)
        output = quantized.reshape(batch, heads, padded, D)[:, :, :tokens].permute(0, 2, 1, 3).contiguous()
        return output, scale.reshape(batch, heads, -1).contiguous()

    qi, qs = quantize(centered_q, True)
    ki, ks = quantize(centered_k, False)
    return centered_kc, (qi, ki, qs, ks, bias, correction, q_group)


def make_kernel(**kwargs):
    return SolAttnForwardSm120INT8QKFP8PV(**kwargs)


__all__ = [
    "SolAttnForwardSm120INT8QKFP8PV",
    "make_kernel",
    "quantize_qk",
    "quantize_values",
    "to_cute_value_tensor",
]
