# SM100 Sol-Attn IKET trace

The trace uses warp 0 for MMA issue, warps 1–4 for score routing and softmax,
and warp 5 for TMA loads. A range measures time between its two IKET events on
that warp. TMA and MMA issue ranges measure submission, not completion; the
associated waits and releases expose completion dependencies. Profiling adds
overhead, so use the trace to locate stalls and compare phases rather than as
an uninstrumented kernel latency measurement.

## Capture

`run-iket` is included with `nvidia-cutlass-dsl[cu13]==4.8.0`; the 4.7.0
installation used for the original benchmark has no `run-iket` executable.

```bash
uv pip install --python .venv/bin/python 'nvidia-cutlass-dsl[cu13]==4.8.0'
IKET_CMD="$PWD/.venv/bin/run-iket" .venv/bin/python \
  techniques/sparse_backends/sol_attn/sm100/bench.py \
  --tokens 1024 --heads 1 --iket --warmup 0 --reps 1 \
  --iket-cluster 0,0,0 \
  --iket-output techniques/sparse_backends/sol_attn/sm100/iket_profile_1024x1024_next
```

This profiles one 1024 × 1024 attention matrix with `[B,T,H,D] = [1,1024,1,128]`
and `tau=1.0`. `--iket-cluster 0,0,0` records one query CTA while the full
16-CTA grid executes. The output directory contains a `*.pftrace` timeline for
Perfetto and a `*.trace.json` file for programmatic analysis. Use a new output
directory for each capture. The benchmark also accepts `--tokens`, `--heads`,
and `--tau` to change the workload. Without these overrides it defaults to the
Wan2.1-T2V-14B shape `[1,75600,40,128]`.

The generated `*.html` viewer embeds the remote Perfetto UI and can show a
blank page when opened in a preview without access to that site. Open the
`*.pftrace` directly in Perfetto, or create an offline SVG from the JSON:

```bash
.venv/bin/python techniques/sparse_backends/sol_attn/sm100/render_iket.py \
  techniques/sparse_backends/sol_attn/sm100/iket_profile_1024x1024/*.trace.json
```

The SVG shows all six warp lanes and hover details for each range. Pass `--png`
to also write a PNG preview if Pillow is installed. The captured
`iket_profile_1024x1024` directory already contains both previews.

## Range guide

| Warp | Range prefix or name | What it brackets |
| --- | --- | --- |
| 5 | `wait_load_*` | Pipeline producer acquire: time before a K, V, KC, VC, or Q stage can be reused. |
| 5 | `issue_load_*` | TMA copy submission and producer state advance. |
| 5 | `load_*` | The enclosing acquire plus issue interval. |
| 0 | `mma_wait_*` | K/V consumer wait, score producer acquire, or probability-ready barrier. |
| 0 | `mma_issue_*` | QK or PV GEMM submission and completion-barrier commit where applicable. |
| 0 | `mma_release_*` | K/V stage release after MMA issue; this can include completion-backed waiting. |
| 1–4 | `owner_route_wait_score`, `owner_route_load_score`, `owner_route_wait_load` | Wait for route QK score, load TMEM score to registers, then wait until all owners have loaded. |
| 1–4 | `owner_route_reduce`, `owner_route_wait_reduce` | Reduce score columns into shared route partials, then rendezvous before selection. |
| 1 only | `owner_route_select` | Select exact columns, publish masks and compact exact indices. |
| 1–4 | `owner_route_wait_mask`, `owner_route_mask` | Wait for the selector's mask, then set exact and invalid pooled score columns to negative infinity. |
| 1–4 | `owner_route_softmax`, `owner_route_exp` | Update the online maximum and denominator using pooled scores; the nested exp range computes per-block exponentials. |
| 1–4 | `owner_route_rescale_o`, `owner_route_store_p` | Rescale the prior O accumulator if needed, then store pooled probabilities to TMEM. `owner_route_approx` encloses mask, softmax, and rescale. |
| 1 only | `owner_exact_mask` | Build valid-column gates for an exact K/V pair. |
| 1–4 | `owner_exact_wait_score`, `owner_exact_load_score`, `owner_exact_wait_load` | Wait for exact QK, load its TMEM score, then rendezvous before the S/P alias is reused. |
| 1–4 | `owner_exact_softmax`, `owner_exact_rescale_o`, `owner_exact_store_p` | Apply masks and online softmax, rescale O if needed, then store P to TMEM. |
| 1–4 | `owner_exact_wait_p_ready`, `owner_wait_final_o` | Publish P to warp 0, then wait for the last O completion before the epilogue. |
| 1–4 | `owner_publish_stats`, `owner_wait_stats`, `owner_load_o`, `owner_store_output` | Publish final normalization stats, synchronize owners, load O from TMEM, then normalize and store output and LSE. |
| 0–4 | `score_wait_packet` | Rendezvous after route mask/P generation, before route PV. |
| 0–5 | `group_wait_exact`, `cta_wait_tmem_free` | CTA joins before each exact group and before TMEM free. |

Payloads are **local to the query CTA**. Route KC/VC and route MMA ranges carry
`route_tile_idx`: `0` through `num_route_tiles - 1`. Exact K/V loads, exact MMA,
and exact owner ranges carry `logical_group_idx * 128 + pair_idx`; the 128 slots
per group keep IDs distinct across groups. These payloads identify work in the
trace; they are not global K/V block numbers. The selected CTA coordinate is
shown separately in the trace. All payload-bearing range ends repeat the
start payload, as required by IKET.

IKET's event timestamps have 32 ns granularity. This kernel has more than 30
unique event names, which increases instrumentation overhead; compare range
durations within the trace and use the ordinary benchmark for throughput.
