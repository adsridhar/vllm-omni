"""Render one run-iket JSON launch as an offline SVG/HTML warp timeline."""

import argparse
from html import escape
import json
from pathlib import Path


COLORS = {
    "wait": "#e9a23b",
    "issue": "#5a9df8",
    "release": "#bd83eb",
    "compute": "#54c9a3",
    "other": "#7e9bbd",
}


def category(name):
    if "wait" in name:
        return "wait"
    if "issue" in name or name.startswith("load_"):
        return "issue"
    if "release" in name:
        return "release"
    if name.startswith("owner_"):
        return "compute"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="run-iket *.trace.json")
    parser.add_argument("--output", type=Path, help="SVG output (default: timeline.svg beside trace)")
    parser.add_argument("--png", action="store_true", help="also render a PNG preview (requires Pillow)")
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    launch = trace["launches"][0]
    names = trace["stringTable"]
    locations = trace["locationTable"]
    lifetime = launch["warpLifetimes"]
    start = min(x["startTs"] for x in lifetime)
    end = max(x["endTs"] for x in lifetime)
    span = end - start
    width, left, right = 1800, 150, 30
    plot_width = width - left - right
    row_height, lane_gap = 21, 12

    by_warp = {}
    for item in launch["ranges"]:
        warp = locations[item["warpLocIdxs"][0]]["warpId"]
        by_warp.setdefault(warp, []).append(item)

    placed = {}
    heights = {}
    for warp, items in by_warp.items():
        levels = []
        placed[warp] = []
        for item in sorted(items, key=lambda x: (x["startTs"], -x["endTs"])):
            level = next((i for i, free_at in enumerate(levels)
                          if item["startTs"] >= free_at), len(levels))
            if level == len(levels):
                levels.append(item["endTs"])
            else:
                levels[level] = item["endTs"]
            placed[warp].append((item, level))
        heights[warp] = max(1, len(levels)) * row_height + lane_gap

    top = 110
    total_height = top + sum(heights[w] for w in sorted(heights)) + 85
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#e6edf5} .small{font-size:12px} .title{font-size:22px;font-weight:bold} .label{font-size:13px;font-weight:bold}</style>',
        f'<rect width="{width}" height="{total_height}" fill="#111821"/>',
        '<text x="24" y="34" class="title">SM100 Sol-Attn · IKET CTA timeline</text>',
        f'<text x="24" y="56" class="small">CTA {escape(str(locations[0]["ctaId"]))} · {len(launch["ranges"])} ranges · {span / 1000:.3f} µs warp lifetime · hover a bar for details</text>',
    ]
    for name, color, xpos in zip(COLORS, COLORS.values(), range(24, 720, 135)):
        pieces += [f'<rect x="{xpos}" y="70" width="14" height="14" fill="{color}"/>',
                   f'<text x="{xpos + 20}" y="82" class="small">{name}</text>']
    for tick in range(9):
        x = left + tick * plot_width / 8
        pieces.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{total_height - 65}" stroke="#2d3c4b"/>')
        pieces.append(f'<text x="{x:.1f}" y="{total_height - 45}" class="small" text-anchor="middle">{tick * span / 8000:.2f}</text>')
    pieces.append(f'<text x="{left + plot_width / 2:.1f}" y="{total_height - 16}" class="small" text-anchor="middle">Time from first warp start (µs)</text>')

    y = top
    for warp in sorted(placed):
        h = heights[warp]
        pieces.append(f'<rect x="0" y="{y - 5}" width="{width}" height="{h}" fill="{("#172330" if warp % 2 == 0 else "#1b2936")}"/>')
        role = "MMA" if warp == 0 else "TMA" if warp == 5 else "owner"
        pieces.append(f'<text x="24" y="{y + 15}" class="label">warp {warp} · {role}</text>')
        for item, level in placed[warp]:
            name = names[item["rangeNameIdx"]]
            x = left + (item["startTs"] - start) * plot_width / span
            w = max(2, (item["endTs"] - item["startTs"]) * plot_width / span)
            yy = y + level * row_height
            payload = item["internalEvents"][0].get("payloadVal")
            detail = f'{name} | warp {warp} | {((item["startTs"] - start) / 1000):.3f}–{((item["endTs"] - start) / 1000):.3f} µs | duration {(item["endTs"] - item["startTs"]) / 1000:.3f} µs'
            if payload is not None:
                detail += f' | CTA-local index {payload}'
            pieces.append(f'<rect x="{x:.2f}" y="{yy}" width="{w:.2f}" height="17" rx="2" fill="{COLORS[category(name)]}" stroke="#101820"><title>{escape(detail)}</title></rect>')
            if w >= 90:
                pieces.append(f'<text x="{x + 4:.2f}" y="{yy + 12}" class="small" fill="#07131d" style="fill:#07131d" pointer-events="none">{escape(name)}</text>')
        y += h
    pieces.append('</svg>')
    output = args.output or args.trace.with_name("timeline.svg")
    output.write_text("\n".join(pieces) + "\n")
    print(output)
    if args.png:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (width, total_height), "#111821")
        draw = ImageDraw.Draw(image)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(font_path, 12)
            title_font = ImageFont.truetype(font_path, 22)
        except OSError:
            font = ImageFont.load_default()
            title_font = font
        draw.text((24, 12), "SM100 Sol-Attn · IKET CTA timeline", fill="#e6edf5", font=title_font)
        draw.text((24, 43), f"CTA {locations[0]['ctaId']} · {len(launch['ranges'])} ranges · {span / 1000:.3f} µs warp lifetime", fill="#e6edf5", font=font)
        for name, color, xpos in zip(COLORS, COLORS.values(), range(24, 720, 135)):
            draw.rectangle((xpos, 70, xpos + 14, 84), fill=color)
            draw.text((xpos + 20, 69), name, fill="#e6edf5", font=font)
        y = top
        for warp in sorted(placed):
            h = heights[warp]
            draw.rectangle((0, y - 5, width, y + h - 5), fill="#172330" if warp % 2 == 0 else "#1b2936")
            for tick in range(9):
                x = left + tick * plot_width / 8
                draw.line((x, y - 5, x, y + h - 5), fill="#2d3c4b")
            role = "MMA" if warp == 0 else "TMA" if warp == 5 else "owner"
            draw.text((24, y), f"warp {warp} · {role}", fill="#e6edf5", font=font)
            for item, level in placed[warp]:
                name = names[item["rangeNameIdx"]]
                x = left + (item["startTs"] - start) * plot_width / span
                w = max(2, (item["endTs"] - item["startTs"]) * plot_width / span)
                yy = y + level * row_height
                draw.rectangle((x, yy, x + w, yy + 17), fill=COLORS[category(name)], outline="#101820")
                if w >= 90:
                    draw.text((x + 4, yy + 1), name, fill="#07131d", font=font)
            y += h
        for tick in range(9):
            x = left + tick * plot_width / 8
            draw.text((x - 18, total_height - 52), f"{tick * span / 8000:.2f}", fill="#e6edf5", font=font)
        draw.text((left + plot_width / 2 - 135, total_height - 24), "Time from first warp start (µs)", fill="#e6edf5", font=font)
        png_output = output.with_suffix(".png")
        image.save(png_output)
        print(png_output)


if __name__ == "__main__":
    main()
