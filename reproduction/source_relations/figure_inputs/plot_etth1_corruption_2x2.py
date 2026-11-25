"""Combine four ETTh1 corruption curves into a compact 2x2 figure with a shared bottom legend."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "Figs"
FIGS_NEW = FIGS / "new"

PANELS = [
    ("96mask.png", "(a)"),
    ("96spike.png", "(b)"),
    ("720mask.png", "(c)"),
    ("720spike.png", "(d)"),
]

LEGEND = [
    ("#ff7f0e", "Flatten + MSE"),
    ("#d62728", "Flatten + scaled log-cosh"),
    ("#1f77b4", "SAR + MSE"),
    ("#2ca02c", "SAR + scaled log-cosh"),
]

GAP_X = 10
GAP_Y = 24
LEGEND_H = 58
LEGEND_GAP = 40
LEGEND_FONT = 7


def trim_whitespace(img: Image.Image, tol: int = 252) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    mask = (arr < tol).any(axis=2)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def draw_panel_tag(img: Image.Image, tag: str) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), tag, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 10, 8
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=(255, 255, 255))
    draw.text((x, y), tag, fill=(0, 0, 0), font=font)
    return out


def render_legend(width: int) -> Image.Image:
    dpi = 150
    fig_w = width / dpi
    fig_h = LEGEND_H / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.6, markersize=4.5, label=label)
        for color, label in LEGEND
    ]
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        fancybox=False,
        fontsize=LEGEND_FONT,
        handlelength=1.5,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.35,
        edgecolor="#999999",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.9)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return trim_whitespace(Image.fromarray(buf).convert("RGB"))


def main() -> None:
    panels = []
    for fname, tag in PANELS:
        img = trim_whitespace(Image.open(FIGS_NEW / fname))
        panels.append(draw_panel_tag(img, tag))

    w = max(panels[0].width, panels[1].width, panels[2].width, panels[3].width)
    h = max(panels[0].height, panels[1].height, panels[2].height, panels[3].height)

    def fit(img: Image.Image) -> Image.Image:
        if img.size == (w, h):
            return img
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        return canvas

    panels = [fit(p) for p in panels]

    grid_w = 2 * w + GAP_X
    grid_h = 2 * h + GAP_Y
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    grid.paste(panels[0], (0, 0))
    grid.paste(panels[1], (w + GAP_X, 0))
    grid.paste(panels[2], (0, h + GAP_Y))
    grid.paste(panels[3], (w + GAP_X, h + GAP_Y))

    legend = render_legend(grid_w)
    legend_gap = LEGEND_GAP
    out = Image.new("RGB", (grid_w, grid_h + legend_gap + legend.height), (255, 255, 255))
    out.paste(grid, (0, 0))
    out.paste(legend, ((grid_w - legend.width) // 2, grid_h + legend_gap))

    out_path = FIGS / "ETTh1_corruption_2x2.png"
    out.save(out_path, dpi=(300, 300))
    print(f"Saved {out_path} ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
