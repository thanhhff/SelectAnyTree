"""
Generate a 360° turntable GIF for SelectAnyTree project page.

One seamless loop = one full 360° rotation.
The 5 click states are revealed progressively: each click gets an equal
slice of the rotation (~72°), so the segmentation builds up as the scene spins.
No axes, no grid — just the point cloud on a dark background.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image, ImageDraw, ImageFont
import io

DUMPS = "/workdir/radish/3d-project/workspace/SelectAnyTree-Dev/viz/dumps"
OUT   = "/workdir/radish/3d-project/workspace/SelectAnyTree-Public/static/images"
os.makedirs(OUT, exist_ok=True)

C_TP = np.array([0.18, 0.63, 0.24, 0.92])
C_FP = np.array([0.88, 0.20, 0.20, 0.92])
C_FN = np.array([0.18, 0.42, 0.92, 0.92])
C_BG = np.array([0.80, 0.80, 0.80, 0.12])

DPI        = 110
TARGET_PTS = 80_000
FRAMES_PER_CLICK = 10   # 5 clicks × 10 frames = 50 frames total = ~7 s loop
ELEV       = 30         # camera elevation (degrees)
FIG_SIZE   = (5.0, 5.0)
BG_COLOR   = "#0d1a0d"
BG_RGB     = (13, 26, 13)
LABEL_COLOR = (200, 230, 200)   # soft green for overlay text


def height_rgba(z, alpha=0.16):
    z = np.asarray(z, float)
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    r = np.clip(1.5 * t,       0, 1)
    g = np.clip(0.45 + 0.55*t, 0, 1)
    b = np.clip(0.40 - 0.40*t, 0, 1)
    a = np.full_like(t, alpha)
    return np.stack([r, g, b, a], axis=1)


def error_rgba(pred, gt, pts_z):
    col = height_rgba(pts_z, alpha=0.11)
    col[gt & pred]  = C_TP
    col[~gt & pred] = C_FP
    col[gt & ~pred] = C_FN
    return col


def iou(pred, gt):
    return np.logical_and(pred, gt).sum() / max(1, np.logical_or(pred, gt).sum())


def render_3d_frame(pts, col, clicks, azim, elev, xlim, ylim, zlim):
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=col, s=0.4, linewidths=0, rasterized=True, depthshade=False)

    for c in (clicks or []):
        xyz, lab = c[:3], int(c[3])
        kind = c[5] if len(c) > 5 else ("pos" if lab == 1 else "neg")
        if kind == "chm":
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="^", s=130,
                       c="darkorange", edgecolors="white", linewidths=0.9,
                       zorder=8, depthshade=False)
        elif lab == 1:
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="*", s=200,
                       c="yellow", edgecolors="black", linewidths=0.9,
                       zorder=8, depthshade=False)
        else:
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="X", s=110,
                       c="red", edgecolors="white", linewidths=0.9,
                       zorder=8, depthshade=False)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)

    # ── remove ALL axes, grid, panes ──────────────────────────────────────────
    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.05,
                facecolor=BG_COLOR)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    return img


def add_overlay(img, click_num, iou_val):
    """Burn a minimal text label into the bottom-left of the frame."""
    draw = ImageDraw.Draw(img)
    W, H = img.size
    text = f"Click {click_num}  |  IoU = {iou_val:.2f}"
    # use default PIL font — no external font needed
    draw.text((12, H - 22), text, fill=LABEL_COLOR)
    return img


def make_360_gif(npz_path, out_path, tree_idx=0, margin=1.5,
                 frames_per_click=FRAMES_PER_CLICK, zoom=1.25):
    d = np.load(npz_path, allow_pickle=True)
    pts, gt, preds = d["points"], d["gt"], d["preds"]
    clicks = d["clicks"].tolist() if "clicks" in d.files else None
    ti     = tree_idx
    K      = preds.shape[0]

    # crop window around target tree
    tpts = pts[gt[ti]]
    cen  = tpts[:, :2].mean(0)
    half = max(tpts[:, :2].max(0) - tpts[:, :2].min(0)) / 2
    half = max(half, 4.0) + margin
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    xlim = (max(cen[0]-half, xmin), min(cen[0]+half, xmax))
    ylim = (max(cen[1]-half, ymin), min(cen[1]+half, ymax))

    mask   = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
              (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
    pts_c  = pts[mask]
    gt_c   = gt[ti][mask]
    pred_c = preds[:, ti][:, mask]

    zlo, zhi = pts_c[:, 2].min(), pts_c[:, 2].max()
    zpad = 0.08 * (zhi - zlo) + 0.5
    zlim = (zlo - zpad, zhi + zpad)

    # equalise axis ranges → no distortion; center on the TREE (not the crop
    # window) and divide by `zoom` so the tree fills more of the frame.
    tcen = tpts.mean(0)
    xc, yc = float(tcen[0]), float(tcen[1])
    zc = 0.5 * (zlim[0] + zlim[1])
    r  = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) * 0.5 / zoom
    xlim3 = (xc - r, xc + r)
    ylim3 = (yc - r, yc + r)
    zlim3 = (zc - r, zc + r)

    stride = max(1, len(pts_c) // TARGET_PTS)
    idx    = np.arange(0, len(pts_c), stride)
    Pz     = pts_c[idx]
    gt_k   = gt_c[idx]

    clks = clicks[ti] if clicks else []

    # azimuths span 0→360 across all frames; click state advances every N frames
    n_total  = K * frames_per_click
    azimuths = np.linspace(0, 360, n_total, endpoint=False)

    print(f"  Rendering {n_total} frames ({K} clicks × {frames_per_click}) …")
    frames = []
    durations = []

    for i, azim in enumerate(azimuths):
        k      = i // frames_per_click + 1   # click index 1..K
        shown  = [c for c in clks if len(c) >= 5 and int(c[4]) <= k]
        pred_k = pred_c[k-1][idx]
        col    = error_rgba(pred_k, gt_k, Pz[:, 2])
        iou_v  = iou(pred_c[k-1], gt_c)

        frame = render_3d_frame(Pz, col, shown, azim, ELEV, xlim3, ylim3, zlim3)
        frame = add_overlay(frame, k, iou_v)
        frames.append(frame)

        # pause 600 ms on first frame of each new click; normal otherwise
        is_first_of_click = (i % frames_per_click == 0)
        durations.append(600 if is_first_of_click else 120)

    # uniform canvas
    W = max(f.size[0] for f in frames)
    H = max(f.size[1] for f in frames)
    def pad(img):
        if img.size == (W, H): return img
        c = Image.new("RGB", (W, H), BG_RGB)
        c.paste(img, ((W-img.size[0])//2, (H-img.size[1])//2))
        return c
    frames = [pad(f) for f in frames]

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    print(f"  Saved {n_total}-frame 360° GIF → {out_path}")


if __name__ == "__main__":
    make_360_gif(
        f"{DUMPS}/selectanytree/NIBIO_NIBIO_plot_13_annotated_val.npz",
        f"{OUT}/demo_360.gif",
        tree_idx=0,
    )
    print("Done.")
