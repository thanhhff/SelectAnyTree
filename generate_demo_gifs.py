"""
Generate demo GIFs for SelectAnyTree project page (v2).
Each frame shows top-down view (left) + side view (right) stacked horizontally.
Comparison uses Point-SAM (2nd-best) vs SelectAnyTree.

Text-overlap fix
----------------
The previous version saved frames with `bbox_inches="tight"`, which makes each
frame a slightly different pixel size (frames with an "IoU=x.xx" title crop
differently from the base frame). `save_gif` then force-*resized* every frame to
the first frame's dimensions, squashing the content; combined with no GIF frame
disposal and an alpha channel, old text ghosted under new text.

Fixes:
  * Render at a FIXED figure size, no `bbox_inches="tight"`  -> every frame is
    exactly the same pixel size.
  * Convert each frame to RGB (drop alpha) so nothing shows through.
  * In `save_gif`, pad (never distort) onto a uniform canvas and write with
    `disposal=2` so each frame fully replaces the previous one.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
import io

DUMPS = "/workdir/radish/3d-project/workspace/SelectAnyTree-Dev/viz/dumps"
OUT   = "/workdir/radish/3d-project/workspace/SelectAnyTree-Public/static/images"
os.makedirs(OUT, exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────────────
C_TP = np.array([0.18, 0.63, 0.24, 0.88])
C_FP = np.array([0.88, 0.20, 0.20, 0.88])
C_FN = np.array([0.18, 0.42, 0.92, 0.88])
C_BG = np.array([0.80, 0.80, 0.80, 0.18])

PT = 0.55
DPI = 130
TARGET_PTS = 100_000
BG_RGB = (13, 26, 13)          # exactly #0d1a0d — must match the figure facecolor
                               # so padding bands blend with the panel background


def height_color(z, alpha=0.22):
    z = np.asarray(z, float)
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    r = np.clip(1.5 * t,       0, 1)
    g = np.clip(0.45 + 0.55*t, 0, 1)
    b = np.clip(0.40 - 0.40*t, 0, 1)
    a = np.full_like(t, alpha)
    return np.stack([r, g, b, a], axis=1)


def error_colors(pred, gt, pts_z):
    col = height_color(pts_z)
    col[gt & pred]  = C_TP
    col[~gt & pred] = C_FP
    col[gt & ~pred] = C_FN
    return col


def iou(pred, gt):
    return np.logical_and(pred, gt).sum() / max(1, np.logical_or(pred, gt).sum())


def _draw_panel(ax, Pz, view, col, clicks, title, xlim, ylim, spine_color):
    a, b = (0, 1) if view == "topdown" else (0, 2)
    ax.scatter(Pz[:, a], Pz[:, b], c=col, s=PT, linewidths=0, rasterized=True)
    for c in (clicks or []):
        xyz, lab = c[:3], int(c[3])
        kind = c[5] if len(c) > 5 else ("pos" if lab == 1 else "neg")
        if kind == "chm":
            ax.scatter(xyz[a], xyz[b], marker="^", s=80,
                       fc="darkorange", ec="white", lw=0.9, zorder=8)
        elif lab == 1:
            ax.scatter(xyz[a], xyz[b], marker="*", s=120,
                       fc="yellow", ec="black", lw=0.9, zorder=8)
        else:
            ax.scatter(xyz[a], xyz[b], marker="X", s=80,
                       fc="red", ec="white", lw=0.9, zorder=8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(spine_color); sp.set_linewidth(1.5)
    ax.set_title(title, color="white", fontsize=8.5, fontweight="bold", pad=3)
    ylabel = "Z (m)" if view == "side" else "Y (m)"
    ax.set_ylabel(ylabel, color="#aaa", fontsize=7, labelpad=2)


def render_frame_both_views(pts_c, gt_c, pred_c, clicks_shown, stride,
                             label, xlim, ylim, zylim, iou_val,
                             spine_color="#2d6a4f", figh=3.4):
    """Render one frame with top-down (left) + side view (right). Returns PIL Image.

    Rendered at a FIXED figure size (no tight bbox) so every frame in a sequence
    has identical pixel dimensions; returned as RGB so nothing shows through.

    Each panel's slot is sized to its data aspect (top-down is square; the side
    view is taller than wide) so the plotted bbox fills the slot instead of
    leaving big dark bands around it.
    """
    x_width  = float(xlim[1] - xlim[0])
    z_height = float(zylim[1] - zylim[0])
    r_side   = max(0.30, min(1.0, x_width / max(z_height, 1e-6)))   # side-view slot width
    figw     = figh * (1.0 + r_side) + 0.5                          # +0.5 for y-labels
    fig, (ax_td, ax_sv) = plt.subplots(
        1, 2, figsize=(figw, figh), facecolor="#0d1a0d",
        gridspec_kw=dict(width_ratios=[1.0, r_side], wspace=0.04))
    for ax in (ax_td, ax_sv):
        ax.set_facecolor("#0d1a0d")

    idx = np.arange(0, len(pts_c), stride)
    Pz  = pts_c[idx]
    gz  = gt_c[idx]
    pz  = pred_c[idx] if pred_c is not None else None

    col = height_color(Pz[:, 2], alpha=0.22) if pz is None else error_colors(pz, gz, Pz[:, 2])
    if pz is None:                 # faint GT highlight on base frame
        col[gz] = [0.25, 0.78, 0.30, 0.45]

    iou_str = f"    IoU={iou_val:.2f}" if iou_val is not None else ""
    # Short per-panel titles only — the long descriptive label goes in a single
    # figure-level suptitle so the two panel titles can never overlap.
    _draw_panel(ax_td, Pz, "topdown", col, clicks_shown, "Top-down", xlim, ylim, spine_color)
    _draw_panel(ax_sv, Pz, "side", col, clicks_shown, "Side view", xlim, zylim, spine_color)

    suptitle = f"{label}{iou_str}".strip(" —")
    if suptitle:
        fig.suptitle(suptitle, color="white", fontsize=10.5, fontweight="bold", y=0.99)
    fig.tight_layout(pad=0.2, w_pad=0.3, rect=[0, 0, 1, 0.94])
    buf = io.BytesIO()
    # bbox_inches="tight" trims the surrounding empty margins so the panels fill
    # the frame and stay centered. Frames within a GIF share the same extents, so
    # they come out the same size; save_gif pads any 1-2px difference onto a
    # common centered canvas.
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGB")   # drop alpha -> no ghosting
    buf.close()
    plt.close(fig)
    return img


def crop_scene(pts, gt_ti, preds_ti, xlim, ylim, zpad_frac=0.08):
    mask = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
            (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
    pts_c = pts[mask]
    zlo, zhi = pts_c[:, 2].min(), pts_c[:, 2].max()
    zpad = zpad_frac * (zhi - zlo) + 0.5
    zylim = (zlo - zpad, zhi + zpad)
    return pts_c, gt_ti[mask], preds_ti[:, mask], zylim


def _window_for_tree(pts, gt_ti, margin=6.0):
    tree_pts = pts[gt_ti]
    cen  = tree_pts[:, :2].mean(0)
    half = max(tree_pts[:, :2].max(0) - tree_pts[:, :2].min(0)) / 2
    half = max(half, 4.0) + margin
    xlim = [cen[0] - half, cen[0] + half]
    ylim = [cen[1] - half, cen[1] + half]

    # Clamp the window to where points actually exist so we never render empty
    # bands beyond the scene edge (happens when the target tree is near the
    # boundary of a small scene), then re-square + re-center on the visible
    # content so the panel stays centered. Scenes whose window already fits
    # inside the point cloud (e.g. NIBIO) are left unchanged.
    xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
    xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())
    xlim = [max(xlim[0], xmin), min(xlim[1], xmax)]
    ylim = [max(ylim[0], ymin), min(ylim[1], ymax)]
    cx, cy = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
    h = min((xlim[1] - xlim[0]) / 2, (ylim[1] - ylim[0]) / 2)
    return (cx - h, cx + h), (cy - h, cy + h)


def _pad_to(img, size):
    """Center-paste an RGB frame onto a uniform canvas (no distortion)."""
    if img.size == size:
        return img
    canvas = Image.new("RGB", size, BG_RGB)
    off = ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
    canvas.paste(img, off)
    return canvas


def unify_gif_sizes(paths):
    """Pad a set of GIFs to one common (max W, max H) canvas so they all share
    the same pixel size (each scene otherwise crops to its own aspect)."""
    metas, W, H = [], 0, 0
    for p in paths:
        im = Image.open(p)
        frs = [im.convert("RGB").copy() for _ in _seek_all(im)]
        durs = [im.info.get("duration", 600) for _ in range(im.n_frames)]
        metas.append((p, frs, durs))
        W, H = max(W, frs[0].size[0]), max(H, frs[0].size[1])
    for p, frs, durs in metas:
        padded = [_pad_to(f, (W, H)) for f in frs]
        padded[0].save(p, save_all=True, append_images=padded[1:],
                       duration=durs, loop=0, disposal=2, optimize=False)
    print(f"  Unified {len(paths)} GIFs to {W}x{H}")


def _seek_all(im):
    for i in range(im.n_frames):
        im.seek(i)
        yield i


def save_gif(frames, out_path, fps=1.0, hold_last=1):
    tick_ms  = int(1000 / fps)   # per-click frame duration
    base_ms  = 300               # brief pause on the base (no-seg) frame
    last_ms  = 1500              # hold the final result so viewers can read it
    duration = [base_ms] + [tick_ms] * (len(frames) - 1 - hold_last) + [last_ms] * hold_last
    duration = duration[:len(frames)]
    while len(duration) < len(frames):
        duration.append(tick_ms)

    # Uniform canvas via padding (never resize/distort) -> text stays crisp.
    W = max(f.size[0] for f in frames)
    H = max(f.size[1] for f in frames)
    frames_r = [_pad_to(f.convert("RGB"), (W, H)) for f in frames]

    # disposal=2 -> each frame fully replaces the previous one (no ghosting).
    frames_r[0].save(out_path, save_all=True, append_images=frames_r[1:],
                     duration=duration, loop=0, optimize=False, disposal=2)
    print(f"  Saved {len(frames_r)}-frame GIF → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def make_scene_gif(npz_sat, out_path, tree_idx=0, fps=1.7, margin=6.0,
                   label="SelectAnyTree", spine_color="#2d6a4f"):
    d = np.load(npz_sat, allow_pickle=True)
    pts, gt, preds = d["points"], d["gt"], d["preds"]
    clicks = d["clicks"].tolist() if "clicks" in d.files else None
    ti = tree_idx

    xlim, ylim   = _window_for_tree(pts, gt[ti], margin)
    pts_c, gt_c, pred_c, zylim = crop_scene(pts, gt[ti], preds[:, ti], xlim, ylim)
    stride = max(1, len(pts_c) // TARGET_PTS)
    clks   = clicks[ti] if clicks else []
    K      = preds.shape[0]

    frames = []
    # single base frame — duration controlled by save_gif (base_ms = 500ms)
    frames.append(render_frame_both_views(
        pts_c, gt_c, None, [], stride, "Input scene", xlim, ylim, zylim,
        None, spine_color))

    for k in range(1, K + 1):
        shown   = [c for c in clks if len(c) >= 5 and int(c[4]) <= k]
        iou_val = iou(pred_c[k-1], gt_c)
        frames.append(render_frame_both_views(
            pts_c, gt_c, pred_c[k-1], shown, stride,
            f"{k} click(s)", xlim, ylim, zylim,
            iou_val, spine_color))

    save_gif(frames, out_path, fps)


def make_compare_gif(npz_sat, npz_base, out_path, tree_idx=0, fps=1.7, margin=6.0,
                     label_sat="SelectAnyTree", label_base="Point-SAM"):
    """2-row comparison: top row = Point-SAM, bottom row = SelectAnyTree.
    Each row shows [top-down | side view].
    """
    ds = np.load(npz_sat,  allow_pickle=True)
    db = np.load(npz_base, allow_pickle=True)
    ti = tree_idx

    pts_s, gt_s, preds_s = ds["points"], ds["gt"], ds["preds"]
    pts_b, gt_b, preds_b = db["points"], db["gt"], db["preds"]
    clks_s = ds["clicks"].tolist()[ti] if "clicks" in ds.files else []
    clks_b = db["clicks"].tolist()[ti] if "clicks" in db.files else []

    xlim, ylim = _window_for_tree(pts_s, gt_s[ti], margin)

    pts_sc, gt_sc, pred_sc, zylim_s = crop_scene(pts_s, gt_s[ti], preds_s[:, ti], xlim, ylim)
    pts_bc, gt_bc, pred_bc, zylim_b = crop_scene(pts_b, gt_b[ti], preds_b[:, ti], xlim, ylim)

    # Use the wider z range for consistent side view
    zylim = (min(zylim_s[0], zylim_b[0]), max(zylim_s[1], zylim_b[1]))

    stride_s = max(1, len(pts_sc) // TARGET_PTS)
    stride_b = max(1, len(pts_bc) // TARGET_PTS)

    K = min(preds_s.shape[0], preds_b.shape[0])
    frames = []

    for k in range(1, K + 1):
        sh_s = [c for c in clks_s if len(c) >= 5 and int(c[4]) <= k]
        sh_b = [c for c in clks_b if len(c) >= 5 and int(c[4]) <= k]

        iou_s = iou(pred_sc[k-1], gt_sc)
        iou_b = iou(pred_bc[k-1], gt_bc)

        fi_b = render_frame_both_views(
            pts_bc, gt_bc, pred_bc[k-1], sh_b, stride_b,
            f"{label_base} — {k} click(s)", xlim, ylim, zylim, iou_b,
            spine_color="#8b1a1a")
        fi_s = render_frame_both_views(
            pts_sc, gt_sc, pred_sc[k-1], sh_s, stride_s,
            f"{label_sat} — {k} click(s)", xlim, ylim, zylim, iou_s,
            spine_color="#2d6a4f")

        # Both frames share a fixed render size, so they align exactly.
        W = max(fi_b.size[0], fi_s.size[0])
        H = max(fi_b.size[1], fi_s.size[1])
        fi_b = _pad_to(fi_b, (W, H))
        fi_s = _pad_to(fi_s, (W, H))

        # Stack vertically: baseline on top, ours on bottom
        combo = Image.new("RGB", (W, H * 2 + 4), BG_RGB)
        combo.paste(fi_b, (0, 0))
        combo.paste(fi_s, (0, H + 4))
        frames.append(combo)

    save_gif(frames, out_path, fps)


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating demo GIFs (v2) ...")

    make_scene_gif(
        f"{DUMPS}/selectanytree/NIBIO_NIBIO_plot_13_annotated_val.npz",
        f"{OUT}/demo_nibio.gif", tree_idx=0
    )
    make_scene_gif(
        f"{DUMPS}/selectanytree/BlueCat_RN_merged_trees_val_subset000.npz",
        f"{OUT}/demo_bluecat.gif", tree_idx=0
    )
    make_scene_gif(
        f"{DUMPS}/selectanytree/CULS_CULS_plot_3_annotated_val.npz",
        f"{OUT}/demo_culs.gif", tree_idx=0
    )
    make_scene_gif(
        f"{DUMPS}/selectanytree/SCION_SCION_plot_35_annotated_val.npz",
        f"{OUT}/demo_scion.gif", tree_idx=0
    )
    # Make all four single-scene demos share one pixel size.
    unify_gif_sizes([f"{OUT}/demo_nibio.gif", f"{OUT}/demo_bluecat.gif",
                     f"{OUT}/demo_culs.gif", f"{OUT}/demo_scion.gif"])

    # Point-SAM vs. SelectAnyTree — two side-by-side GIFs in the SAME style as the
    # single-scene demos above (top-down + side, click-by-click). Scene chosen for
    # the clearest single-click gap: SAT ~0.97 @ 1 click vs Point-SAM ~0.46.
    COMPARE = "NIBIO_NIBIO_plot_16_annotated_val.npz"
    make_scene_gif(
        f"{DUMPS}/pointsam/{COMPARE}",
        f"{OUT}/demo_pointsam.gif", tree_idx=0,
        label="Point-SAM", spine_color="#8b1a1a"
    )
    make_scene_gif(
        f"{DUMPS}/selectanytree/{COMPARE}",
        f"{OUT}/demo_sat_compare.gif", tree_idx=0,
        label="SelectAnyTree", spine_color="#2d6a4f"
    )
    # Match pixel sizes so the two panels align side-by-side in the page.
    unify_gif_sizes([f"{OUT}/demo_pointsam.gif", f"{OUT}/demo_sat_compare.gif"])

    print("Done.")
