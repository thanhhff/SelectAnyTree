"""
Generate teaser figure for SelectAnyTree project page.

Layout (single wide figure):
  Left panel  : top-down view of full scene, height-colored background,
                3 selected trees in distinct vivid colors, click markers
  Right panels: 3 zoom columns (one per tree) × 2 rows (top-down / side view)
                showing the prediction at 1 click and 3 clicks
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch

DUMPS = "/workdir/radish/3d-project/workspace/SelectAnyTree-Dev/viz/dumps"
OUT   = "/workdir/radish/3d-project/workspace/SelectAnyTree-Public/static/images"
os.makedirs(OUT, exist_ok=True)

TARGET_BG   = 200_000   # background points per overview panel
TARGET_ZOOM = 80_000    # points per zoom panel
PT_BG   = 0.4
PT_ZOOM = 0.6
DPI     = 150
PAD     = 0.08

TREE_COLORS = [
    np.array([0.96, 0.49, 0.00, 0.92]),  # orange
    np.array([0.20, 0.75, 0.95, 0.92]),  # cyan
    np.array([0.95, 0.22, 0.55, 0.92]),  # pink/magenta
]
C_TP = np.array([0.18, 0.72, 0.28, 0.90])
C_FP = np.array([0.90, 0.22, 0.22, 0.90])
C_FN = np.array([0.20, 0.44, 0.95, 0.90])
C_BG = np.array([0.78, 0.78, 0.78, 0.15])
BG_COLOR = "#0d1a0d"


def height_rgba(z, alpha=0.16):
    z = np.asarray(z, float)
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    r = np.clip(1.5 * t,        0, 1)
    g = np.clip(0.45 + 0.55*t,  0, 1)
    b = np.clip(0.40 - 0.40*t,  0, 1)
    a = np.full_like(t, alpha)
    return np.stack([r, g, b, a], axis=1)


def error_rgba(pred, gt, pts_z):
    col = np.tile(C_BG, (len(gt), 1))
    col[gt & pred]  = C_TP
    col[~gt & pred] = C_FP
    col[gt & ~pred] = C_FN
    return col


def iou(pred, gt):
    return np.logical_and(pred, gt).sum() / max(1, np.logical_or(pred, gt).sum())


def draw_clicks(ax, clicks_shown, a, b):
    for c in (clicks_shown or []):
        xyz, lab = c[:3], int(c[3])
        kind = c[5] if len(c) > 5 else ("pos" if lab == 1 else "neg")
        if kind == "chm":
            ax.scatter(xyz[a], xyz[b], marker="^", s=90,
                       fc="darkorange", ec="white", lw=0.9, zorder=9)
        elif lab == 1:
            ax.scatter(xyz[a], xyz[b], marker="*", s=140,
                       fc="yellow", ec="black", lw=0.9, zorder=9)
        else:
            ax.scatter(xyz[a], xyz[b], marker="X", s=90,
                       fc="red", ec="white", lw=0.9, zorder=9)


# ── Load scene ───────────────────────────────────────────────────────────────
npz_path = f"{DUMPS}/selectanytree/NIBIO_NIBIO_plot_13_annotated_val.npz"
d = np.load(npz_path, allow_pickle=True)
pts    = d["points"]          # (N, 3)
gt     = d["gt"]              # (G, N) bool
preds  = d["preds"]           # (K, G, N) bool
clicks = d["clicks"].tolist() # list[G][list[click]]
ground = d["ground"] if "ground" in d.files else None
G      = gt.shape[0]          # number of trees = 3

# ── Overview window ──────────────────────────────────────────────────────────
cen  = 0.5 * (pts[:, :2].min(0) + pts[:, :2].max(0))
half = 0.5 * float((pts[:, :2].max(0) - pts[:, :2].min(0)).max())
H    = half * (1 + PAD) + 0.5
xlim_ov = (cen[0] - H, cen[0] + H)
ylim_ov = (cen[1] - H, cen[1] + H)
zlo, zhi = float(pts[:, 2].min()), float(pts[:, 2].max())
zpad = PAD * (zhi - zlo) + 0.5
zlim = (zlo - zpad, zhi + zpad)

# Subsample background for overview
not_any_tree = ~(gt[0] | gt[1] | gt[2])
bg_idx = np.where(not_any_tree)[0]
stride_bg = max(1, len(bg_idx) // TARGET_BG)
bg_idx = bg_idx[::stride_bg]
fg_idx = np.where(gt[0] | gt[1] | gt[2])[0]
stride_fg = max(1, len(fg_idx) // (TARGET_BG // 4))
fg_idx = fg_idx[::stride_fg]
ov_idx = np.concatenate([bg_idx, fg_idx])
Pov   = pts[ov_idx]
# Remove ground for top-down clarity
if ground is not None:
    ov_keep = ~ground[ov_idx]
else:
    ov_keep = np.ones(len(ov_idx), bool)

# Colors for overview: height for BG, vivid color per selected tree
col_ov = height_rgba(Pov[:, 2], alpha=0.18)
for ti, tc in enumerate(TREE_COLORS[:G]):
    mask_ti = gt[ti][ov_idx]
    col_ov[mask_ti] = tc

# ── Zoom windows (one per tree) ──────────────────────────────────────────────
zoom_margin = 5.0
zooms = []
for ti in range(G):
    tpts = pts[gt[ti]]
    cen_t = tpts[:, :2].mean(0)
    half_t = max(tpts[:, :2].max(0) - tpts[:, :2].min(0)) / 2
    half_t = max(half_t, 3.5) + zoom_margin
    xl = (cen_t[0] - half_t, cen_t[0] + half_t)
    yl = (cen_t[1] - half_t, cen_t[1] + half_t)

    win = ((pts[:, 0] >= xl[0]) & (pts[:, 0] <= xl[1]) &
           (pts[:, 1] >= yl[0]) & (pts[:, 1] <= yl[1]))
    pts_z = pts[win]
    gt_z  = gt[ti][win]
    preds_z = preds[:, ti][..., win]
    clks_t  = clicks[ti]

    zlo_z, zhi_z = pts_z[:, 2].min(), pts_z[:, 2].max()
    zp = 0.08 * (zhi_z - zlo_z) + 0.5
    zlim_z = (zlo_z - zp, zhi_z + zp)

    stride_z = max(1, len(pts_z) // TARGET_ZOOM)
    idx_z = np.arange(0, len(pts_z), stride_z)

    zooms.append(dict(
        pts_z=pts_z, gt_z=gt_z, preds_z=preds_z, clks=clks_t,
        xl=xl, yl=yl, zlim_z=zlim_z, stride_z=stride_z, idx_z=idx_z,
        color=TREE_COLORS[ti]
    ))

# ── Build figure ─────────────────────────────────────────────────────────────
# Layout: 2 rows × (1 + G*2) cols
# Col 0       : overview top-down (rowspan 2)
# Cols 1-2    : tree 0 [top-down, side] in rows 0, 1
# Cols 3-4    : tree 1 [top-down, side] in rows 0, 1
# Cols 5-6    : tree 2 [top-down, side] in rows 0, 1
#
# Actually let's do a cleaner 2-row layout:
# Row 0 (top-down): overview | zoom_tree0_td | zoom_tree1_td | zoom_tree2_td
# Row 1 (side)    : overview | zoom_tree0_sv | zoom_tree1_sv | zoom_tree2_sv
# The overview column spans both rows.

n_cols = 1 + G   # 4 total
n_rows = 2
figw = 3.8 * n_cols
figh = 3.8

fig = plt.figure(figsize=(figw, figh * n_rows), facecolor=BG_COLOR)
# Use gridspec; overview spans both rows
from matplotlib.gridspec import GridSpec
gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.06)

ax_ov_td = fig.add_subplot(gs[0, 0])
ax_ov_sv = fig.add_subplot(gs[1, 0])
zoom_axes = [[fig.add_subplot(gs[row, 1 + ti]) for row in range(n_rows)]
             for ti in range(G)]

# ── Paint overview top-down ──────────────────────────────────────────────────
ax = ax_ov_td
ax.set_facecolor(BG_COLOR)
keep = ov_keep
ax.scatter(Pov[keep, 0], Pov[keep, 1], c=col_ov[keep],
           s=PT_BG, linewidths=0, rasterized=True)
# click markers for all trees
for ti, z in enumerate(zooms):
    shown = [c for c in z["clks"] if len(c) >= 5 and int(c[4]) <= 1]
    draw_clicks(ax, shown, 0, 1)
ax.set_xlim(*xlim_ov); ax.set_ylim(*ylim_ov)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(1.2)
ax.set_title("Scene overview — top-down", color="white", fontsize=9,
             fontweight="bold", pad=3)
ax.set_ylabel("Y (m)", color="#aaa", fontsize=8, labelpad=2)

# ── Paint overview side view ──────────────────────────────────────────────────
ax = ax_ov_sv
ax.set_facecolor(BG_COLOR)
# For side view include ground
col_ov_sv = height_rgba(Pov[:, 2], alpha=0.18)
for ti, tc in enumerate(TREE_COLORS[:G]):
    col_ov_sv[gt[ti][ov_idx]] = tc
ax.scatter(Pov[:, 0], Pov[:, 2], c=col_ov_sv,
           s=PT_BG, linewidths=0, rasterized=True)
ax.set_xlim(*xlim_ov); ax.set_ylim(*zlim)
ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(1.2)
ax.set_title("Scene overview — side view", color="white", fontsize=9,
             fontweight="bold", pad=3)
ax.set_ylabel("Z (m)", color="#aaa", fontsize=8, labelpad=2)
ax.set_xlabel("X (m)", color="#aaa", fontsize=8, labelpad=2)

# ── Paint zoom panels ─────────────────────────────────────────────────────────
SHOW_CLICKS = 3   # show prediction after 3 clicks in zoom panels

for ti, (z, axrows) in enumerate(zip(zooms, zoom_axes)):
    idx_z  = z["idx_z"]; pts_z  = z["pts_z"]
    gt_z   = z["gt_z"];  preds_z = z["preds_z"]
    clks   = z["clks"];  xl = z["xl"]; yl = z["yl"]; zlim_z = z["zlim_z"]
    stride_z = z["stride_z"]
    tc     = z["color"]

    Pz  = pts_z[idx_z]
    gtz = gt_z[idx_z]
    k   = min(SHOW_CLICKS, preds_z.shape[0])
    pdz = preds_z[k-1][idx_z]
    col_z = error_rgba(pdz, gtz, Pz[:, 2])
    shown = [c for c in clks if len(c) >= 5 and int(c[4]) <= k]
    iov   = iou(pdz, gtz)

    tree_label = f"Tree {ti+1} · IoU={iov:.2f} @ {k} click(s)"

    # top-down
    ax = axrows[0]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(Pz[:, 0], Pz[:, 1], c=col_z, s=PT_ZOOM, linewidths=0, rasterized=True)
    draw_clicks(ax, shown, 0, 1)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    color_hex = "#{:02x}{:02x}{:02x}".format(
        int(tc[0]*255), int(tc[1]*255), int(tc[2]*255))
    for sp in ax.spines.values(): sp.set_color(color_hex); sp.set_linewidth(2.2)
    ax.set_title(tree_label, color="white", fontsize=8.5, fontweight="bold", pad=3)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7, labelpad=2)

    # side view
    ax = axrows[1]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(Pz[:, 0], Pz[:, 2], c=col_z, s=PT_ZOOM, linewidths=0, rasterized=True)
    draw_clicks(ax, shown, 0, 2)
    ax.set_xlim(*xl); ax.set_ylim(*zlim_z)
    ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(color_hex); sp.set_linewidth(2.2)
    ax.set_title("side view", color="white", fontsize=8.5, pad=3)
    ax.set_ylabel("Z (m)", color="#aaa", fontsize=7, labelpad=2)
    ax.set_xlabel("X (m)", color="#aaa", fontsize=7, labelpad=2)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_TP, label="True positive"),
    mpatches.Patch(color=C_FP, label="False positive"),
    mpatches.Patch(color=C_FN, label="False negative"),
    mlines.Line2D([], [], marker="^", color="none", mfc="darkorange",
                  mec="white", ms=10, label="CHM treetop"),
    mlines.Line2D([], [], marker="*", color="none", mfc="yellow",
                  mec="black", ms=12, label="Positive click"),
    mlines.Line2D([], [], marker="X", color="none", mfc="red",
                  mec="white", ms=10, label="Negative click"),
]
# Reserve a thin band at the very bottom and seat the legend in it, so it sits
# right under the panels instead of floating far below them.
fig.subplots_adjust(bottom=0.10)
fig.legend(handles=legend_handles, loc="upper center", ncol=6,
           fontsize=8, frameon=True, framealpha=0.85,
           facecolor="#1a2a1a", labelcolor="white",
           edgecolor="#444",
           bbox_to_anchor=(0.5, 0.05))

out_path = f"{OUT}/teaser.png"
fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05,
            facecolor=BG_COLOR)
plt.close(fig)
print(f"Saved teaser → {out_path}")
