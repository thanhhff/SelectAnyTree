"""
Teaser-v2 for SelectAnyTree — 4-column rotating turntable.

Mirrors the static teaser's layout, but every panel is an animated 3D
turntable that all rotate in sync:

    ┌──────────┬──────────┬──────────┬──────────┐
    │  Scene   │  Tree 1  │  Tree 2  │  Tree 3  │
    │ all trees│  + clicks│  + clicks│  + clicks│
    │ (colored)│  IoU/err │  IoU/err │  IoU/err │
    └──────────┴──────────┴──────────┴──────────┘

  * column 1: full plot, every tree in its own vivid color + all click prompts
  * columns 2..G+1: one tree each, zoomed, TP/FP/FN error coloring, its own
    click prompts (▲ CHM · ★ positive · ✕ negative) and per-tree IoU title

No axes, no grid. One seamless 360° loop.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
import io

DUMPS = "/workdir/radish/3d-project/workspace/SelectAnyTree-Dev/viz/dumps"
OUT   = "/workdir/radish/3d-project/workspace/SelectAnyTree-Public/static/images"
os.makedirs(OUT, exist_ok=True)

TREE_COLORS = [
    np.array([0.96, 0.49, 0.00, 0.95]),  # orange
    np.array([0.20, 0.78, 0.95, 0.95]),  # cyan
    np.array([0.95, 0.22, 0.55, 0.95]),  # pink/magenta
    np.array([0.62, 0.80, 0.20, 0.95]),  # lime
    np.array([0.70, 0.45, 0.95, 0.95]),  # violet
    np.array([0.98, 0.82, 0.10, 0.95]),  # gold
]
# TP / FP / FN error coloring for the per-tree panels
C_TP = np.array([0.18, 0.70, 0.28, 0.95])
C_FP = np.array([0.90, 0.22, 0.22, 0.95])
C_FN = np.array([0.20, 0.48, 0.95, 0.95])

DPI              = 95
FRAMES_PER_CLICK = 10        # frames spent at each click step (× K clicks = total)
# Camera elevation sweeps from side-on up to near top-down and back over the
# loop, so a single rotation reveals BOTH the side view and the top-down view.
ELEV_LO          = 12        # side-on
ELEV_HI          = 82        # near top-down
PANEL_IN    = 3.3        # inches per panel
FIG_H       = 3.6
BG_COLOR    = "#0d1a0d"      # match generate_demo_gifs.py
BG_RGB      = (13, 26, 13)
OVERVIEW_PTS = 130_000
TREE_PTS     = 55_000


def ground_rgba(z, alpha=0.09):
    z = np.asarray(z, float)
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    base = 0.45 + 0.20 * t
    a = np.full_like(t, alpha)
    return np.stack([base*0.8, base, base*0.8, a], axis=1)


def error_rgba(pred, gt, z):
    col = ground_rgba(z, alpha=0.10)
    col[gt & pred]  = C_TP
    col[~gt & pred] = C_FP
    col[gt & ~pred] = C_FN
    return col


def iou(pred, gt):
    return np.logical_and(pred, gt).sum() / max(1, np.logical_or(pred, gt).sum())


def _draw_clicks(ax, clks):
    for c in (clks or []):
        xyz, lab = c[:3], int(c[3])
        kind = c[5] if len(c) > 5 else ("pos" if lab == 1 else "neg")
        if kind == "chm":
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="^", s=110,
                       c="darkorange", edgecolors="white", linewidths=0.9,
                       zorder=10, depthshade=False)
        elif lab == 1:
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="*", s=170,
                       c="yellow", edgecolors="black", linewidths=0.9,
                       zorder=10, depthshade=False)
        else:
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker="X", s=95,
                       c="red", edgecolors="white", linewidths=0.9,
                       zorder=10, depthshade=False)


def _style_3d(ax, lim3, azim, elev, zoom=1.0):
    (xl, yl, zl) = lim3
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    # Box proportions follow the DATA ranges (not a cube) so a tall, thin tree
    # renders tall and thin — no big black bands on the sides. `zoom` enlarges
    # the plotted box within the axes to use up the remaining empty margin.
    dx, dy, dz = (xl[1]-xl[0]), (yl[1]-yl[0]), (zl[1]-zl[0])
    try:
        ax.set_box_aspect((dx, dy, dz), zoom=zoom)
    except TypeError:                       # older matplotlib: no zoom kwarg
        ax.set_box_aspect((dx, dy, dz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)


def _data_lims(pts_xyz, pad_frac=0.06, floor=0.4):
    """Tight axis ranges from the actual point extent (with a little padding).
    Returns ((xlo,xhi),(ylo,yhi),(zlo,zhi))."""
    out = []
    for ax_i in range(3):
        lo, hi = float(pts_xyz[:, ax_i].min()), float(pts_xyz[:, ax_i].max())
        pad = pad_frac * (hi - lo) + floor
        out.append((lo - pad, hi + pad))
    return tuple(out)


def make_teaser_v2(npz_path, out_path, frames_per_click=FRAMES_PER_CLICK,
                   drop_ground=True):
    d = np.load(npz_path, allow_pickle=True)
    pts, gt, preds = d["points"], d["gt"], d["preds"]
    clicks = d["clicks"].tolist() if "clicks" in d.files else None
    ground = d["ground"] if "ground" in d.files else None
    G      = gt.shape[0]
    K      = preds.shape[0]
    final  = preds[K-1]                       # (G, N) — used for crop windows

    any_tree = np.zeros(pts.shape[0], bool)
    for ti in range(G):
        any_tree |= final[ti]

    # ── OVERVIEW panel data (thinned scene; per-click tree predictions) ───────
    bg_mask = ~any_tree
    if drop_ground and ground is not None:
        bg_mask &= ~ground
    bg_idx = np.where(bg_mask)[0]
    tr_idx = np.where(any_tree)[0]
    bg_keep = int(OVERVIEW_PTS * 0.45)
    if len(bg_idx) > bg_keep:
        bg_idx = bg_idx[:: max(1, len(bg_idx) // bg_keep)]
    tr_keep = OVERVIEW_PTS - bg_keep
    if len(tr_idx) > tr_keep:
        tr_idx = tr_idx[:: max(1, len(tr_idx) // tr_keep)]
    ov_idx = np.concatenate([bg_idx, tr_idx])
    P_ov   = pts[ov_idx]
    base_col_ov = ground_rgba(P_ov[:, 2], alpha=0.09)
    # per-tree prediction at every click step, in overview index space
    ov_pred = [preds[:, ti][:, ov_idx] for ti in range(G)]   # each (K, n_ov)

    tree_pts_all = pts[any_tree]
    ov_lim = _data_lims(tree_pts_all, pad_frac=0.05)

    # ── PER-TREE panel data (zoomed crop; per-click pred / clicks / IoU) ───────
    tree_data = []
    for ti in range(G):
        tp = pts[gt[ti]]
        cen = tp[:, :2].mean(0)
        half = max(tp[:, :2].max(0) - tp[:, :2].min(0)) / 2 + 1.5
        xl = (cen[0]-half, cen[0]+half); yl = (cen[1]-half, cen[1]+half)
        m = ((pts[:, 0] >= xl[0]) & (pts[:, 0] <= xl[1]) &
             (pts[:, 1] >= yl[0]) & (pts[:, 1] <= yl[1]))
        pc, gc = pts[m], gt[ti][m]
        prc = preds[:, ti][:, m]                       # (K, n_crop)
        stride = max(1, len(pc) // TREE_PTS)
        sub = np.arange(0, len(pc), stride)
        Pz, gz = pc[sub], gc[sub]
        pred_sub = prc[:, sub]                          # (K, n_sub)
        iou_k = [iou(preds[k][ti], gt[ti]) for k in range(K)]   # full-res IoU per click
        # frame the panel to the TREE points, with margin so rotation never
        # clips the crown/trunk when viewed from a diagonal angle
        lim = _data_lims(tp, pad_frac=0.12, floor=0.8)
        color_hex = "#{:02x}{:02x}{:02x}".format(
            *[int(x*255) for x in TREE_COLORS[ti % len(TREE_COLORS)][:3]])
        tree_data.append(dict(P=Pz, gz=gz, pred_sub=pred_sub, iou_k=iou_k,
                              lim=lim, tcolor=color_hex))

    n_panels = G + 1
    figw = PANEL_IN * n_panels
    n_frames = K * frames_per_click
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    print(f"  {G} trees → {n_panels} panels; {K} clicks × {frames_per_click} "
          f"= {n_frames} frames …")
    frames, durations = [], []
    for i, azim in enumerate(azimuths):
        k = i // frames_per_click + 1            # current click step 1..K
        k = min(k, K)
        # elevation sweeps LO → HI → LO over the loop (seamless): side → top → side
        elev = ELEV_LO + (ELEV_HI - ELEV_LO) * 0.5 * (1 - np.cos(2*np.pi * i / n_frames))

        fig = plt.figure(figsize=(figw, FIG_H), facecolor=BG_COLOR)

        # overview — color each tree by its prediction at click k
        col_ov = base_col_ov.copy()
        ov_clicks = []
        for ti in range(G):
            col_ov[ov_pred[ti][k-1]] = TREE_COLORS[ti % len(TREE_COLORS)]
            if clicks:
                ov_clicks += [c for c in clicks[ti]
                              if len(c) >= 5 and int(c[4]) <= k]
        ax0 = fig.add_subplot(1, n_panels, 1, projection="3d")
        ax0.set_facecolor(BG_COLOR)
        ax0.scatter(P_ov[:, 0], P_ov[:, 1], P_ov[:, 2], c=col_ov,
                    s=0.4, linewidths=0, rasterized=True, depthshade=False)
        _draw_clicks(ax0, ov_clicks)
        _style_3d(ax0, ov_lim, azim, elev, zoom=1.0)
        panel_titles = [(f"Scene — all trees ({k} click{'s' if k>1 else ''})",
                         "#e1f5e1")]

        # per-tree — prediction / clicks / IoU at click k
        for ti, td in enumerate(tree_data):
            pz   = td["pred_sub"][k-1]
            col  = error_rgba(pz, td["gz"], td["P"][:, 2])
            clks = [c for c in clicks[ti]
                    if len(c) >= 5 and int(c[4]) <= k] if clicks else []
            ax = fig.add_subplot(1, n_panels, ti + 2, projection="3d")
            ax.set_facecolor(BG_COLOR)
            ax.scatter(td["P"][:, 0], td["P"][:, 1], td["P"][:, 2], c=col,
                       s=0.5, linewidths=0, rasterized=True, depthshade=False)
            _draw_clicks(ax, clks)
            title = f"Tree {ti+1} · IoU={td['iou_k'][k-1]:.2f} @ {k} click{'s' if k>1 else ''}"
            _style_3d(ax, td["lim"], azim, elev, zoom=1.05)
            panel_titles.append((title, td["tcolor"]))

        # ── titles in a reserved top band (figure-level → never overlap points)
        L, R = 0.005, 0.995
        span = R - L
        for j, (txt, color) in enumerate(panel_titles):
            cx = L + span * (j + 0.5) / n_panels
            fig.text(cx, 0.92, txt, color=color, fontsize=13, fontweight="bold",
                     ha="center", va="center")

        # Push the axes (point cloud) down so even the tallest treetops + click
        # markers stay well below the title band — no overlap at any rotation.
        fig.subplots_adjust(left=L, right=R, top=0.82, bottom=0.01, wspace=0.02)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG_COLOR)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)

        # uniform frame time → smooth, continuous rotation (no per-click stutter)
        durations.append(130)

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
    print(f"  Saved {n_frames}-frame 4-column teaser-v2 → {out_path}")


if __name__ == "__main__":
    make_teaser_v2(
        f"{DUMPS}/selectanytree/NIBIO_NIBIO_plot_13_annotated_val.npz",
        f"{OUT}/demo_360.gif",
    )
    print("Done.")
