"""
Teaser-v3 for SelectAnyTree — the teaser_360 layout, with a SECOND scene.

Same idea as teaser_360 (column 1 = full scene with every tree colored, then one
zoomed per-tree panel each) but stacked into TWO ROWS for two scenes from
different countries, so the teaser shows breadth as well as depth:

    Row 1  ┌ Scene (all trees) │ Tree 1 │ Tree 2 │ Tree 3 ┐   ← Norway (NIBIO)
    Row 2  └ Scene (all trees) │ Tree 1 │ Tree 2 │ Tree 3 ┘   ← Australia (BlueCat)

All eight panels rotate in sync with the same elevation sweep (side → top-down →
side) and the same click schedule (1→K). Per-tree panels use TP/FP/FN error
coloring + click prompts + per-click IoU. No axes, no grid.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
import io

from generate_teaser_v2 import (
    ground_rgba, error_rgba, iou, _draw_clicks, _style_3d, _data_lims,
    TREE_COLORS, BG_COLOR, BG_RGB, ELEV_LO, ELEV_HI,
    FRAMES_PER_CLICK, DPI, PANEL_IN, OVERVIEW_PTS, TREE_PTS,
)

DUMPS = "/workdir/radish/3d-project/workspace/SelectAnyTree-Dev/viz/dumps"
OUT   = "/workdir/radish/3d-project/workspace/SelectAnyTree-Public/static/images"

# (file, row label) — one scene per row
SCENES = [
    ("NIBIO_NIBIO_plot_13_annotated_val.npz",     "Norway (NIBIO)"),
    ("BlueCat_RN_merged_trees_val_subset000.npz", "Australia (BlueCat)"),
]


def _prep_scene(npz_path, drop_ground=True):
    d = np.load(npz_path, allow_pickle=True)
    pts, gt, preds = d["points"], d["gt"], d["preds"]
    clicks = d["clicks"].tolist() if "clicks" in d.files else None
    ground = d["ground"] if "ground" in d.files else None
    G, K = gt.shape[0], preds.shape[0]
    final = preds[K-1]

    any_tree = np.zeros(pts.shape[0], bool)
    for ti in range(G):
        any_tree |= final[ti]

    # overview: thinned scene, per-click tree predictions in overview index space
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
    P_ov = pts[ov_idx]
    ov_pred = [preds[:, ti][:, ov_idx] for ti in range(G)]
    tree_pts_all = pts[any_tree]
    ov_lim = _data_lims(tree_pts_all, pad_frac=0.05)
    ov_clicks_full = clicks

    # per-tree zoom panels
    tdata = []
    for ti in range(G):
        tp = pts[gt[ti]]
        cen = tp[:, :2].mean(0)
        half = max(tp[:, :2].max(0) - tp[:, :2].min(0)) / 2 + 1.5
        xl = (cen[0]-half, cen[0]+half); yl = (cen[1]-half, cen[1]+half)
        m = ((pts[:, 0] >= xl[0]) & (pts[:, 0] <= xl[1]) &
             (pts[:, 1] >= yl[0]) & (pts[:, 1] <= yl[1]))
        pc, gc = pts[m], gt[ti][m]
        prc = preds[:, ti][:, m]
        stride = max(1, len(pc) // TREE_PTS)
        sub = np.arange(0, len(pc), stride)
        iou_k = [iou(preds[k][ti], gt[ti]) for k in range(K)]
        tcolor = "#{:02x}{:02x}{:02x}".format(
            *[int(x*255) for x in TREE_COLORS[ti % len(TREE_COLORS)][:3]])
        tdata.append(dict(P=pc[sub], gz=gc[sub], pred_sub=prc[:, sub],
                          iou_k=iou_k, lim=_data_lims(tp, pad_frac=0.12, floor=0.8),
                          tcolor=tcolor))

    return dict(G=G, K=K, P_ov=P_ov, base_col=ground_rgba(P_ov[:, 2], alpha=0.09),
                ov_pred=ov_pred, ov_lim=ov_lim, clicks=ov_clicks_full, tdata=tdata)


def make_teaser_v3(out_path, frames_per_click=FRAMES_PER_CLICK):
    scenes = [(_prep_scene(f"{DUMPS}/selectanytree/{f}"), lbl) for f, lbl in SCENES]
    n_rows = len(scenes)
    G = scenes[0][0]["G"]
    n_cols = G + 1
    K = min(s["K"] for s, _ in scenes)
    n_frames = K * frames_per_click
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    figw = PANEL_IN * n_cols
    figh = 3.4 * n_rows

    # manual panel geometry (figure fractions) → reserved title band per row
    Lm, Rm = 0.005, 0.995
    gap_x   = 0.006
    panel_w = (Rm - Lm - (n_cols - 1) * gap_x) / n_cols
    row_h   = 0.355                       # panel height per row
    title_dy = 0.045                      # title sits this far above the panels
    sub_dy   = 0.092                      # iou subtitle below the title
    # vertical placement: row 0 on top, row 1 below, with a title band each
    row_bottoms = [0.55, 0.055] if n_rows == 2 else \
                  [0.92 - (r+1)*(row_h+0.12) for r in range(n_rows)]

    print(f"  {n_rows} scenes × {n_cols} panels; {K} clicks × {frames_per_click} "
          f"= {n_frames} frames …")
    frames, durations = [], []
    for i, azim in enumerate(azimuths):
        k = min(i // frames_per_click + 1, K)
        elev = ELEV_LO + (ELEV_HI - ELEV_LO) * 0.5 * (1 - np.cos(2*np.pi * i / n_frames))

        fig = plt.figure(figsize=(figw, figh), facecolor=BG_COLOR)
        for r, (sc, row_label) in enumerate(scenes):
            y_b = row_bottoms[r]
            # ── overview panel (column 0) ──────────────────────────────────
            col_ov = sc["base_col"].copy()
            ov_clicks = []
            for ti in range(sc["G"]):
                col_ov[sc["ov_pred"][ti][k-1]] = TREE_COLORS[ti % len(TREE_COLORS)]
                if sc["clicks"]:
                    ov_clicks += [c for c in sc["clicks"][ti]
                                  if len(c) >= 5 and int(c[4]) <= k]
            ax = fig.add_axes([Lm, y_b, panel_w, row_h], projection="3d")
            ax.set_facecolor(BG_COLOR)
            ax.scatter(sc["P_ov"][:, 0], sc["P_ov"][:, 1], sc["P_ov"][:, 2],
                       c=col_ov, s=0.4, linewidths=0, rasterized=True, depthshade=False)
            _draw_clicks(ax, ov_clicks)
            _style_3d(ax, sc["ov_lim"], azim, elev, zoom=1.0)
            cx0 = Lm + panel_w / 2
            fig.text(cx0, y_b + row_h + title_dy, row_label,
                     color="#e1f5e1", fontsize=12.5, fontweight="bold",
                     ha="center", va="center")
            fig.text(cx0, y_b + row_h + title_dy - sub_dy + 0.04,
                     f"all trees · {k} click{'s' if k>1 else ''}",
                     color="#cfe8cf", fontsize=10, ha="center", va="center")

            # ── per-tree panels (columns 1..G) ─────────────────────────────
            for j, td in enumerate(sc["tdata"]):
                lft = Lm + (j + 1) * (panel_w + gap_x)
                ax = fig.add_axes([lft, y_b, panel_w, row_h], projection="3d")
                ax.set_facecolor(BG_COLOR)
                pz  = td["pred_sub"][k-1]
                col = error_rgba(pz, td["gz"], td["P"][:, 2])
                ax.scatter(td["P"][:, 0], td["P"][:, 1], td["P"][:, 2], c=col,
                           s=0.5, linewidths=0, rasterized=True, depthshade=False)
                clks = [c for c in sc["clicks"][j] if len(c) >= 5 and int(c[4]) <= k] \
                       if sc["clicks"] else []
                _draw_clicks(ax, clks)
                _style_3d(ax, td["lim"], azim, elev, zoom=1.05)
                cx = lft + panel_w / 2
                fig.text(cx, y_b + row_h + title_dy, f"Tree {j+1}",
                         color=td["tcolor"], fontsize=12.5, fontweight="bold",
                         ha="center", va="center")
                fig.text(cx, y_b + row_h + title_dy - sub_dy + 0.04,
                         f"IoU={td['iou_k'][k-1]:.2f} @ {k} click{'s' if k>1 else ''}",
                         color="#cfe8cf", fontsize=10, ha="center", va="center")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG_COLOR)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)
        durations.append(130)

    W = max(f.size[0] for f in frames); H = max(f.size[1] for f in frames)
    def pad(img):
        if img.size == (W, H): return img
        c = Image.new("RGB", (W, H), BG_RGB)
        c.paste(img, ((W-img.size[0])//2, (H-img.size[1])//2)); return c
    frames = [pad(f) for f in frames]

    pframes = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pframes[0].save(out_path, save_all=True, append_images=pframes[1:],
                    duration=durations, loop=0, optimize=True, disposal=2)
    print(f"  Saved {n_frames}-frame teaser-v3 → {out_path}")


if __name__ == "__main__":
    make_teaser_v3(f"{OUT}/teaser_v3.gif")
    print("Done.")
