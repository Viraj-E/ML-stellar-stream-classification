"""Refined tiled-only FIRE visualisations for LitePT inference outputs."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_INPUT = Path("runs/fire_m12i_native_base_tiled_multigrid_inference.hdf5")
DEFAULT_OUT_DIR = Path("figures/fire_simulation/tiled_refined_new")

PROJECTIONS = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}

STRUCT_CMAP = LinearSegmentedColormap.from_list(
    "structure_blue_green_yellow",
    ["#0087ff", "#00d6d6", "#4cff7a", "#fff45c"],
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_csv(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def finite_extent(half_width):
    return (-half_width, half_width, -half_width, half_width)


def auto_stream_extent(pos, prob, threshold, projection, min_half_width=35.0, pad_fraction=0.18):
    i, j, _xl, _yl = PROJECTIONS[projection]
    r = np.linalg.norm(pos, axis=1)
    mask = (prob >= threshold) & (r > 25.0)
    if int(mask.sum()) < 100:
        mask = prob >= np.quantile(prob, 0.995)
    if int(mask.sum()) < 10:
        return finite_extent(160.0)
    xy = pos[mask][:, [i, j]]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    cen = 0.5 * (lo + hi)
    half = max(float(0.5 * np.max(hi - lo) * (1.0 + pad_fraction)), float(min_half_width))
    return (float(cen[0] - half), float(cen[0] + half), float(cen[1] - half), float(cen[1] + half))


def hist2d(pos, extent, projection, bins, weights=None, mask=None):
    i, j, _xl, _yl = PROJECTIONS[projection]
    if mask is None:
        x = pos[:, i]
        y = pos[:, j]
        w = weights
    else:
        x = pos[mask, i]
        y = pos[mask, j]
        w = None if weights is None else weights[mask]
    h, _xedges, _yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
        weights=w,
    )
    return h.T


def gaussian_kernel1d(sigma):
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def smooth_image(img, sigma):
    if sigma <= 0:
        return img
    k = gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=tmp)


def robust_vmax(img, q=0.995):
    vals = np.asarray(img)[np.isfinite(img)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.quantile(vals, q))


def add_panel_colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(label)


def render_refined(pos, prob, mode, projection, view_name, extent, threshold, bins, smooth_sigma, out_path):
    selected = prob >= threshold
    bg = hist2d(pos, extent, projection, bins)
    selected_count = hist2d(pos, extent, projection, bins, mask=selected)
    prob_weighted = hist2d(pos, extent, projection, bins, weights=prob)
    prob_sum = hist2d(pos, extent, projection, bins, weights=prob)

    bg_s = smooth_image(bg.astype(np.float64), smooth_sigma)
    selected_s = smooth_image(selected_count.astype(np.float64), smooth_sigma)
    weighted_s = smooth_image(prob_weighted.astype(np.float64), smooth_sigma)
    prob_sum_s = smooth_image(prob_sum.astype(np.float64), smooth_sigma)
    bg_for_mean = smooth_image(bg.astype(np.float64), smooth_sigma)
    mean_prob_s = np.divide(prob_sum_s, bg_for_mean, out=np.zeros_like(prob_sum_s), where=bg_for_mean > 0)

    bg_log = np.log1p(bg_s)
    selected_log = np.ma.masked_where(selected_s <= 0, np.log1p(selected_s))
    weighted_log = np.ma.masked_where(weighted_s <= 0, np.log1p(weighted_s))
    mean_prob = np.ma.masked_where(bg_for_mean <= 0, mean_prob_s)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor("#080012")
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    im0 = axes[0].imshow(
        bg_log,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(bg_log, 0.997),
    )
    axes[0].set_title("all-star projected density")
    add_panel_colorbar(fig, axes[0], im0, "log(1 + all stars / pixel)")

    axes[1].imshow(bg_log, origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.38)
    im1 = axes[1].imshow(
        selected_log,
        origin="lower",
        extent=extent,
        cmap=STRUCT_CMAP,
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(selected_log.filled(0.0), 0.997),
        alpha=0.94,
    )
    axes[1].set_title(f"hard selected structure density | P >= {threshold:.2f}")
    add_panel_colorbar(fig, axes[1], im1, "log(1 + selected particles / pixel)")

    axes[2].imshow(bg_log, origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.30)
    im2 = axes[2].imshow(
        weighted_log,
        origin="lower",
        extent=extent,
        cmap=STRUCT_CMAP,
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(weighted_log.filled(0.0), 0.997),
        alpha=0.95,
    )
    axes[2].set_title("probability-weighted structure density")
    add_panel_colorbar(fig, axes[2], im2, "log(1 + sum P / pixel)")

    axes[3].imshow(bg_log, origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.36)
    im3 = axes[3].imshow(
        mean_prob,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        alpha=0.95,
    )
    axes[3].set_title(f"smoothed mean probability | sigma={smooth_sigma:g} px")
    add_panel_colorbar(fig, axes[3], im3, "mean P(structure)")

    _i, _j, xl, yl = PROJECTIONS[projection]
    for ax in axes:
        ax.set_xlabel(f"{xl} [kpc]")
        ax.set_ylabel(f"{yl} [kpc]")

    fig.suptitle(f"FIRE m12i LitePT native_base best-MCC | {mode} | {view_name} {projection}", fontsize=13)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_views(pos, prob, threshold):
    views = {
        "full_800kpc": finite_extent(400.0),
        "inner_400kpc": finite_extent(200.0),
        "central_160kpc": finite_extent(80.0),
        "central_80kpc": finite_extent(40.0),
    }
    for projection in PROJECTIONS:
        views[f"adaptive_stream_{projection}"] = auto_stream_extent(pos, prob, threshold, projection)
    return views


def render_mode(pos, prob, mode, threshold, out_dir, bins, smooth_sigma):
    views = build_views(pos, prob, threshold)
    produced = []
    for view_name, extent in views.items():
        for projection in PROJECTIONS:
            if view_name.startswith("adaptive_stream_") and not view_name.endswith(projection):
                continue
            out_path = Path(out_dir) / mode / f"{mode}_{view_name}_{projection}_refined_thr{str(threshold).replace('.', 'p')}.png"
            ensure_dir(out_path.parent)
            render_refined(pos, prob, mode, projection, view_name, extent, threshold, bins, smooth_sigma, out_path)
            produced.append(str(out_path))
    return produced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_h5", default=str(DEFAULT_INPUT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--modes", default="tiled_blend_multigrid")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--bins", type=int, default=1200)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    produced = []
    with h5py.File(args.input_h5, "r") as f:
        pos = np.asarray(f["source/Coordinates"][:], dtype=np.float32)
        pred_root = f["predictions"]
        for mode in parse_csv(args.modes):
            if mode not in pred_root:
                print(f"Skipping missing prediction mode: {mode}", flush=True)
                continue
            prob = np.asarray(pred_root[mode]["probability"][:], dtype=np.float32)
            print(f"Plotting refined mode={mode} particles={len(prob):,} threshold={args.threshold}", flush=True)
            produced.extend(render_mode(pos, prob, mode, args.threshold, args.out_dir, args.bins, args.smooth_sigma))

    manifest = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "modes": parse_csv(args.modes),
        "threshold": float(args.threshold),
        "bins": int(args.bins),
        "smooth_sigma_pixels": float(args.smooth_sigma),
        "panels": [
            "all-star projected density",
            "hard selected structure density with colourbar",
            "probability-weighted structure density",
            "smoothed mean probability",
        ],
        "figures": produced,
    }
    manifest_path = Path(args.out_dir) / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(produced)} refined figures and manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
