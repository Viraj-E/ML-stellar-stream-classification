"""Side-by-side FIRE particle-density plots before and after LitePT filtering."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("runs/fire_m12i_native_base_tiled_multigrid_inference.hdf5")
DEFAULT_OUT_DIR = Path("figures/fire_simulation/side_by_side_particle_filter_new")

PROJECTIONS = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_csv(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def finite_extent(half_width):
    return (-half_width, half_width, -half_width, half_width)


def hist2d(pos, extent, projection, bins, mask=None):
    i, j, _xl, _yl = PROJECTIONS[projection]
    if mask is None:
        x = pos[:, i]
        y = pos[:, j]
    else:
        x = pos[mask, i]
        y = pos[mask, j]
    h, _xedges, _yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    return h.T


def gaussian_kernel1d(sigma):
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_image(img, sigma):
    sigma = float(sigma)
    if sigma <= 0:
        return img.astype(np.float64, copy=False)
    kernel = gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=img)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=tmp)


def robust_vmax(img, q=0.997):
    values = np.asarray(img)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    return max(float(np.quantile(values, q)), 1e-6)


def render_side_by_side(pos, selected, probability, extent, projection, view_name, threshold, bins, smooth_sigma, out_path):
    all_density = smooth_image(hist2d(pos, extent, projection, bins), smooth_sigma)
    selected_density = smooth_image(hist2d(pos, extent, projection, bins, mask=selected), smooth_sigma)
    all_img = np.log1p(all_density)
    selected_img = np.ma.masked_where(selected_density <= 0, np.log1p(selected_density))

    i, j, xl, yl = PROJECTIONS[projection]
    in_view = (
        (pos[:, i] >= extent[0])
        & (pos[:, i] <= extent[1])
        & (pos[:, j] >= extent[2])
        & (pos[:, j] <= extent[3])
    )
    selected_in_view = in_view & selected
    all_count = int(in_view.sum())
    selected_count = int(selected_in_view.sum())
    selected_frac = selected_count / max(all_count, 1)
    mean_selected_p = float(probability[selected_in_view].mean()) if selected_count else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax in axes:
        ax.set_facecolor("#070010")
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(f"{xl} [kpc]")
        ax.set_ylabel(f"{yl} [kpc]")

    im0 = axes[0].imshow(
        all_img,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(all_img),
    )
    axes[0].set_title(f"all particles\nN={all_count:,}")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.025)
    cb0.set_label("log(1 + particle density)")

    im1 = axes[1].imshow(
        selected_img,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(selected_img.filled(0.0)),
    )
    axes[1].set_title(
        f"filtered particles: P >= {threshold:.2f}\n"
        f"N={selected_count:,} ({selected_frac:.3%}), mean P={mean_selected_p:.3f}"
    )
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.025)
    cb1.set_label("log(1 + selected-particle density)")

    fig.suptitle(
        f"FIRE m12i LitePT tiled multigrid particle filter | {view_name} {projection}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_h5", default=str(DEFAULT_INPUT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--prediction_group", default="tiled_blend_multigrid")
    parser.add_argument("--thresholds", default="0.7")
    parser.add_argument("--projections", default="xy,xz,yz")
    parser.add_argument("--bins", type=int, default=1400)
    parser.add_argument("--smooth_sigma", type=float, default=1.6)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    views = {
        "full_800kpc": finite_extent(400.0),
        "inner_400kpc": finite_extent(200.0),
        "central_160kpc": finite_extent(80.0),
        "central_80kpc": finite_extent(40.0),
    }
    produced = []
    with h5py.File(args.input_h5, "r") as f:
        pos = np.asarray(f["source/Coordinates"][:], dtype=np.float32)
        probability = np.asarray(f[f"predictions/{args.prediction_group}/probability"][:], dtype=np.float32)

    for threshold in parse_thresholds(args.thresholds):
        selected = probability >= threshold
        thr_name = str(float(threshold)).replace(".", "p")
        for view_name, extent in views.items():
            for projection in parse_csv(args.projections):
                out_path = (
                    Path(args.out_dir)
                    / args.prediction_group
                    / f"{args.prediction_group}_{view_name}_{projection}_all_vs_selected_thr{thr_name}.png"
                )
                ensure_dir(out_path.parent)
                print(f"Rendering {out_path}", flush=True)
                render_side_by_side(
                    pos,
                    selected,
                    probability,
                    extent,
                    projection,
                    view_name,
                    threshold,
                    args.bins,
                    args.smooth_sigma,
                    out_path,
                )
                produced.append(str(out_path))

    manifest = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "prediction_group": args.prediction_group,
        "thresholds": parse_thresholds(args.thresholds),
        "bins": int(args.bins),
        "smooth_sigma_pixels": float(args.smooth_sigma),
        "figures": produced,
    }
    manifest_path = Path(args.out_dir) / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(produced)} side-by-side figures and manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
