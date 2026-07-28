"""Second-pass FIRE particle diagnostics for LitePT tiled multigrid inference.

This script only reads the existing inference HDF5. It does not rerun LitePT.

Outputs:
  - shared-scale all-vs-selected density comparisons
  - all-particle background plus selected-particle overlays
  - projection-space connected-component instance visualisations
  - selected-particle uncertainty maps from probability_variance
  - component-aware crop sheets for the largest projected structures
"""

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


DEFAULT_INPUT = Path("runs/fire_m12i_native_base_tiled_multigrid_inference.hdf5")
DEFAULT_OUT_DIR = Path("figures/fire_simulation/diagnostics_new")

PROJECTIONS = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}

STRUCT_CMAP = LinearSegmentedColormap.from_list(
    "litept_structure_overlay",
    ["#00a7ff", "#14e2d8", "#78ff75", "#fff06a"],
)

INSTANCE_COLORS = np.array(
    [
        [0.00, 0.72, 1.00, 1.00],
        [1.00, 0.67, 0.04, 1.00],
        [0.15, 0.94, 0.42, 1.00],
        [0.95, 0.18, 0.55, 1.00],
        [0.67, 0.45, 1.00, 1.00],
        [1.00, 0.24, 0.18, 1.00],
        [0.05, 0.84, 0.72, 1.00],
        [0.84, 0.88, 0.00, 1.00],
        [0.49, 0.76, 1.00, 1.00],
        [1.00, 0.43, 0.76, 1.00],
        [0.74, 1.00, 0.43, 1.00],
        [0.99, 0.53, 0.20, 1.00],
        [0.58, 0.57, 1.00, 1.00],
        [0.98, 0.86, 0.35, 1.00],
        [0.30, 0.96, 0.95, 1.00],
        [0.93, 0.42, 1.00, 1.00],
    ],
    dtype=np.float32,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_csv(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def finite_extent(half_width):
    return (-half_width, half_width, -half_width, half_width)


def make_views():
    return {
        "full_800kpc": finite_extent(400.0),
        "inner_400kpc": finite_extent(200.0),
        "central_160kpc": finite_extent(80.0),
        "central_80kpc": finite_extent(40.0),
    }


def threshold_name(threshold):
    return str(float(threshold)).replace(".", "p")


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


def axis_setup(ax, extent, projection):
    _i, _j, xl, yl = PROJECTIONS[projection]
    ax.set_facecolor("#05000b")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(f"{xl} [kpc]")
    ax.set_ylabel(f"{yl} [kpc]")


def in_view_mask(pos, extent, projection):
    i, j, _xl, _yl = PROJECTIONS[projection]
    return (
        (pos[:, i] >= extent[0])
        & (pos[:, i] <= extent[1])
        & (pos[:, j] >= extent[2])
        & (pos[:, j] <= extent[3])
    )


def density_products(pos, selected, extent, projection, bins, smooth_sigma):
    all_density = smooth_image(hist2d(pos, extent, projection, bins), smooth_sigma)
    selected_density = smooth_image(hist2d(pos, extent, projection, bins, mask=selected), smooth_sigma)
    return all_density, selected_density, np.log1p(all_density), np.log1p(selected_density)


def selected_stats(pos, selected, probability, extent, projection):
    view = in_view_mask(pos, extent, projection)
    selected_view = view & selected
    all_count = int(view.sum())
    selected_count = int(selected_view.sum())
    selected_frac = selected_count / max(all_count, 1)
    mean_p = float(probability[selected_view].mean()) if selected_count else 0.0
    return all_count, selected_count, selected_frac, mean_p


def render_same_scale(pos, selected, probability, extent, projection, view_name, threshold, bins, smooth_sigma, out_path):
    _all_d, selected_d, all_img, selected_img = density_products(pos, selected, extent, projection, bins, smooth_sigma)
    selected_img = np.ma.masked_where(selected_d <= 0, selected_img)
    shared_vmax = robust_vmax(all_img, 0.997)
    all_count, selected_count, selected_frac, mean_p = selected_stats(pos, selected, probability, extent, projection)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax in axes:
        axis_setup(ax, extent, projection)

    im0 = axes[0].imshow(
        all_img,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=shared_vmax,
    )
    axes[0].set_title(f"all particles\nN={all_count:,}")

    im1 = axes[1].imshow(
        selected_img,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=shared_vmax,
    )
    axes[1].set_title(
        f"filtered particles: P >= {threshold:.2f}\n"
        f"N={selected_count:,} ({selected_frac:.3%}), mean P={mean_p:.3f}"
    )
    cb = fig.colorbar(im1, ax=axes, fraction=0.030, pad=0.020)
    cb.set_label("shared log(1 + particles / pixel)")
    fig.suptitle(f"shared scale | {view_name} {projection}", fontsize=13)
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    return str(out_path)


def render_overlay(pos, selected, probability, extent, projection, view_name, threshold, bins, smooth_sigma, out_path):
    _all_d, selected_d, all_img, selected_img = density_products(pos, selected, extent, projection, bins, smooth_sigma)
    selected_img = np.ma.masked_where(selected_d <= 0, selected_img)
    all_count, selected_count, selected_frac, mean_p = selected_stats(pos, selected, probability, extent, projection)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.5), constrained_layout=True)
    axis_setup(ax, extent, projection)
    ax.imshow(
        all_img,
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(all_img, 0.998),
        alpha=0.72,
    )
    im = ax.imshow(
        selected_img,
        origin="lower",
        extent=extent,
        cmap=STRUCT_CMAP,
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(selected_img.filled(0.0), 0.997),
        alpha=0.92,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label("log(1 + selected particles / pixel)")
    ax.set_title(
        f"overlay | P >= {threshold:.2f} | {view_name} {projection}\n"
        f"selected {selected_count:,}/{all_count:,} ({selected_frac:.3%}), mean P={mean_p:.3f}"
    )
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    return str(out_path)


def pixel_indices(pos, extent, projection, bins, mask):
    i, j, _xl, _yl = PROJECTIONS[projection]
    x = pos[mask, i]
    y = pos[mask, j]
    ix = np.floor((x - extent[0]) / (extent[1] - extent[0]) * bins).astype(np.int64)
    iy = np.floor((y - extent[2]) / (extent[3] - extent[2]) * bins).astype(np.int64)
    valid = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    return ix, iy, valid


def component_segmentation(
    pos,
    selected,
    probability,
    variance,
    extent,
    projection,
    bins,
    component_smooth_sigma,
    component_density_quantile,
    component_abs_cut,
    min_component_pixels,
    min_component_particles,
    max_components,
):
    selected_count = hist2d(pos, extent, projection, bins, mask=selected)
    selected_smooth = smooth_image(selected_count.astype(np.float64), component_smooth_sigma)
    positive = selected_smooth[np.isfinite(selected_smooth) & (selected_smooth > 0)]
    if positive.size == 0:
        empty = np.zeros((bins, bins), dtype=np.int32)
        return empty, [], selected_smooth

    density_cut = max(float(np.quantile(positive, component_density_quantile)), float(component_abs_cut))
    component_mask = selected_smooth >= density_cut
    labels, _num = ndi.label(component_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if labels.max() == 0:
        empty = np.zeros((bins, bins), dtype=np.int32)
        return empty, [], selected_smooth

    objects = ndi.find_objects(labels)
    keep_pixel_area = {}
    for label_value, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        pixel_area = int(np.count_nonzero(labels[obj] == label_value))
        if pixel_area >= min_component_pixels:
            keep_pixel_area[label_value] = pixel_area

    selected_view = selected & in_view_mask(pos, extent, projection)
    ix, iy, valid = pixel_indices(pos, extent, projection, bins, selected_view)
    selected_indices = np.flatnonzero(selected_view)
    selected_indices = selected_indices[valid]
    ix = ix[valid]
    iy = iy[valid]
    raw_particle_labels = labels[iy, ix]

    components = []
    for raw_label in sorted(keep_pixel_area):
        particle_sel = raw_particle_labels == raw_label
        particle_count = int(particle_sel.sum())
        if particle_count < min_component_particles:
            continue

        particle_ids = selected_indices[particle_sel]
        coords = pos[particle_ids]
        pix_y, pix_x = np.nonzero(labels == raw_label)
        if pix_x.size == 0:
            continue
        x_min = extent[0] + float(pix_x.min()) / bins * (extent[1] - extent[0])
        x_max = extent[0] + float(pix_x.max() + 1) / bins * (extent[1] - extent[0])
        y_min = extent[2] + float(pix_y.min()) / bins * (extent[3] - extent[2])
        y_max = extent[2] + float(pix_y.max() + 1) / bins * (extent[3] - extent[2])
        components.append(
            {
                "raw_label": int(raw_label),
                "particle_count": particle_count,
                "pixel_area": int(keep_pixel_area[raw_label]),
                "mean_probability": float(probability[particle_ids].mean()),
                "mean_probability_sigma": float(np.sqrt(np.maximum(variance[particle_ids], 0.0)).mean()),
                "bbox": [x_min, x_max, y_min, y_max],
                "centroid": [
                    float(coords[:, PROJECTIONS[projection][0]].mean()),
                    float(coords[:, PROJECTIONS[projection][1]].mean()),
                ],
            }
        )

    components.sort(key=lambda item: item["particle_count"], reverse=True)
    components = components[:max_components]

    display_labels = np.zeros_like(labels, dtype=np.int32)
    for instance_id, component in enumerate(components, start=1):
        display_labels[labels == component["raw_label"]] = instance_id
        component["instance_id"] = instance_id
        component["density_cut"] = density_cut

    return display_labels, components, selected_smooth


def instance_rgba(display_labels, selected_density, alpha_floor=0.20):
    rgba = np.zeros((*display_labels.shape, 4), dtype=np.float32)
    density = np.log1p(selected_density)
    vmax = robust_vmax(density, 0.995)
    alpha = np.clip(density / max(vmax, 1e-6), alpha_floor, 0.96)
    for label_id in range(1, int(display_labels.max()) + 1):
        color = INSTANCE_COLORS[(label_id - 1) % len(INSTANCE_COLORS)].copy()
        mask = display_labels == label_id
        rgba[mask, :3] = color[:3]
        rgba[mask, 3] = alpha[mask]
    return rgba


def render_instances(
    pos,
    selected,
    probability,
    extent,
    projection,
    view_name,
    threshold,
    bins,
    smooth_sigma,
    display_labels,
    components,
    out_path,
):
    all_density, selected_density, all_img, _selected_img = density_products(pos, selected, extent, projection, bins, smooth_sigma)
    _ = all_density
    rgba = instance_rgba(display_labels, selected_density)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.8), constrained_layout=True)
    axis_setup(ax, extent, projection)
    ax.imshow(
        all_img,
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(all_img, 0.998),
        alpha=0.68,
    )
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    for component in components[:10]:
        cx, cy = component["centroid"]
        color = INSTANCE_COLORS[(component["instance_id"] - 1) % len(INSTANCE_COLORS)]
        ax.text(
            cx,
            cy,
            str(component["instance_id"]),
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": color, "edgecolor": "white", "linewidth": 0.5},
        )
    summary = ", ".join(f"{c['instance_id']}:{c['particle_count']:,}" for c in components[:6])
    if not summary:
        summary = "no projected components above filters"
    ax.set_title(
        f"projection-space instances | P >= {threshold:.2f} | {view_name} {projection}\n"
        f"component counts: {summary}"
    )
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    return str(out_path)


def render_uncertainty(pos, selected, variance, extent, projection, view_name, threshold, bins, smooth_sigma, out_path):
    all_density = smooth_image(hist2d(pos, extent, projection, bins), smooth_sigma)
    selected_density = smooth_image(hist2d(pos, extent, projection, bins, mask=selected), smooth_sigma)
    variance_sum = smooth_image(hist2d(pos, extent, projection, bins, weights=variance, mask=selected), smooth_sigma)
    mean_variance = np.divide(
        variance_sum,
        selected_density,
        out=np.zeros_like(variance_sum),
        where=selected_density > 0,
    )
    sigma = np.sqrt(np.maximum(mean_variance, 0.0))
    sigma_masked = np.ma.masked_where(selected_density <= 0, sigma)
    selected_log = np.log1p(selected_density)
    alpha = np.clip(selected_log / max(robust_vmax(selected_log, 0.995), 1e-6), 0.20, 0.92)
    alpha[selected_density <= 0] = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.5), constrained_layout=True)
    axis_setup(ax, extent, projection)
    ax.imshow(
        np.log1p(all_density),
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(np.log1p(all_density), 0.998),
        alpha=0.65,
    )
    im = ax.imshow(
        sigma_masked,
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
        vmin=0.0,
        vmax=robust_vmax(sigma_masked.filled(0.0), 0.995),
        alpha=alpha,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label("mean sigma(P) across tiled predictions")
    ax.set_title(f"uncertainty overlay | P >= {threshold:.2f} | {view_name} {projection}")
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    return str(out_path)


def crop_extent(component, source_extent, pad_fraction=0.32, min_half_width=18.0, max_half_width=170.0):
    x_min, x_max, y_min, y_max = component["bbox"]
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    half = max(min_half_width, half * (1.0 + pad_fraction))
    half = min(max_half_width, half)
    x0 = max(source_extent[0], cx - half)
    x1 = min(source_extent[1], cx + half)
    y0 = max(source_extent[2], cy - half)
    y1 = min(source_extent[3], cy + half)
    return (float(x0), float(x1), float(y0), float(y1))


def render_component_crops(
    pos,
    selected,
    projection,
    parent_view_name,
    parent_extent,
    threshold,
    components,
    out_path,
    crop_bins,
    smooth_sigma,
    max_crops,
):
    selected_components = components[:max_crops]
    if not selected_components:
        return None

    ncols = 3
    nrows = int(math.ceil(len(selected_components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    i, j, xl, yl = PROJECTIONS[projection]
    for ax, component in zip(axes, selected_components):
        extent = crop_extent(component, parent_extent)
        component_box = component["bbox"]
        in_crop = (
            (pos[:, i] >= extent[0])
            & (pos[:, i] <= extent[1])
            & (pos[:, j] >= extent[2])
            & (pos[:, j] <= extent[3])
        )
        in_component_bbox = (
            (pos[:, i] >= component_box[0])
            & (pos[:, i] <= component_box[1])
            & (pos[:, j] >= component_box[2])
            & (pos[:, j] <= component_box[3])
        )
        component_mask = selected & in_crop & in_component_bbox
        bg = smooth_image(hist2d(pos, extent, projection, crop_bins, mask=in_crop), smooth_sigma)
        comp = smooth_image(hist2d(pos, extent, projection, crop_bins, mask=component_mask), smooth_sigma)

        axis_setup(ax, extent, projection)
        ax.imshow(
            np.log1p(bg),
            origin="lower",
            extent=extent,
            cmap="gray",
            interpolation="nearest",
            vmin=0.0,
            vmax=robust_vmax(np.log1p(bg), 0.998),
            alpha=0.70,
        )
        comp_log = np.ma.masked_where(comp <= 0, np.log1p(comp))
        color = INSTANCE_COLORS[(component["instance_id"] - 1) % len(INSTANCE_COLORS)]
        cmap = LinearSegmentedColormap.from_list(
            f"component_{component['instance_id']}",
            [(color[0], color[1], color[2], 0.25), (color[0], color[1], color[2], 1.0)],
        )
        ax.imshow(
            comp_log,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=0.0,
            vmax=robust_vmax(comp_log.filled(0.0), 0.997),
            alpha=0.95,
        )
        ax.set_title(
            f"component {component['instance_id']} | {component['particle_count']:,} particles\n"
            f"mean P={component['mean_probability']:.3f}, sigma(P)={component['mean_probability_sigma']:.3f}"
        )

    fig.suptitle(
        f"component-aware crops | {parent_view_name} {projection} | P >= {threshold:.2f}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(out_path)


def write_component_rows(rows, out_path):
    if not rows:
        return
    fields = [
        "prediction_group",
        "threshold",
        "view",
        "projection",
        "instance_id",
        "particle_count",
        "pixel_area",
        "mean_probability",
        "mean_probability_sigma",
        "bbox_x_min",
        "bbox_x_max",
        "bbox_y_min",
        "bbox_y_max",
        "centroid_x",
        "centroid_y",
        "density_cut",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def component_rows(prediction_group, threshold, view_name, projection, components):
    rows = []
    for component in components:
        bbox = component["bbox"]
        centroid = component["centroid"]
        rows.append(
            {
                "prediction_group": prediction_group,
                "threshold": threshold,
                "view": view_name,
                "projection": projection,
                "instance_id": component["instance_id"],
                "particle_count": component["particle_count"],
                "pixel_area": component["pixel_area"],
                "mean_probability": f"{component['mean_probability']:.6f}",
                "mean_probability_sigma": f"{component['mean_probability_sigma']:.6f}",
                "bbox_x_min": f"{bbox[0]:.6f}",
                "bbox_x_max": f"{bbox[1]:.6f}",
                "bbox_y_min": f"{bbox[2]:.6f}",
                "bbox_y_max": f"{bbox[3]:.6f}",
                "centroid_x": f"{centroid[0]:.6f}",
                "centroid_y": f"{centroid[1]:.6f}",
                "density_cut": f"{component['density_cut']:.8f}",
            }
        )
    return rows


def make_contact_sheet(image_paths, out_path, title, thumb_width=420):
    image_paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not image_paths:
        return None
    thumbs = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            scale = thumb_width / img.width
            thumb = img.resize((thumb_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
            thumbs.append((path, thumb))

    label_height = 36
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    cell_h = max(t.height for _p, t in thumbs) + label_height
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((10, 8), title, fill="black", font=font)

    for idx, (path, thumb) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * thumb_width
        y = label_height + row * cell_h
        sheet.paste(thumb, (x, y + label_height))
        label = path.name
        if len(label) > 64:
            label = label[:61] + "..."
        draw.text((x + 6, y + 8), label, fill="black", font=small)

    ensure_dir(Path(out_path).parent)
    sheet.save(out_path)
    return str(out_path)


def write_readme(out_dir, input_h5, prediction_group, threshold, figures, component_table):
    readme = Path(out_dir) / "README.md"
    text = f"""# 6D-LitePT particle diagnostics

Input HDF5: `{Path(input_h5).resolve()}`

Prediction group: `{prediction_group}`

Main threshold: `P >= {threshold:.2f}`

This directory implements the requested follow-up plots:

1. `same_scale/`: left/right density panels use the same colour scale, so the filtered map is not visually boosted relative to all particles.
2. `overlay/`: all particles are a grey background and selected particles are coloured on top.
3. `instances/`: projection-space connected components of selected-particle density. Colours are component IDs in that projection/view, not physical labels guaranteed across projections.
4. `component_crops/`: crops centred on the largest projected components, ignoring tiny/noisy islands through minimum pixel and particle filters.
5. `uncertainty/`: selected-particle overlays coloured by mean `sigma(P) = sqrt(probability_variance)` from the tiled/multigrid inference.

The component catalogue is written to `{Path(component_table).name}`.

Contact sheets are included at the top level for quick inspection.
"""
    readme.write_text(text)
    return str(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_h5", default=str(DEFAULT_INPUT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--prediction_group", default="tiled_blend_multigrid")
    parser.add_argument("--thresholds", default="0.7")
    parser.add_argument("--projections", default="xy,xz,yz")
    parser.add_argument("--bins", type=int, default=1200)
    parser.add_argument("--crop_bins", type=int, default=700)
    parser.add_argument("--smooth_sigma", type=float, default=1.6)
    parser.add_argument("--component_smooth_sigma", type=float, default=2.2)
    parser.add_argument("--component_density_quantile", type=float, default=0.62)
    parser.add_argument("--component_abs_cut", type=float, default=0.010)
    parser.add_argument("--min_component_pixels", type=int, default=20)
    parser.add_argument("--min_component_particles", type=int, default=70)
    parser.add_argument("--max_components", type=int, default=16)
    parser.add_argument("--max_crops", type=int, default=9)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    views = make_views()
    thresholds = parse_thresholds(args.thresholds)
    projections = parse_csv(args.projections)

    with h5py.File(args.input_h5, "r") as f:
        pos = np.asarray(f["source/Coordinates"][:], dtype=np.float32)
        pred = f[f"predictions/{args.prediction_group}"]
        probability = np.asarray(pred["probability"][:], dtype=np.float32)
        if "probability_variance" in pred:
            variance = np.asarray(pred["probability_variance"][:], dtype=np.float32)
        else:
            variance = np.zeros_like(probability, dtype=np.float32)

    produced = {
        "same_scale": [],
        "overlay": [],
        "instances": [],
        "uncertainty": [],
        "component_crops": [],
        "contact_sheets": [],
    }
    all_component_rows = []

    for threshold in thresholds:
        selected = probability >= threshold
        thr = threshold_name(threshold)
        print(
            f"Plotting diagnostics for {args.prediction_group}, threshold={threshold:.3f}, "
            f"selected={int(selected.sum()):,}/{len(selected):,}",
            flush=True,
        )
        for view_name, extent in views.items():
            for projection in projections:
                prefix = f"{args.prediction_group}_{view_name}_{projection}_thr{thr}"
                same_path = out_dir / "same_scale" / f"{prefix}_same_scale.png"
                overlay_path = out_dir / "overlay" / f"{prefix}_overlay.png"
                instance_path = out_dir / "instances" / f"{prefix}_instances.png"
                uncertainty_path = out_dir / "uncertainty" / f"{prefix}_uncertainty.png"
                ensure_dir(same_path.parent)
                ensure_dir(overlay_path.parent)
                ensure_dir(instance_path.parent)
                ensure_dir(uncertainty_path.parent)

                print(f"  {view_name} {projection}", flush=True)
                produced["same_scale"].append(
                    render_same_scale(
                        pos,
                        selected,
                        probability,
                        extent,
                        projection,
                        view_name,
                        threshold,
                        args.bins,
                        args.smooth_sigma,
                        same_path,
                    )
                )
                produced["overlay"].append(
                    render_overlay(
                        pos,
                        selected,
                        probability,
                        extent,
                        projection,
                        view_name,
                        threshold,
                        args.bins,
                        args.smooth_sigma,
                        overlay_path,
                    )
                )

                display_labels, components, _selected_smooth = component_segmentation(
                    pos,
                    selected,
                    probability,
                    variance,
                    extent,
                    projection,
                    args.bins,
                    args.component_smooth_sigma,
                    args.component_density_quantile,
                    args.component_abs_cut,
                    args.min_component_pixels,
                    args.min_component_particles,
                    args.max_components,
                )
                all_component_rows.extend(
                    component_rows(args.prediction_group, threshold, view_name, projection, components)
                )
                produced["instances"].append(
                    render_instances(
                        pos,
                        selected,
                        probability,
                        extent,
                        projection,
                        view_name,
                        threshold,
                        args.bins,
                        args.smooth_sigma,
                        display_labels,
                        components,
                        instance_path,
                    )
                )
                produced["uncertainty"].append(
                    render_uncertainty(
                        pos,
                        selected,
                        variance,
                        extent,
                        projection,
                        view_name,
                        threshold,
                        args.bins,
                        args.smooth_sigma,
                        uncertainty_path,
                    )
                )

                if view_name in {"full_800kpc", "inner_400kpc"}:
                    crop_path = out_dir / "component_crops" / f"{prefix}_component_crops.png"
                    ensure_dir(crop_path.parent)
                    crop_result = render_component_crops(
                        pos,
                        selected,
                        projection,
                        view_name,
                        extent,
                        threshold,
                        components,
                        crop_path,
                        args.crop_bins,
                        args.smooth_sigma,
                        args.max_crops,
                    )
                    if crop_result:
                        produced["component_crops"].append(crop_result)

    component_table = out_dir / "component_catalogue.csv"
    write_component_rows(all_component_rows, component_table)

    for key in ["same_scale", "overlay", "instances", "uncertainty", "component_crops"]:
        contact = make_contact_sheet(
            produced[key],
            out_dir / f"{key}_contact_sheet.png",
            f"FIRE m12i LitePT {args.prediction_group} | {key}",
        )
        if contact:
            produced["contact_sheets"].append(contact)

    readme_path = write_readme(
        out_dir,
        args.input_h5,
        args.prediction_group,
        thresholds[0],
        produced,
        component_table,
    )

    manifest = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "prediction_group": args.prediction_group,
        "thresholds": thresholds,
        "bins": int(args.bins),
        "crop_bins": int(args.crop_bins),
        "smooth_sigma_pixels": float(args.smooth_sigma),
        "component_smooth_sigma_pixels": float(args.component_smooth_sigma),
        "component_density_quantile": float(args.component_density_quantile),
        "component_abs_cut": float(args.component_abs_cut),
        "min_component_pixels": int(args.min_component_pixels),
        "min_component_particles": int(args.min_component_particles),
        "max_components": int(args.max_components),
        "max_crops": int(args.max_crops),
        "readme": readme_path,
        "component_catalogue": str(component_table),
        "figures": produced,
    }
    manifest_path = out_dir / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote diagnostics to {out_dir.resolve()}", flush=True)
    print(f"Wrote component catalogue: {component_table}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
