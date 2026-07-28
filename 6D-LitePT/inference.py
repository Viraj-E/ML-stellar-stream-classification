"""Full FIRE snapshot inference for the native-6D LitePT model.

This script is intentionally separate from the train/test metric iterator.  It
loads one FIRE HDF5 snapshot, runs the trained native-6D LitePT checkpoint, and
writes particle-row probabilities.  Two inference modes are supported:

* global: one uncapped native-6D voxelisation of the whole snapshot.
* tiled: overlapping 128 kpc xyz inference windows merged with Gaussian weights.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import config as C
from data import (
    enrich_features,
    get_model_feature_names,
    make_litept_inputs,
    recompute_r_vr_from_pos_vel,
    resolve_feature_set,
)
from dataset import SpatialCellIndex, grid_sample_chunk
from network import LitePTBinarySeg


DEFAULT_RUN_DIR = Path("artifacts/native_base_6d_best")
DEFAULT_INPUT = Path("data/fire_snapshot.hdf5")
DEFAULT_OUTPUT = Path("runs/fire_m12i_native_base_tiled_multigrid_inference.hdf5")
DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def fmt_seconds(sec):
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_csv_floats(text):
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def read_meta(path):
    with open(path, "r") as f:
        return json.load(f)


def configure_from_meta(meta, activation_checkpointing=False):
    args = meta.get("args", {})
    C.MODEL_PRESET = str(meta.get("model_preset", args.get("model_preset", "native_base")))
    C.FEATURE_SET = str(meta.get("feature_set", args.get("feature_set", "all")))
    C.LABEL_REDUCE = str(meta.get("label_reduce", args.get("label_reduce", "any_positive")))
    C.EVAL_MAX_NATIVE_VOXELS_PER_CHUNK = 0
    C.MAX_NATIVE_VOXELS_PER_CHUNK = 0
    C.ACTIVATION_CHECKPOINTING = bool(activation_checkpointing)
    C.RETURN_INVERSE = False
    C.USE_AMP = False


def load_snapshot(path, part_type, center_mode, max_particles=0):
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        g = f[part_type]
        n_total = int(g["Coordinates"].shape[0])
        n = n_total if max_particles <= 0 else min(int(max_particles), n_total)
        pos_file = np.asarray(g["Coordinates"][:n], dtype=np.float32)
        vel = np.asarray(g["Velocities"][:n], dtype=np.float32)
        if "ParticleIDs" in g:
            ids = np.asarray(g["ParticleIDs"][:n], dtype=np.int64)
        else:
            ids = np.arange(n, dtype=np.int64)
        mass_name = "Mass" if "Mass" in g else ("Masses" if "Masses" in g else None)
        mass = np.asarray(g[mass_name][:n], dtype=np.float32) if mass_name else np.ones(n, dtype=np.float32)

    if center_mode == "mean":
        center_offset = pos_file.mean(axis=0, dtype=np.float64).astype(np.float32)
    elif center_mode == "none":
        center_offset = np.zeros(3, dtype=np.float32)
    else:
        raise ValueError("center_mode must be 'mean' or 'none'")

    pos = np.ascontiguousarray(pos_file - center_offset[None, :], dtype=np.float32)
    x8 = np.hstack([pos, vel, recompute_r_vr_from_pos_vel(pos, vel)]).astype(np.float32)
    x14 = enrich_features(x8)
    print(
        f"Loaded {n:,}/{n_total:,} particles from {path} in {fmt_seconds(time.perf_counter() - t0)}; "
        f"center_mode={center_mode} center_offset={center_offset.tolist()}",
        flush=True,
    )
    return x14, ids, mass, center_offset, n_total


def load_model(checkpoint_path, in_channels):
    t0 = time.perf_counter()
    model = LitePTBinarySeg(in_channels=in_channels).to(C.DEVICE)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {checkpoint_path} in {fmt_seconds(time.perf_counter() - t0)}", flush=True)
    return model


def _to_device_batch(coord_s, feat_s, grid_coord_s):
    return {
        "coord": torch.from_numpy(coord_s).float().to(C.DEVICE, non_blocking=True),
        "grid_coord": torch.from_numpy(grid_coord_s).int().to(C.DEVICE, non_blocking=True),
        "feat": torch.from_numpy(feat_s).float().to(C.DEVICE, non_blocking=True),
        "offset": torch.tensor([len(coord_s)], dtype=torch.long, device=C.DEVICE),
    }


def infer_particle_probabilities(model, x14, centre, stats, feature_set, grid_size, context_names):
    """Return raw-particle probabilities plus compact chunk diagnostics."""
    t0 = time.perf_counter()
    coord_xyz, feat, coord6, grid_size_eff = make_litept_inputs(
        x14, centre, stats, feature_set=feature_set, grid_size=grid_size
    )
    dummy_labels = np.zeros(len(x14), dtype=np.int64)
    feature_names = get_model_feature_names(feature_set)
    base_feature_names = feature_names[: feat.shape[1]]
    coord_s, feat_s, _labels, grid_coord_s, _origin_s, inverse = grid_sample_chunk(
        coord_xyz,
        feat,
        dummy_labels,
        grid_size_eff,
        label_reduce=C.LABEL_REDUCE,
        return_inverse=True,
        feature_names=base_feature_names,
        context_names=context_names,
        phase_coord=coord6,
        max_voxels=0,
        cap_mode="hash",
    )
    voxel_seconds = time.perf_counter() - t0
    batch = _to_device_batch(coord_s, feat_s, grid_coord_s)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.inference_mode():
        logits, _ = model(batch, return_dense=False)
        prob_vox = torch.softmax(logits.detach().float(), dim=1)[:, 1].cpu().numpy().astype(np.float32)
    forward_seconds = time.perf_counter() - t1
    peak_gib = 0.0
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    prob_raw = prob_vox[inverse].astype(np.float32, copy=False)
    return prob_raw, {
        "raw_particles": int(len(x14)),
        "native_voxels": int(len(coord_s)),
        "grid_size_eff": float(grid_size_eff),
        "voxel_seconds": float(voxel_seconds),
        "forward_seconds": float(forward_seconds),
        "gpu_peak_reserved_gib": float(peak_gib),
    }


def create_or_replace_group(parent, name):
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def write_base_h5(out_path, source_path, x14, ids, mass, center_offset, n_total, meta, thresholds, overwrite):
    out_path = Path(out_path)
    ensure_parent(out_path)
    mode = "w" if overwrite or not out_path.exists() else "a"
    with h5py.File(out_path, mode) as f:
        if "source" in f and not overwrite:
            if int(f.attrs.get("n_particles", -1)) != int(len(x14)):
                raise ValueError(
                    f"Existing output has n_particles={f.attrs.get('n_particles')}, "
                    f"but this run has {len(x14)}"
                )
            f.require_group("predictions")
            return out_path
        f.attrs["source_hdf5"] = str(Path(source_path).resolve())
        f.attrs["created_by"] = "fire_full_inference.py"
        f.attrs["n_particles"] = int(len(x14))
        f.attrs["source_n_particles_total"] = int(n_total)
        f.attrs["center_offset_subtracted"] = center_offset.astype(np.float32)
        f.attrs["thresholds"] = np.asarray(thresholds, dtype=np.float32)
        f.attrs["feature_set"] = str(meta.get("feature_set", "all"))
        f.attrs["model_preset"] = str(meta.get("model_preset", "native_base"))
        src = create_or_replace_group(f, "source")
        chunks_1d = min(1_000_000, len(x14))
        src.create_dataset("row_index", data=np.arange(len(x14), dtype=np.int64), chunks=(chunks_1d,))
        src.create_dataset("ParticleIDs", data=ids, chunks=(chunks_1d,), compression="gzip", compression_opts=2, shuffle=True)
        src.create_dataset("Mass", data=mass, chunks=(chunks_1d,), compression="gzip", compression_opts=2, shuffle=True)
        src.create_dataset("Coordinates", data=x14[:, 0:3], chunks=(min(250_000, len(x14)), 3), compression="gzip", compression_opts=2, shuffle=True)
        src.create_dataset("Velocities", data=x14[:, 3:6], chunks=(min(250_000, len(x14)), 3), compression="gzip", compression_opts=2, shuffle=True)
        f.attrs["meta_json"] = json.dumps(meta)
        f.require_group("predictions")
    return out_path


def write_prediction_group(out_path, group_name, probabilities, thresholds, diagnostics, extra_arrays=None):
    with h5py.File(out_path, "a") as f:
        pred_root = f.require_group("predictions")
        g = create_or_replace_group(pred_root, group_name)
        chunks_1d = min(1_000_000, len(probabilities))
        g.create_dataset(
            "probability",
            data=probabilities.astype(np.float32, copy=False),
            chunks=(chunks_1d,),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
        )
        masks = g.create_group("masks")
        for thr in thresholds:
            name = f"thr_{str(float(thr)).replace('.', 'p')}"
            masks.create_dataset(name, data=(probabilities >= float(thr)), chunks=(chunks_1d,), compression="gzip", compression_opts=1)
        if extra_arrays:
            for name, arr in extra_arrays.items():
                data = np.asarray(arr)
                g.create_dataset(name, data=data, chunks=(chunks_1d,), compression="gzip", compression_opts=2, shuffle=True)
        for key, value in diagnostics.items():
            if isinstance(value, (dict, list, tuple)):
                g.attrs[key] = json.dumps(value)
            else:
                g.attrs[key] = value


def run_global(model, x14, stats, args, context_names):
    centre = np.zeros(3, dtype=np.float32)
    t0 = time.perf_counter()
    probs, diag = infer_particle_probabilities(
        model, x14, centre, stats, args.feature_set, args.grid_size, context_names
    )
    diag.update({
        "mode": "global",
        "centre": centre.tolist(),
        "seconds_total": float(time.perf_counter() - t0),
    })
    print(
        f"Global inference: raw={diag['raw_particles']:,} voxels={diag['native_voxels']:,} "
        f"total={fmt_seconds(diag['seconds_total'])} gpu_peak={diag['gpu_peak_reserved_gib']:.2f} GiB",
        flush=True,
    )
    return probs, diag


def generate_tile_centres(x14, tile_size, stride, min_particles, strategy, max_tiles):
    pos = x14[:, 0:3]
    half = float(tile_size) / 2.0
    centres = []
    centre0 = np.zeros(3, dtype=np.float32)
    index = SpatialCellIndex(x14, cell_size=float(stride))
    central_idx = index.query_box(centre0, float(tile_size))
    if len(central_idx) >= min_particles:
        centres.append((centre0, central_idx, "central"))

    lo = np.floor((pos.min(axis=0) - half) / float(stride)) * float(stride)
    hi = np.ceil((pos.max(axis=0) + half) / float(stride)) * float(stride)
    grids = [np.arange(lo[d], hi[d] + 0.5 * float(stride), float(stride), dtype=np.float32) for d in range(3)]
    central_box = np.all(np.abs(pos) < half, axis=1)
    for cx in grids[0]:
        for cy in grids[1]:
            for cz in grids[2]:
                c = np.array([cx, cy, cz], dtype=np.float32)
                if np.allclose(c, centre0):
                    continue
                idx = index.query_box(c, float(tile_size))
                if len(idx) < int(min_particles):
                    continue
                if strategy == "central_shell" and np.all(central_box[idx]):
                    continue
                centres.append((c, idx, "shell"))
                if max_tiles > 0 and len(centres) >= max_tiles:
                    return centres
    return centres


def run_tiled(model, x14, stats, args, context_names):
    t0 = time.perf_counter()
    centres = generate_tile_centres(
        x14,
        tile_size=args.tile_size,
        stride=args.tile_stride,
        min_particles=args.tile_min_particles,
        strategy=args.tile_strategy,
        max_tiles=args.max_tiles,
    )
    n = len(x14)
    prob_sum = np.zeros(n, dtype=np.float64)
    prob2_sum = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.uint16)
    tile_rows = []
    half = float(args.tile_size) / 2.0
    sigma = float(args.gaussian_sigma_norm)
    print(f"Tiled inference: {len(centres)} tiles, tile_size={args.tile_size}, stride={args.tile_stride}, sigma_norm={sigma:.4f}", flush=True)
    for i, (centre, idx, kind) in enumerate(centres, start=1):
        tile_t0 = time.perf_counter()
        x_tile = np.ascontiguousarray(x14[idx], dtype=np.float32)
        probs, diag = infer_particle_probabilities(
            model, x_tile, centre.astype(np.float32), stats, args.feature_set, args.grid_size, context_names
        )
        rel = (x14[idx, 0:3].astype(np.float64) - centre[None, :].astype(np.float64)) / half
        dist2 = np.sum(rel * rel, axis=1)
        weights = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12)).astype(np.float64)
        prob_sum[idx] += weights * probs.astype(np.float64)
        prob2_sum[idx] += weights * probs.astype(np.float64) * probs.astype(np.float64)
        weight_sum[idx] += weights
        count[idx] = np.minimum(count[idx].astype(np.uint32) + 1, np.iinfo(np.uint16).max).astype(np.uint16)
        row = {
            "tile_index": i - 1,
            "kind": kind,
            "centre_x": float(centre[0]),
            "centre_y": float(centre[1]),
            "centre_z": float(centre[2]),
            "raw_particles": int(diag["raw_particles"]),
            "native_voxels": int(diag["native_voxels"]),
            "seconds": float(time.perf_counter() - tile_t0),
            "gpu_peak_reserved_gib": float(diag["gpu_peak_reserved_gib"]),
        }
        tile_rows.append(row)
        if i % max(1, args.progress_every) == 0 or i == len(centres):
            covered = int(np.count_nonzero(weight_sum > 0))
            print(
                f"tile {i}/{len(centres)} kind={kind} raw={row['raw_particles']:,} vox={row['native_voxels']:,} "
                f"tile_time={fmt_seconds(row['seconds'])} covered={covered:,}/{n:,} "
                f"gpu_peak={row['gpu_peak_reserved_gib']:.2f} GiB",
                flush=True,
            )

    covered = weight_sum > 0
    probs_final = np.zeros(n, dtype=np.float32)
    probs_final[covered] = (prob_sum[covered] / weight_sum[covered]).astype(np.float32)
    variance = np.zeros(n, dtype=np.float32)
    variance[covered] = np.maximum(
        prob2_sum[covered] / weight_sum[covered] - probs_final[covered].astype(np.float64) ** 2,
        0.0,
    ).astype(np.float32)
    diag = {
        "mode": "tiled_blend",
        "tile_strategy": args.tile_strategy,
        "tile_size": float(args.tile_size),
        "tile_stride": float(args.tile_stride),
        "gaussian_sigma_norm": float(sigma),
        "tiles": len(centres),
        "covered_particles": int(np.count_nonzero(covered)),
        "uncovered_particles": int(n - np.count_nonzero(covered)),
        "seconds_total": float(time.perf_counter() - t0),
        "tile_summary_json": tile_rows,
    }
    return probs_final, diag, {
        "n_tiles_seen": count,
        "blend_weight_sum": weight_sum.astype(np.float32),
        "probability_variance": variance,
    }


def write_tiles(out_path, rows):
    if not rows:
        return
    with h5py.File(out_path, "a") as f:
        pred = f["predictions/tiled_blend"]
        if "tiles" in pred:
            del pred["tiles"]
        g = pred.create_group("tiles")
        centres = np.array([[r["centre_x"], r["centre_y"], r["centre_z"]] for r in rows], dtype=np.float32)
        raw = np.array([r["raw_particles"] for r in rows], dtype=np.int64)
        vox = np.array([r["native_voxels"] for r in rows], dtype=np.int64)
        secs = np.array([r["seconds"] for r in rows], dtype=np.float32)
        kinds = np.array([r["kind"].encode("utf-8") for r in rows])
        g.create_dataset("centres", data=centres)
        g.create_dataset("raw_particles", data=raw)
        g.create_dataset("native_voxels", data=vox)
        g.create_dataset("seconds", data=secs)
        g.create_dataset("kind", data=kinds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_h5", default=str(DEFAULT_INPUT))
    parser.add_argument("--part_type", default="PartType4")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--meta", default="")
    parser.add_argument("--output_h5", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode", choices=("global", "tiled", "both"), default="both")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_particles", type=int, default=0, help="Debug cap; 0 means all particles")
    parser.add_argument("--center_mode", choices=("mean", "none"), default="mean")
    parser.add_argument("--feature_set", default="")
    parser.add_argument("--grid_size", type=float, default=4.0)
    parser.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--tile_strategy", choices=("central_shell", "all"), default="central_shell")
    parser.add_argument("--tile_size", type=float, default=128.0)
    parser.add_argument("--tile_stride", type=float, default=64.0)
    parser.add_argument("--tile_min_particles", type=int, default=1)
    parser.add_argument("--max_tiles", type=int, default=0, help="Debug cap on number of tiles; 0 means no cap")
    parser.add_argument("--gaussian_edge_weight", type=float, default=0.10)
    parser.add_argument("--gaussian_sigma_norm", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--activation_checkpointing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "litept_best_mcc.pt"
    meta_path = Path(args.meta) if args.meta else run_dir / "litept_meta.json"
    meta = read_meta(meta_path)
    configure_from_meta(meta, activation_checkpointing=args.activation_checkpointing)
    if args.feature_set:
        C.FEATURE_SET = args.feature_set
    args.feature_set = C.FEATURE_SET
    thresholds = parse_csv_floats(args.thresholds)
    if args.gaussian_sigma_norm <= 0:
        edge_weight = min(max(float(args.gaussian_edge_weight), 1e-6), 0.999)
        args.gaussian_sigma_norm = 1.0 / math.sqrt(2.0 * math.log(1.0 / edge_weight))

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Using device:  ", C.DEVICE, flush=True)
    print(f"Model preset: {C.MODEL_PRESET}; feature_set={C.FEATURE_SET}", flush=True)
    print(f"Threshold masks: {thresholds}; primary test threshold is 0.7", flush=True)

    x14, ids, mass, center_offset, n_total = load_snapshot(
        args.input_h5, args.part_type, args.center_mode, max_particles=args.max_particles
    )
    stats = meta.get("stats", {"feat_mean": np.zeros(11, dtype=np.float32), "feat_std": np.ones(11, dtype=np.float32)})
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
    feature_names = get_model_feature_names(args.feature_set)
    in_channels = int(meta.get("in_channels", len(feature_names)))
    base_feature_names = resolve_feature_set(args.feature_set)
    context_names = feature_names[len(base_feature_names) :]
    model = load_model(checkpoint, in_channels=in_channels)

    out_path = write_base_h5(
        args.output_h5,
        args.input_h5,
        x14,
        ids,
        mass,
        center_offset,
        n_total,
        meta,
        thresholds,
        overwrite=args.overwrite,
    )

    if args.mode in ("global", "both"):
        probs, diag = run_global(model, x14, stats, args, context_names)
        diag["checkpoint"] = str(checkpoint)
        diag["selection_threshold"] = 0.7
        write_prediction_group(out_path, "global", probs, thresholds, diag)

    if args.mode in ("tiled", "both"):
        probs, diag, extra = run_tiled(model, x14, stats, args, context_names)
        diag["checkpoint"] = str(checkpoint)
        diag["selection_threshold"] = 0.7
        rows = json.loads(json.dumps(diag["tile_summary_json"]))
        write_prediction_group(out_path, "tiled_blend", probs, thresholds, diag, extra_arrays=extra)
        write_tiles(out_path, rows)

    print(f"Wrote inference output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
