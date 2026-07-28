
"""
Data loading, physically consistent augmentations, dimensionless feature
construction, and run utilities for the isolated LitePT experiment.
"""

import os
import time
import json
from contextlib import contextmanager

import numpy as np
import torch

try:
    import h5py
except Exception:  # The local Apptainer image used here may not include h5py.
    h5py = None

import config as C


def now():
    return time.perf_counter()


def fmt(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    if sec < 3600:
        return f"{sec / 60:.2f}m"
    return f"{sec / 3600:.2f}h"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def base_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def ascontig_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def load_npy(path):
    return np.load(path, mmap_mode="r") if C.USE_MEMMAP else np.load(path)


def stats_to_jsonable(stats):
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


@contextmanager
def dummy_context():
    yield


def autocast_context():
    if C.USE_AMP and C.DEVICE == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast()
    return dummy_context()


def make_grad_scaler():
    if C.USE_AMP and C.DEVICE == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.GradScaler()
    return None


class xlogger:
    """Pipe-delimited logger. One call to write(row) appends one flushed row."""

    def __init__(self, filename, sep="|", lineterminator="\n"):
        self.f = filename
        self.sep = sep
        self.end = lineterminator
        ensure_dir(os.path.dirname(filename))
        if os.path.exists(self.f):
            head, tail = os.path.split(self.f)
            stamp = time.strftime("%d-%m-%Y::%Hh-%Mm-%S")
            self.f = os.path.join(head, tail.replace(".dat", f"_copy_on_{stamp}.dat"))
            print(f"Warning: log exists; writing to {self.f}")

    def _write(self, values, mode):
        with open(self.f, mode) as ff:
            print(*values, file=ff, flush=True, sep=self.sep, end=self.end)

    def write(self, row):
        row = dict(row)
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                row[k] = v.tolist()
        if not os.path.exists(self.f):
            self._write(list(row.keys()), "a")
        self._write(list(row.values()), "a")


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------- file discovery --------------------------------
def configure_training_validation_root(tv_root):
    tv_root = os.path.abspath(tv_root)
    C.TV_ROOT = tv_root
    C.TRAIN_STREAM_DIR = os.path.join(tv_root, "training", "streams")
    C.VAL_STREAM_DIR = os.path.join(tv_root, "validation", "streams")
    C.TRAIN_BACK_DIR = os.path.join(tv_root, "training", "background")
    C.VAL_BACK_DIR = os.path.join(tv_root, "validation", "background")
    C.TEST_STREAM_DIR = os.path.join(tv_root, "testing", "streams")
    C.TEST_BACK_DIR = os.path.join(tv_root, "testing", "background")
    C.STREAM_DIR = C.TRAIN_STREAM_DIR
    C.BACK_DIR = C.TRAIN_BACK_DIR


def list_stream_files(split="train"):
    d = C.TRAIN_STREAM_DIR if split in ("train", "training") else (C.TEST_STREAM_DIR if split in ("test", "testing") else C.VAL_STREAM_DIR)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy"))


def list_background_files(split="train"):
    d = C.TRAIN_BACK_DIR if split in ("train", "training") else (C.TEST_BACK_DIR if split in ("test", "testing") else C.VAL_BACK_DIR)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith((".npy", ".hdf5", ".h5")))


def parse_large_background_names(names):
    if names is None:
        return set()
    names = str(names).strip()
    if not names:
        return set()
    return set(x.strip() for x in names.split(",") if x.strip())


def _require_h5(path):
    if h5py is None:
        raise ImportError(f"h5py is required to read {path}, but it is not available in this container")


def get_file_nrows(path, part_type=C.BACKGROUND_PART_TYPE):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return int(load_npy(path).shape[0])
    if ext in (".hdf5", ".h5"):
        _require_h5(path)
        with h5py.File(path, "r") as f:
            return int(f[part_type]["Coordinates"].shape[0])
    raise ValueError(f"Unsupported file format: {path}")


def format_int(n):
    return "{:,}".format(int(n))


def proportional_counts(cap, sizes, mode, min_per_nonzero=1):
    sizes = np.asarray(sizes, dtype=np.float64)
    if len(sizes) == 0:
        return np.zeros(0, dtype=np.int64)
    mode = str(mode).lower()
    if mode == "equal_files":
        weights = np.ones_like(sizes)
    elif mode == "sqrt_size":
        weights = np.sqrt(np.maximum(sizes, 1.0))
    elif mode == "proportional":
        weights = np.maximum(sizes, 1.0)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    weights = weights / weights.sum()
    raw = int(cap) * weights
    counts = np.floor(raw).astype(np.int64)
    counts = np.minimum(counts, sizes.astype(np.int64))
    leftover = int(cap) - int(counts.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    for j in order:
        if leftover <= 0:
            break
        if counts[j] < sizes[j]:
            counts[j] += 1
            leftover -= 1
    if min_per_nonzero > 0 and int(cap) > 0:
        eligible = np.where(sizes >= min_per_nonzero)[0]
        for j in eligible:
            if counts[j] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > min_per_nonzero:
                    counts[donor] -= 1
                    counts[j] += 1
    return counts.astype(np.int64)


def background_counts_full_small_files(cap, paths, sizes, mode, full_small_files=False, large_file_count=2, large_file_names=""):
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if not full_small_files:
        return proportional_counts(cap, sizes_arr, mode, min_per_nonzero=1)
    n_files = len(paths)
    if n_files == 0:
        return np.zeros(0, dtype=np.int64)
    explicit = parse_large_background_names(large_file_names)
    large_mask = np.zeros(n_files, dtype=bool)
    if explicit:
        for i, path in enumerate(paths):
            b = os.path.basename(path)
            stem = base_stem(path)
            if b in explicit or stem in explicit:
                large_mask[i] = True
        if not np.any(large_mask):
            raise ValueError("--large_background_file_names did not match any background files")
    else:
        n_large = min(int(large_file_count), n_files)
        if n_large > 0:
            large_mask[np.argsort(-sizes_arr)[:n_large]] = True
    counts = sizes_arr.copy()
    large_idx = np.where(large_mask)[0]
    if len(large_idx) > 0:
        counts[large_idx] = proportional_counts(cap, sizes_arr[large_idx], mode, min_per_nonzero=1)
    return counts.astype(np.int64)


def sample_rows_by_count(arr, count, rng):
    arr = np.asarray(arr, dtype=np.float32)
    count = int(count)
    if count >= len(arr):
        return arr
    if count <= 0:
        return arr[:0]
    return np.asarray(arr[rng.choice(len(arr), size=count, replace=False)], dtype=np.float32)


# ----------------------------- loaders ---------------------------------------
def recompute_r_vr_from_pos_vel(pos, vel):
    r = np.linalg.norm(pos, axis=1)
    vr = np.sum(pos * vel, axis=1) / np.where(r == 0, 1e-8, r)
    return np.column_stack([r, vr]).astype(np.float32)


def load_stream_file(path):
    arr = ascontig_f32(load_npy(path))
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"Stream npy must be (N, >=8): {path}")
    return arr[:, :8]


def load_background_file(path, part_type=C.BACKGROUND_PART_TYPE):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = ascontig_f32(load_npy(path))
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError(f"Background npy must be (N, >=8): {path}")
        return arr[:, :8]
    if ext in (".hdf5", ".h5"):
        _require_h5(path)
        with h5py.File(path, "r") as f:
            pos = np.array(f[part_type]["Coordinates"][:], dtype=np.float32, order="C")
            vel = np.array(f[part_type]["Velocities"][:], dtype=np.float32, order="C")
        return np.hstack([pos, vel, recompute_r_vr_from_pos_vel(pos, vel)]).astype(np.float32)
    raise ValueError(f"Unsupported background format: {path}")


def _read_ids_from_group(g, n):
    for name in C.ID_DATASET_NAMES:
        if name in g:
            ids = np.asarray(g[name][:])
            return ids.astype(np.int64, copy=False) if np.issubdtype(ids.dtype, np.integer) else ids
    return np.arange(n, dtype=np.int64)


def load_test_snapshot(path, part_type=C.TEST_PART_TYPE, pos_scale=C.TEST_POS_SCALE):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".hdf5", ".h5"):
        _require_h5(path)
        with h5py.File(path, "r") as f:
            g = f[part_type]
            pos0 = np.array(g["Coordinates"][:], dtype=np.float32, order="C")
            _, idx = np.unique(pos0, axis=0, return_index=True)
            idx = np.sort(idx)
            pos = pos0[idx] * np.float32(pos_scale)
            centre = pos.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
            pos = pos - centre
            vel = np.array(g["Velocities"][:], dtype=np.float32, order="C")[idx]
            mass = np.array(g["Mass"][:], dtype=np.float32, order="C")[idx] if "Mass" in g else np.ones(len(pos), dtype=np.float32)
            particle_ids = _read_ids_from_group(g, len(pos0))[idx]
        X8 = np.hstack([pos, vel, recompute_r_vr_from_pos_vel(pos, vel)]).astype(np.float32)
        return X8, mass.astype(np.float32), particle_ids
    if ext == ".npy":
        arr = ascontig_f32(load_npy(path))
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError(f"Test npy must be (N, >=8): {path}")
        X8 = arr[:, :8].astype(np.float32)
        X8[:, 0:3] -= X8[:, 0:3].mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
        X8[:, 6:8] = recompute_r_vr_from_pos_vel(X8[:, 0:3], X8[:, 3:6])
        particle_ids = arr[:, 8].astype(np.int64) if arr.shape[1] >= 9 else np.arange(len(X8), dtype=np.int64)
        mass = arr[:, 9].astype(np.float32) if arr.shape[1] >= 10 else np.ones(len(X8), dtype=np.float32)
        return X8, mass, particle_ids
    raise ValueError(f"Unsupported test format: {path}")


# ----------------------------- feature engineering ---------------------------
ALL_FEATURE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "r", "vr", "speed", "vt", "Lx", "Ly", "Lz", "Lmag"]
_COL_INDEX = {n: i for i, n in enumerate(ALL_FEATURE_NAMES)}
_COORD_NAMES = frozenset({"x", "y", "z"})
_VEL_NAMES = frozenset({"vx", "vy", "vz", "vr", "speed", "vt"})
_POS_NAMES = frozenset({"r"})
_L_NAMES = frozenset({"Lx", "Ly", "Lz", "Lmag"})
FEATURE_GROUPS = {
    "all": ["vx", "vy", "vz", "r", "vr", "speed", "vt", "Lx", "Ly", "Lz", "Lmag"],
    "spatial_only": ["r"],
    "phase_space_cartesian": ["vx", "vy", "vz"],
    "phase_space_extended": ["vx", "vy", "vz", "r", "vr", "vt"],
    "angular_momentum": ["vx", "vy", "vz", "r", "vr", "Lx", "Ly", "Lz", "Lmag"],
}


def resolve_feature_set(feature_set=None):
    feature_set = C.FEATURE_SET if feature_set is None else feature_set
    if isinstance(feature_set, str):
        if feature_set in FEATURE_GROUPS:
            names = list(FEATURE_GROUPS[feature_set])
        elif "," in feature_set:
            names = [x.strip() for x in feature_set.split(",")]
        else:
            raise ValueError(f"Unknown FEATURE_SET {feature_set!r}; choose {list(FEATURE_GROUPS)} or comma list")
    else:
        names = list(feature_set)
    feat_names = [n for n in names if n not in _COORD_NAMES]
    if not feat_names:
        raise ValueError("FEATURE_SET must include at least one non-coordinate feature")
    unknown = [n for n in feat_names if n not in _COL_INDEX]
    if unknown:
        raise ValueError(f"Unknown feature names {unknown}; valid {ALL_FEATURE_NAMES}")
    return feat_names


def get_context_feature_names(feature_set=None):
    if not bool(getattr(C, "ENABLE_VOXEL_CONTEXT_FEATURES", False)):
        return []
    requested = tuple(getattr(C, "VOXEL_CONTEXT_FEATURES", ()))
    if not requested:
        return []
    base = set(resolve_feature_set(feature_set))
    requirements = {
        "voxel_log_count": (),
        "voxel_sigma_v": ("vx", "vy", "vz"),
        "voxel_sigma_vr": ("vr",),
        "voxel_sigma_speed": ("speed",),
        "voxel_L_align": ("Lx", "Ly", "Lz"),
        "voxel_sigma_Lmag": ("Lmag",),
    }
    unknown = [name for name in requested if name not in requirements]
    if unknown:
        raise ValueError(f"Unknown VOXEL_CONTEXT_FEATURES {unknown}; valid {list(requirements)}")
    return [name for name in requested if all(req in base for req in requirements[name])]


def get_model_feature_names(feature_set=None):
    return resolve_feature_set(feature_set) + get_context_feature_names(feature_set)


def get_feature_channels(feature_set=None):
    return len(get_model_feature_names(feature_set))


def recompute_phase_columns(X):
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return X
    pos = X[:, 0:3]
    vel = X[:, 3:6]
    r = np.linalg.norm(pos, axis=1, keepdims=True).astype(np.float32)
    vr = (np.sum(pos * vel, axis=1, keepdims=True) / np.where(r == 0, 1e-8, r)).astype(np.float32)
    X[:, 6:8] = np.hstack([r, vr])
    if X.shape[1] >= 14:
        speed = np.linalg.norm(vel, axis=1, keepdims=True).astype(np.float32)
        vt = np.sqrt(np.maximum(speed * speed - vr * vr, 0.0)).astype(np.float32)
        L = np.cross(pos, vel).astype(np.float32)
        Lmag = np.linalg.norm(L, axis=1, keepdims=True).astype(np.float32)
        X[:, 8:14] = np.hstack([speed, vt, L, Lmag]).astype(np.float32)
    return X


def enrich_features(X8):
    X8 = np.asarray(X8, dtype=np.float32)
    pos, vel, vr = X8[:, 0:3], X8[:, 3:6], X8[:, 7:8]
    speed = np.linalg.norm(vel, axis=1, keepdims=True).astype(np.float32)
    vt = np.sqrt(np.maximum(speed * speed - vr * vr, 0.0)).astype(np.float32)
    L = np.cross(pos, vel).astype(np.float32)
    Lmag = np.linalg.norm(L, axis=1, keepdims=True).astype(np.float32)
    return np.hstack([X8, speed, vt, L, Lmag]).astype(np.float32)


def _robust_scale(values, quantile, floor):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(floor)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(floor)
    scale = float(np.quantile(finite, float(quantile)))
    return max(scale, float(floor))


def _phase_scales(X14):
    pos_mag = np.linalg.norm(X14[:, 0:3], axis=1)
    vel_mag = np.linalg.norm(X14[:, 3:6], axis=1)
    pos_scale = _robust_scale(pos_mag, C.POSITION_SCALE_QUANTILE, C.MIN_POSITION_SCALE)
    vel_scale = _robust_scale(vel_mag, C.VELOCITY_SCALE_QUANTILE, C.MIN_VELOCITY_SCALE)
    return pos_scale, vel_scale


def _dimensionless_feature_matrix(X14, feat_names, pos_scale, vel_scale):
    cols = []
    l_scale = max(pos_scale * vel_scale, C.MIN_POSITION_SCALE * C.MIN_VELOCITY_SCALE)
    for name in feat_names:
        raw = X14[:, _COL_INDEX[name]].astype(np.float32)
        if name in _POS_NAMES:
            raw = raw / pos_scale
        elif name in _VEL_NAMES:
            raw = raw / vel_scale
        elif name in _L_NAMES:
            raw = raw / l_scale
        cols.append(raw)
    return np.stack(cols, axis=1).astype(np.float32)


def compute_feature_stats(stream_arr, bg_arr, sample_cap=1_500_000, feature_set=None):
    feat_names = resolve_feature_set(feature_set)
    if C.DIMENSIONLESS_INPUTS and not C.STANDARDIZE_DIMENSIONLESS_FEATURES:
        n = len(feat_names)
        return {"feat_mean": np.zeros(n, dtype=np.float32), "feat_std": np.ones(n, dtype=np.float32)}
    cols = [_COL_INDEX[n] for n in feat_names]
    rng = np.random.default_rng(C.RANDOM_SEED)
    parts = []
    for arr in (stream_arr, bg_arr):
        n = min(len(arr), sample_cap // 2)
        parts.append(arr[rng.choice(len(arr), size=n, replace=False)])
    raw = np.vstack(parts).astype(np.float32)[:, cols]
    return {"feat_mean": raw.mean(axis=0).astype(np.float32), "feat_std": raw.std(axis=0).astype(np.float32) + 1e-6}


def make_litept_inputs(X14, centre, stats, feature_set=None, grid_size=None):
    """Return xyz diagnostics, model features, and native 6D phase coordinates."""
    feat_names = resolve_feature_set(feature_set)
    X14 = np.asarray(X14, dtype=np.float32)
    centre = np.asarray(centre, dtype=np.float32)
    phase_vel_weight = float(getattr(C, "PHASE_VEL_WEIGHT", 0.05))
    if C.DIMENSIONLESS_INPUTS:
        pos_scale, vel_scale = _phase_scales(X14)
        coord_xyz = (X14[:, 0:3] - centre[None, :]) / np.float32(pos_scale)
        vel_norm = X14[:, 3:6] / np.float32(vel_scale)
        phase_vel = vel_norm * np.float32(phase_vel_weight * vel_scale / pos_scale)
        coord6 = np.hstack([coord_xyz, phase_vel]).astype(np.float32)
        feat = _dimensionless_feature_matrix(X14, feat_names, pos_scale, vel_scale)
        if C.STANDARDIZE_DIMENSIONLESS_FEATURES:
            feat = (feat - stats["feat_mean"][None, :]) / stats["feat_std"][None, :]
        grid_size_eff = max(float(grid_size) / pos_scale, float(C.MIN_NORMALIZED_GRID_SIZE)) if grid_size is not None else None
    else:
        cols = [_COL_INDEX[n] for n in feat_names]
        coord_xyz = X14[:, 0:3] - centre[None, :]
        coord6 = np.hstack([coord_xyz, X14[:, 3:6] * np.float32(phase_vel_weight)]).astype(np.float32)
        feat = (X14[:, cols] - stats["feat_mean"][None, :]) / stats["feat_std"][None, :]
        grid_size_eff = grid_size
    coord_xyz = coord_xyz.astype(np.float32)
    feat = feat.astype(np.float32)
    coord6 = coord6.astype(np.float32)
    if coord6.shape[1] != int(getattr(C, "NATIVE_COORD_DIM", 6)):
        raise ValueError(f"native coord width must be 6, got {coord6.shape}")
    if grid_size is None:
        return coord_xyz, feat, coord6
    return coord_xyz, feat, coord6, float(grid_size_eff)

def random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array([
        [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
        [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
        [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
    ], dtype=np.float32)


def _rotate_about_origin(X, R):
    if len(X) == 0:
        return X
    X[:, 0:3] = X[:, 0:3] @ R.T
    X[:, 3:6] = X[:, 3:6] @ R.T
    return recompute_phase_columns(X)


def _rotate_stream_about_centre(X, centre, R):
    if len(X) == 0:
        return X
    X[:, 0:3] = (X[:, 0:3] - centre[None, :]) @ R.T + centre[None, :]
    X[:, 3:6] = X[:, 3:6] @ R.T
    return recompute_phase_columns(X)


def _scale_units(X, scale):
    if len(X) == 0:
        return X
    X[:, 0:6] *= np.float32(scale)
    return recompute_phase_columns(X)


def apply_physical_augmentations(stream, background, centre, rng, enable_relative_stream_rotation=False):
    Xs = np.asarray(stream, dtype=np.float32).copy()
    Xb = np.asarray(background, dtype=np.float32).copy()
    centre_aug = np.asarray(centre, dtype=np.float32).copy()
    meta = {"unit_scale": 1.0, "common_rotation": 0, "relative_stream_rotation": 0}

    # Only xyz/vxyz participate in the geometric transforms below. Derived
    # phase columns are recomputed once at the end instead of after each step.
    if C.SYNTHESIS_AUGMENT and enable_relative_stream_rotation and len(Xs) > 0 and rng.random() < C.RELATIVE_ROTATION_PROB:
        R_rel = random_rotation_matrix(rng)
        Xs[:, 0:3] = (Xs[:, 0:3] - centre_aug[None, :]) @ R_rel.T + centre_aug[None, :]
        Xs[:, 3:6] = Xs[:, 3:6] @ R_rel.T
        meta["relative_stream_rotation"] = 1

    if C.AUGMENT_ROTATION and rng.random() < C.ROTATION_PROB:
        R = random_rotation_matrix(rng)
        if len(Xs) > 0:
            Xs[:, 0:3] = Xs[:, 0:3] @ R.T
            Xs[:, 3:6] = Xs[:, 3:6] @ R.T
        if len(Xb) > 0:
            Xb[:, 0:3] = Xb[:, 0:3] @ R.T
            Xb[:, 3:6] = Xb[:, 3:6] @ R.T
        centre_aug = centre_aug @ R.T
        meta["common_rotation"] = 1

    if C.AUGMENT_UNIT_SCALE:
        log2_scale = rng.uniform(float(C.UNIT_SCALE_LOG2_MIN), float(C.UNIT_SCALE_LOG2_MAX))
        scale = float(2.0 ** log2_scale)
        if len(Xs) > 0:
            Xs[:, 0:6] *= np.float32(scale)
        if len(Xb) > 0:
            Xb[:, 0:6] *= np.float32(scale)
        centre_aug = centre_aug * np.float32(scale)
        meta["unit_scale"] = scale

    Xs = recompute_phase_columns(Xs)
    Xb = recompute_phase_columns(Xb)
    return Xs, Xb, centre_aug.astype(np.float32), meta


def geometry_report(name, arr, sample_cap=None, rng=None):
    rng = np.random.default_rng(C.RANDOM_SEED + 404) if rng is None else rng
    sample_cap = int(C.GEOMETRY_SAMPLE_CAP if sample_cap is None else sample_cap)
    arr = np.asarray(arr, dtype=np.float32)
    n = min(len(arr), sample_cap)
    if n <= 3:
        return {"name": name, "n": int(len(arr)), "sampled": int(n), "axis_ratio": float("nan"), "evals": []}
    idx = rng.choice(len(arr), size=n, replace=False) if n < len(arr) else np.arange(len(arr))
    pos = np.asarray(arr[idx, 0:3], dtype=np.float64)
    pos = pos - pos.mean(axis=0, keepdims=True)
    cov = np.cov(pos, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals, 1e-12)
    axis_ratio = float(np.sqrt(evals.max() / evals.min()))
    return {"name": name, "n": int(len(arr)), "sampled": int(n), "axis_ratio": axis_ratio, "evals": evals.tolist()}


def build_split_candidate_arrays(
    split, max_stream, max_back, stream_sampling_mode=None, background_sampling_mode=None,
    full_background_small_files=None, background_large_file_count=None, large_background_file_names=None,
):
    rng = np.random.default_rng(C.RANDOM_SEED + (0 if split in ("train", "training") else 1000 if split in ("val", "validation") else 2000))
    stream_sampling_mode = C.STREAM_SAMPLING_MODE if stream_sampling_mode is None else stream_sampling_mode
    background_sampling_mode = C.BACKGROUND_SAMPLING_MODE if background_sampling_mode is None else background_sampling_mode
    full_background_small_files = C.FULL_BACKGROUND_SMALL_FILES if full_background_small_files is None else bool(full_background_small_files)
    background_large_file_count = C.BACKGROUND_LARGE_FILE_COUNT if background_large_file_count is None else int(background_large_file_count)
    large_background_file_names = C.LARGE_BACKGROUND_FILE_NAMES if large_background_file_names is None else str(large_background_file_names)
    stream_files = list_stream_files(split)
    back_files = list_background_files(split)
    if not stream_files:
        raise RuntimeError(f"No {split} stream files found")
    if not back_files:
        raise RuntimeError(f"No {split} background files found")
    stream_sizes = [get_file_nrows(p) for p in stream_files]
    back_sizes = [get_file_nrows(p) for p in back_files]
    stream_counts = proportional_counts(max_stream, stream_sizes, stream_sampling_mode, min_per_nonzero=32)
    back_counts = background_counts_full_small_files(max_back, back_files, back_sizes, background_sampling_mode, full_background_small_files, background_large_file_count, large_background_file_names)
    stream_parts = [sample_rows_by_count(load_stream_file(p), c, rng) for p, c in zip(stream_files, stream_counts)]
    back_parts = [sample_rows_by_count(load_background_file(p), c, rng) for p, c in zip(back_files, back_counts) if int(c) > 0]
    X_stream = enrich_features(np.vstack(stream_parts))
    X_back = enrich_features(np.vstack(back_parts))
    print(f"{split} candidate pools | stream={len(X_stream):,} background={len(X_back):,}")
    return X_stream, X_back


def build_train_val_candidate_arrays(
    max_stream_train=C.MAX_STREAM_TRAIN_POINTS,
    max_back_train=C.MAX_BACK_TRAIN_POINTS,
    max_stream_val=C.MAX_STREAM_VAL_POINTS,
    max_back_val=C.MAX_BACK_VAL_POINTS,
    stream_sampling_mode=None,
    background_sampling_mode=None,
    full_background_small_files=None,
    background_large_file_count=None,
    large_background_file_names=None,
    **_ignored,
):
    Xs_tr, Xb_tr = build_split_candidate_arrays(
        "train", max_stream_train, max_back_train, stream_sampling_mode, background_sampling_mode,
        full_background_small_files, background_large_file_count, large_background_file_names,
    )
    Xs_va, Xb_va = build_split_candidate_arrays(
        "val", max_stream_val, max_back_val, stream_sampling_mode, background_sampling_mode,
        full_background_small_files, background_large_file_count, large_background_file_names,
    )
    return Xs_tr, Xb_tr, Xs_va, Xb_va
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
