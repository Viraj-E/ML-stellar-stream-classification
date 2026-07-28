#!/usr/bin/env python3
"""Hybrid point-cloud classifier for stellar-stream segmentation.

The model combines:

1) raw point-patch features (centre-relative phase-space features)
2) explicit patch-summary features (density/dispersion/anistropy/alignment, etc.)
3) attention pooling over points

Outputs
-------
Always saves all particles in original order plus:
  *_prob_stream.npy
  *_logit_stream.npy
  *_margin.npy
  *_feature_norm.npy
  *_pred_label.npy
  *_latent256.npy            (optional; on by default)
  *_patch_embed768.npy       (optional)
  *_hidden512.npy            (optional)
  *_patch_summary12.npy      (always)

Checkpoint files:
  pointcloud_hybrid_best.pt
  pointcloud_hybrid_meta.json
"""

import os
import time
import json
import argparse
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report
import faiss

# ============================================================
# DEFAULT SETTINGS
# ============================================================

TV_ROOT = os.path.join("data", "training_validation")
TRAIN_STREAM_DIR = os.path.join(TV_ROOT, "training", "streams")
VAL_STREAM_DIR   = os.path.join(TV_ROOT, "validation", "streams")
TRAIN_BACK_DIR   = os.path.join(TV_ROOT, "training", "background")
VAL_BACK_DIR     = os.path.join(TV_ROOT, "validation", "background")
STREAM_DIR = TRAIN_STREAM_DIR
BACK_DIR   = TRAIN_BACK_DIR
TEST_DIR   = os.path.join("data", "test")
OUT_DIR    = os.path.join("outputs", "pointcloud")

TEST_POS_SCALE = 1
TEST_PART_TYPE = "PartType4"
BACKGROUND_PART_TYPE = "PartType1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ------------------------------------------------------------
# Data / compute settings
# ------------------------------------------------------------
USE_MEMMAP = True
USE_AMP = False
USE_DATA_PARALLEL = True
PIN_MEMORY = False

PATCH_K = 48
INFER_CHUNK_SIZE = 32768
FAISS_QUERY_BATCH = 131072

LOCAL_BOX_SIZE = 80.0
BOX_EXPAND_FACTORS = (1.0, 2.0, 4.0)

TRAIN_STREAM_REF_CAP = 1_800_000
TRAIN_BACK_REF_CAP   = 3_500_000
VAL_STREAM_REF_CAP   = 300_000
VAL_BACK_REF_CAP     = 600_000

TRAIN_STREAM_CENTRES = 120_000
TRAIN_BACK_CENTRES   = 120_000
VAL_STREAM_CENTRES   = 15_000
VAL_BACK_CENTRES     = 15_000

VAL_STREAM_FILE_FRACTION = 0.15
VAL_BACKGROUND_PARTICLE_FRACTION = 0.20

STREAM_SAMPLING_MODE = "equal_files"    # equal_files / sqrt_size / proportional
BACKGROUND_SAMPLING_MODE = "proportional"
NEGATIVE_OVERLAY_STREAM_PROB = 0.25

PATCH_SPACE = "xyzv"                    # xyz / xyzv
PATCH_VEL_WEIGHT = 0.05

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------
BATCH_SIZE = 1536
EPOCHS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.02
DROPOUT = 0.12

# ------------------------------------------------------------
# Loss / augmentation settings
# ------------------------------------------------------------
# CrossEntropy class weight for class 0 = background.
# Values > 1.0 penalise background->stream false positives more strongly.
BACKGROUND_LOSS_WEIGHT = 2.0

# Train-time patch augmentations only. These are deliberately mild and are
# applied after feature construction, so validation/inference remain unchanged.
USE_PATCH_AUGMENTATION = True
AUGMENT_POS_JITTER_STD = 0.01
AUGMENT_VEL_JITTER_STD = 0.01
AUGMENT_POINT_DROPOUT_PROB = 0.05
AUGMENT_FEATURE_DROPOUT_PROB = 0.02

# ------------------------------------------------------------
# CPU worker settings
# ------------------------------------------------------------
AVAILABLE_CPUS = 24
NUM_WORKERS_DEFAULT = min(12, max(4, AVAILABLE_CPUS - 8))

# ------------------------------------------------------------
# Output settings
# ------------------------------------------------------------
DEFAULT_THRESHOLD = 0.50
CKPT_NAME = "pointcloud_hybrid_best.pt"
META_NAME = "pointcloud_hybrid_meta.json"
POINTCLOUD_LOG = "pointcloud_training_log.csv"

LOG_NAME = "pointcloud_training_log.dat"
LIVE_LOG_NAME = "pointcloud_live_batch_log.csv"
THRESHOLD_LOG_NAME = "pointcloud_validation_threshold_metrics.csv"
VALIDATION_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)

# ============================================================
# REPRODUCIBILITY / PERFORMANCE
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.benchmark = True
N_GPUS = torch.cuda.device_count()

try:
    torch.set_num_threads(AVAILABLE_CPUS)
    torch.set_num_interop_threads(4)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", str(AVAILABLE_CPUS))
os.environ.setdefault("MKL_NUM_THREADS", str(AVAILABLE_CPUS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(AVAILABLE_CPUS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(AVAILABLE_CPUS))

# ============================================================
# HELPERS
# ============================================================

class CSVLogger(object):
    """
    Crash-safe CSV logger.
    Writes the header once, appends one row at a time, and fsyncs every write so
    progress already completed is preserved if the job later fails.
    """
    def __init__(self, filename):
        self.filename = filename
        self.header = None
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def _stringify(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def write(self, row):
        row = {k: self._stringify(v) for k, v in row.items()}
        if self.header is None:
            self.header = list(row.keys())
        elif list(row.keys()) != self.header:
            raise ValueError("CSVLogger received inconsistent keys")

        file_exists = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0
        with open(self.filename, "a", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=self.header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


class BinaryMetricAccumulator(object):
    """
    Version-independent binary metric accumulator.
    Avoids torchmetrics API differences across cluster environments.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
        y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)

        self.tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.tn += int(np.sum((y_true == 0) & (y_pred == 0)))
        self.fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        self.n += int(y_true.size)

    def compute(self):
        tp = int(self.tp)
        tn = int(self.tn)
        fp = int(self.fp)
        fn = int(self.fn)
        n = max(int(self.n), 1)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        acc = (tp + tn) / n
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        dice = f1

        prod = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        denom = float(np.sqrt(np.float64(max(prod, 1.0))))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        return {
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "iou": float(iou),
            "dice": float(dice),
            "mcc": float(mcc),
            "balanced_acc": float(0.5 * (recall + specificity)),
            "pred_pos_frac": float((tp + fp) / n),
            "true_pos_frac": float((tp + fn) / n),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }


def compute_threshold_metrics(y_true, prob_stream, thresholds=VALIDATION_THRESHOLDS):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    prob_stream = np.asarray(prob_stream, dtype=np.float32).reshape(-1)
    rows = []
    for thr in thresholds:
        acc = BinaryMetricAccumulator()
        y_pred = (prob_stream >= float(thr)).astype(np.int64)
        acc.update(y_true, y_pred)
        out = acc.compute()
        out["threshold"] = float(thr)
        rows.append(out)
    return rows


@contextmanager
def dummy_context():
    yield


def autocast_context():
    if USE_AMP and DEVICE == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast()
    return dummy_context()


def make_grad_scaler():
    if USE_AMP and DEVICE == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ascontig_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def base_stem(path_or_fname):
    return os.path.splitext(os.path.basename(path_or_fname))[0]


def resolve_test_dataset_path(path_or_name):
    if os.path.isabs(path_or_name) or os.path.exists(path_or_name):
        return path_or_name
    candidate = os.path.join(TEST_DIR, path_or_name)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Test dataset not found: {path_or_name} (also checked {candidate})")


def load_npy(path):
    if USE_MEMMAP:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def format_int(n):
    return "{:,}".format(int(n))


def now():
    return time.perf_counter()


def fmt(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    if sec < 3600:
        return f"{sec/60:.2f}m"
    return f"{sec/3600:.2f}h"


def maybe_wrap_dataparallel(model):
    if USE_DATA_PARALLEL and DEVICE == "cuda" and N_GPUS > 1:
        print(f"Wrapping model in DataParallel across {N_GPUS} GPUs")
        return nn.DataParallel(model)
    return model


def get_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def recompute_r_vr_from_pos_vel(pos, vel):
    r = np.linalg.norm(pos, axis=1)
    r_safe = np.where(r == 0, 1e-8, r)
    vr = np.sum(pos * vel, axis=1) / r_safe
    return np.column_stack([r, vr]).astype(np.float32)


def split_file_lists(files, val_fraction=0.15, seed=RANDOM_SEED):
    files = list(files)
    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    files = [files[i] for i in order]

    if len(files) <= 1:
        return files, files

    n_val = max(1, int(round(len(files) * val_fraction)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    if len(train_files) == 0:
        train_files = val_files
    return train_files, val_files


def build_dataloader(dataset, batch_size, shuffle, drop_last, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        #pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
        pin_memory=False,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)

# ============================================================
# FILE DISCOVERY / LOADING
# ============================================================

def list_stream_files(split="train"):
    """Return whole stream files for the requested split. No stream file is shared across train/val."""
    d = TRAIN_STREAM_DIR if split in ("train", "training") else VAL_STREAM_DIR
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith(".npy")
    )


def list_background_files(split="train"):
    """Return background files for the requested split. Background particles are sampled within these files."""
    d = TRAIN_BACK_DIR if split in ("train", "training") else VAL_BACK_DIR
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith((".npy", ".hdf5", ".h5"))
    )


def configure_training_validation_root(tv_root):
    """
    Update the prepared train/validation directory used by training.

    This lets the same code switch between, for example:
      data/training_validation

    Inference is unaffected.
    """
    global TV_ROOT
    global TRAIN_STREAM_DIR, VAL_STREAM_DIR, TRAIN_BACK_DIR, VAL_BACK_DIR
    global STREAM_DIR, BACK_DIR

    TV_ROOT = os.path.abspath(tv_root)
    TRAIN_STREAM_DIR = os.path.join(TV_ROOT, "training", "streams")
    VAL_STREAM_DIR   = os.path.join(TV_ROOT, "validation", "streams")
    TRAIN_BACK_DIR   = os.path.join(TV_ROOT, "training", "background")
    VAL_BACK_DIR     = os.path.join(TV_ROOT, "validation", "background")

    # Backwards-compatible aliases
    STREAM_DIR = TRAIN_STREAM_DIR
    BACK_DIR   = TRAIN_BACK_DIR


def parse_large_background_names(names):
    """
    Parse an optional comma-separated list of large synthetic background basenames.

    Examples:
      --large_background_file_names t_background00001.npy,t_background00002.npy
      --large_background_file_names background_1,background_2

    Matching is done against both basename-with-extension and basename-without-extension.
    """
    if names is None:
        return set()
    names = str(names).strip()
    if not names:
        return set()
    return set(x.strip() for x in names.split(",") if x.strip())


def background_counts_full_small_files(cap, paths, sizes, mode, full_small_files=False,
                                       large_file_count=2, large_file_names=""):
    """
    Background sampling helper.

    Default behaviour is unchanged:
      - counts are allocated across all background files according to `mode` and `cap`.

    Optional behaviour for the user's synthetic-background setup:
      - fully include every non-large background file
      - only subsample the large synthetic files
      - by default, the large files are identified as the largest N files in that split
      - optionally, provide exact/partial large file basenames via --large_background_file_names

    Important: when full_small_files=True, `cap` applies to the large files only.
    The total loaded background pool may therefore be:
      all particles from small/coherent files + up to `cap` particles from large synthetic files.
    """
    sizes_arr = np.asarray(sizes, dtype=np.int64)

    if not full_small_files:
        return proportional_counts(cap, sizes_arr, mode, min_per_nonzero=1)

    n_files = len(paths)
    if n_files == 0:
        return np.zeros(0, dtype=np.int64)

    explicit_names = parse_large_background_names(large_file_names)
    large_mask = np.zeros(n_files, dtype=bool)

    if explicit_names:
        for i, path in enumerate(paths):
            b = os.path.basename(path)
            s = base_stem(path)
            if b in explicit_names or s in explicit_names:
                large_mask[i] = True

        if not np.any(large_mask):
            raise ValueError(
                "full_small_files=True but none of --large_background_file_names matched "
                "the available background files."
            )
    else:
        n_large = min(int(large_file_count), n_files)
        if n_large > 0:
            large_idx = np.argsort(-sizes_arr)[:n_large]
            large_mask[large_idx] = True

    counts = sizes_arr.copy()

    large_idx = np.where(large_mask)[0]
    if len(large_idx) > 0:
        large_counts = proportional_counts(
            cap,
            sizes_arr[large_idx],
            mode,
            min_per_nonzero=1,
        )
        counts[large_idx] = large_counts

    return counts.astype(np.int64)


def get_file_nrows(path, part_type=BACKGROUND_PART_TYPE):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = load_npy(path)
        return int(arr.shape[0])
    if ext in [".hdf5", ".h5"]:
        with h5py.File(path, "r") as f:
            return int(f[part_type]["Coordinates"].shape[0])
    raise ValueError(f"Unsupported file format: {path}")


def load_stream_file(path):
    arr = ascontig_f32(load_npy(path))
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"Stream file must have shape (N, >=8): {path}")
    return arr[:, :8]


def load_background_npy(path):
    arr = ascontig_f32(load_npy(path))
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"Background npy must have shape (N, >=8): {path}")
    return arr[:, :8]


def load_background_hdf5(path, part_type=BACKGROUND_PART_TYPE):
    with h5py.File(path, "r") as f:
        pos = np.array(f[part_type]["Coordinates"][:], dtype=np.float32, order="C")
        vel = np.array(f[part_type]["Velocities"][:], dtype=np.float32, order="C")
    r_vr = recompute_r_vr_from_pos_vel(pos, vel)
    return np.hstack([pos, vel, r_vr]).astype(np.float32)


def load_background_file(path, part_type=BACKGROUND_PART_TYPE):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_background_npy(path)
    if ext in [".hdf5", ".h5"]:
        return load_background_hdf5(path, part_type=part_type)
    raise ValueError(f"Unsupported background file format: {path}")


def load_test_snapshot(path, part_type=TEST_PART_TYPE, pos_scale=TEST_POS_SCALE):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".hdf5", ".h5"]:
        with h5py.File(path, "r") as f:
            pos = np.array(f[part_type]["Coordinates"][:], dtype=np.float32, order="C")
            _, idx = np.unique(pos, axis=0, return_index=True)
            idx = np.sort(idx)
            pos = pos[idx] * np.float32(pos_scale)
            centre = pos.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
            pos -= centre
            vel = np.array(f[part_type]["Velocities"][:], dtype=np.float32, order="C")[idx]
            if "Mass" in f[part_type]:
                mass = np.array(f[part_type]["Mass"][:], dtype=np.float32, order="C")[idx]
            else:
                mass = np.ones(len(pos), dtype=np.float32)
            if "ParticleIDs" in f[part_type]:
                ids = np.array(f[part_type]["ParticleIDs"][:], dtype=np.int64, order="C")[idx]
            else:
                ids = np.arange(len(pos), dtype=np.int64)
        r_vr = recompute_r_vr_from_pos_vel(pos, vel)
        X = np.hstack([pos, vel, r_vr]).astype(np.float32)
        return X, mass, ids

    if ext == ".npy":
        arr = ascontig_f32(load_npy(path))
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError(f"Test npy must have shape (N, >=8): {path}")
        X = arr[:, :8].astype(np.float32)
        centre = X[:, 0:3].mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
        X[:, 0:3] -= centre
        mass = np.ones(len(X), dtype=np.float32)
        ids = np.arange(len(X), dtype=np.int64)
        return X, mass, ids
    raise ValueError(f"Unsupported test dataset format: {path}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def enrich_features(X8):
    """
    Input columns:
      [x,y,z,vx,vy,vz,r,vr]
    Output columns:
      [x,y,z,vx,vy,vz,r,vr,speed,vt,Lx,Ly,Lz,Lmag]
    """
    X8 = np.asarray(X8, dtype=np.float32)
    pos = X8[:, 0:3]
    vel = X8[:, 3:6]
    vr = X8[:, 7:8]

    speed = np.linalg.norm(vel, axis=1, keepdims=True).astype(np.float32)
    vt2 = np.maximum(speed * speed - vr * vr, 0.0)
    vt = np.sqrt(vt2, dtype=np.float32)
    L = np.cross(pos, vel).astype(np.float32)
    Lmag = np.linalg.norm(L, axis=1, keepdims=True).astype(np.float32)

    return np.hstack([X8, speed, vt, L, Lmag]).astype(np.float32)


def proportional_counts(cap, sizes, mode, min_per_nonzero=1):
    sizes = np.asarray(sizes, dtype=np.float64)
    if len(sizes) == 0:
        return np.zeros(0, dtype=np.int64)

    if mode == "equal_files":
        weights = np.ones_like(sizes)
    elif mode == "sqrt_size":
        weights = np.sqrt(np.maximum(sizes, 1.0))
    elif mode == "proportional":
        weights = np.maximum(sizes, 1.0)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    weights /= weights.sum()
    raw = cap * weights
    counts = np.floor(raw).astype(np.int64)
    counts = np.minimum(counts, sizes.astype(np.int64))

    leftover = int(cap - counts.sum())
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    for j in order:
        if leftover <= 0:
            break
        if counts[j] < sizes[j]:
            counts[j] += 1
            leftover -= 1

    if min_per_nonzero > 0:
        eligible = np.where(sizes >= min_per_nonzero)[0]
        for j in eligible:
            if counts[j] == 0 and cap > 0:
                donor = np.argmax(counts)
                if counts[donor] > min_per_nonzero:
                    counts[donor] -= 1
                    counts[j] += 1

    return counts.astype(np.int64)


def sample_rows_by_count(arr, count, rng):
    n = len(arr)
    if count >= n:
        return np.asarray(arr, dtype=np.float32)
    idx = rng.choice(n, size=int(count), replace=False)
    return np.asarray(arr[idx], dtype=np.float32)


def sample_rows_and_split(arr, train_count, val_count, rng):
    total = int(train_count + val_count)
    n = len(arr)
    if total >= n:
        idx = np.arange(n)
        rng.shuffle(idx)
        train_idx = idx[:min(train_count, n)]
        remain = idx[min(train_count, n):min(train_count + val_count, n)]
        train = np.asarray(arr[train_idx], dtype=np.float32)
        val = np.asarray(arr[remain], dtype=np.float32)
        return train, val

    idx = rng.choice(n, size=total, replace=False)
    rng.shuffle(idx)
    train = np.asarray(arr[idx[:train_count]], dtype=np.float32)
    val = np.asarray(arr[idx[train_count:train_count + val_count]], dtype=np.float32)
    return train, val


def compute_feature_stats(stream_arrays, bg_array):
    parts = []
    for arr in stream_arrays:
        if len(arr) > 0:
            parts.append(arr)
    if len(bg_array) > 0:
        parts.append(bg_array)
    X = np.vstack(parts).astype(np.float32)

    stats = {}
    stats["pos_scale"] = (np.std(X[:, 0:3], axis=0).astype(np.float32) + 1e-6)
    stats["vel_mean"] = np.mean(X[:, 3:6], axis=0).astype(np.float32)
    stats["vel_std"] = (np.std(X[:, 3:6], axis=0).astype(np.float32) + 1e-6)
    stats["extra_mean"] = np.mean(X[:, 6:10], axis=0).astype(np.float32)
    stats["extra_std"] = (np.std(X[:, 6:10], axis=0).astype(np.float32) + 1e-6)
    stats["L_mean"] = np.mean(X[:, 10:13], axis=0).astype(np.float32)
    stats["L_std"] = (np.std(X[:, 10:13], axis=0).astype(np.float32) + 1e-6)
    stats["Lmag_mean"] = np.mean(X[:, 13:14], axis=0).astype(np.float32)
    stats["Lmag_std"] = (np.std(X[:, 13:14], axis=0).astype(np.float32) + 1e-6)
    return stats

# ============================================================
# SPATIAL INDEX
# ============================================================

class SpatialCellIndex(object):
    def __init__(self, X, cell_size):
        self.X = np.asarray(X, dtype=np.float32)
        self.cell_size = float(cell_size)
        self.pos = self.X[:, 0:3]
        self.min_corner = self.pos.min(axis=0)
        cell_ids = np.floor((self.pos - self.min_corner) / self.cell_size).astype(np.int32)

        cell_map = {}
        for i in range(len(cell_ids)):
            key = (int(cell_ids[i, 0]), int(cell_ids[i, 1]), int(cell_ids[i, 2]))
            if key not in cell_map:
                cell_map[key] = []
            cell_map[key].append(i)

        self.cell_map = {k: np.asarray(v, dtype=np.int64) for k, v in cell_map.items()}

    def query_box(self, centre, box_size):
        half = 0.5 * float(box_size)
        lo = centre - half
        hi = centre + half

        lo_id = np.floor((lo - self.min_corner) / self.cell_size).astype(np.int32)
        hi_id = np.floor((hi - self.min_corner) / self.cell_size).astype(np.int32)

        candidates = []
        for ix in range(lo_id[0], hi_id[0] + 1):
            for iy in range(lo_id[1], hi_id[1] + 1):
                for iz in range(lo_id[2], hi_id[2] + 1):
                    key = (int(ix), int(iy), int(iz))
                    if key in self.cell_map:
                        candidates.append(self.cell_map[key])

        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64)

        idx = np.concatenate(candidates)
        pos = self.pos[idx]
        mask = np.all((pos >= lo) & (pos < hi), axis=1)
        return idx[mask]

# ============================================================
# FAISS KNN
# ============================================================

class FastSelfKNN(object):
    def __init__(self, X_pos):
        self.X_pos = np.ascontiguousarray(X_pos.astype(np.float32))
        d = self.X_pos.shape[1]
        cpu_index = faiss.IndexFlatL2(d)
        if torch.cuda.is_available():
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            self.index = cpu_index
        self.index.add(self.X_pos)

    def query(self, Q, k, batch=FAISS_QUERY_BATCH):
        Q = np.ascontiguousarray(Q.astype(np.float32))
        n = len(Q)
        out = np.empty((n, k), dtype=np.int64)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            _, inds = self.index.search(Q[s:e], k)
            out[s:e] = inds
        return out

# ============================================================
# TRAINING DATA PREP
# ============================================================

@dataclass
class TrainPools:
    train_stream_arrays: list
    val_stream_arrays: list
    train_bg: np.ndarray
    val_bg: np.ndarray
    stream_train_files: list
    stream_val_files: list
    train_stats: dict


def build_training_pools(args):
    """
    Build explicit train/validation pools from the prepared training_validation tree.

    Streams are split at file level upstream and are never re-split here. The optional
    particle caps only control how many stream particles are sampled into the per-epoch
    candidate pool for compute/memory reasons; validation stream files remain entirely
    separate from training stream files.

    Backgrounds are treated as large mixed reservoirs and are sampled within the files
    already assigned to each split.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    train_stream_files = list_stream_files("train")
    val_stream_files = list_stream_files("val")
    train_back_files = list_background_files("train")
    val_back_files = list_background_files("val")

    if len(train_stream_files) == 0:
        raise RuntimeError(f"No training stream files found in {TRAIN_STREAM_DIR}")
    if len(val_stream_files) == 0:
        raise RuntimeError(f"No validation stream files found in {VAL_STREAM_DIR}")
    if len(train_back_files) == 0:
        raise RuntimeError(f"No training background files found in {TRAIN_BACK_DIR}")
    if len(val_back_files) == 0:
        raise RuntimeError(f"No validation background files found in {VAL_BACK_DIR}")

    print("\nDiscovering split file sizes...")
    train_stream_sizes = [get_file_nrows(p) for p in train_stream_files]
    val_stream_sizes   = [get_file_nrows(p) for p in val_stream_files]
    train_back_sizes   = [get_file_nrows(p) for p in train_back_files]
    val_back_sizes     = [get_file_nrows(p) for p in val_back_files]

    print(f"Train stream files:      {len(train_stream_files)}")
    print(f"Validation stream files: {len(val_stream_files)}")
    print(f"Train background files:  {len(train_back_files)}")
    print(f"Validation bg files:     {len(val_back_files)}")
    print(f"Training/validation root: {TV_ROOT}")
    if args.full_background_small_files:
        print("Background loading mode: full small/coherent files + capped large synthetic files")
        print(f"  large synthetic files per split: {args.background_large_file_count}")
        if args.large_background_file_names:
            print(f"  explicit large background names: {args.large_background_file_names}")
    else:
        print("Background loading mode: capped sampling across all background files")

    train_stream_counts = proportional_counts(
        args.train_stream_ref_cap, train_stream_sizes, args.stream_sampling_mode, min_per_nonzero=32
    )
    val_stream_counts = proportional_counts(
        args.val_stream_ref_cap, val_stream_sizes, args.stream_sampling_mode, min_per_nonzero=32
    )
    train_bg_counts = background_counts_full_small_files(
        args.train_back_ref_cap,
        train_back_files,
        train_back_sizes,
        args.background_sampling_mode,
        full_small_files=args.full_background_small_files,
        large_file_count=args.background_large_file_count,
        large_file_names=args.large_background_file_names,
    )
    val_bg_counts = background_counts_full_small_files(
        args.val_back_ref_cap,
        val_back_files,
        val_back_sizes,
        args.background_sampling_mode,
        full_small_files=args.full_background_small_files,
        large_file_count=args.background_large_file_count,
        large_file_names=args.large_background_file_names,
    )

    if args.full_background_small_files:
        def _print_bg_plan(label, files, sizes, counts):
            print(f"\n{label} background sampling plan:")
            for path, size, cnt in sorted(zip(files, sizes, counts), key=lambda x: -int(x[1])):
                mode_txt = "FULL" if int(cnt) >= int(size) else "SAMPLED"
                print(f"  {base_stem(path)} | size={format_int(size)} | load={format_int(cnt)} | {mode_txt}")
        _print_bg_plan("Training", train_back_files, train_back_sizes, train_bg_counts)
        _print_bg_plan("Validation", val_back_files, val_back_sizes, val_bg_counts)

    print("\nLoading sampled training stream pools...")
    train_stream_arrays = []
    for i, (path, cnt) in enumerate(zip(train_stream_files, train_stream_counts)):
        arr = enrich_features(load_stream_file(path))
        part = sample_rows_by_count(arr, int(cnt), rng)
        train_stream_arrays.append(part)
        print(f"  train stream {i+1}/{len(train_stream_files)} | {base_stem(path)} | sampled {format_int(len(part))}")

    print("\nLoading sampled validation stream pools...")
    val_stream_arrays = []
    for i, (path, cnt) in enumerate(zip(val_stream_files, val_stream_counts)):
        arr = enrich_features(load_stream_file(path))
        part = sample_rows_by_count(arr, int(cnt), rng)
        val_stream_arrays.append(part)
        print(f"  val stream   {i+1}/{len(val_stream_files)} | {base_stem(path)} | sampled {format_int(len(part))}")

    print("\nLoading sampled training background pool...")
    train_bg_parts = []
    for i, (path, cnt) in enumerate(zip(train_back_files, train_bg_counts)):
        if cnt <= 0:
            continue
        arr = enrich_features(load_background_file(path))
        part = sample_rows_by_count(arr, int(cnt), rng)
        train_bg_parts.append(part)
        print(f"  train bg {i+1}/{len(train_back_files)} | {base_stem(path)} | sampled {format_int(len(part))}")

    print("\nLoading sampled validation background pool...")
    val_bg_parts = []
    for i, (path, cnt) in enumerate(zip(val_back_files, val_bg_counts)):
        if cnt <= 0:
            continue
        arr = enrich_features(load_background_file(path))
        part = sample_rows_by_count(arr, int(cnt), rng)
        val_bg_parts.append(part)
        print(f"  val bg   {i+1}/{len(val_back_files)} | {base_stem(path)} | sampled {format_int(len(part))}")

    train_bg = np.vstack(train_bg_parts).astype(np.float32)
    val_bg = np.vstack(val_bg_parts).astype(np.float32)

    print("\nFinal sampled pools:")
    print(f"  train stream total: {format_int(sum(len(a) for a in train_stream_arrays))}")
    print(f"  val stream total:   {format_int(sum(len(a) for a in val_stream_arrays))}")
    print(f"  train background:   {format_int(len(train_bg))}")
    print(f"  val background:     {format_int(len(val_bg))}")

    train_stats = compute_feature_stats(train_stream_arrays, train_bg)

    return TrainPools(
        train_stream_arrays=train_stream_arrays,
        val_stream_arrays=val_stream_arrays,
        train_bg=train_bg,
        val_bg=val_bg,
        stream_train_files=train_stream_files,
        stream_val_files=val_stream_files,
        train_stats=train_stats,
    )

# ============================================================
# PATCH BUILDING
# ============================================================

def patch_space_coords(X):
    if PATCH_SPACE == "xyz":
        return X[:, 0:3]
    if PATCH_SPACE == "xyzv":
        pos = X[:, 0:3]
        vel = X[:, 3:6] * np.float32(PATCH_VEL_WEIGHT)
        return np.hstack([pos, vel]).astype(np.float32)
    raise ValueError(f"Unknown patch space: {PATCH_SPACE}")


def rank_candidates_knn(candidates, centre_row, k_total):
    if len(candidates) == 0:
        return np.empty((0, candidates.shape[1]), dtype=np.float32)
    C = patch_space_coords(candidates)
    c = patch_space_coords(centre_row[None, :])[0]
    d2 = np.sum((C - c[None, :]) ** 2, axis=1)
    take = min(k_total, len(candidates))
    idx = np.argpartition(d2, take - 1)[:take]
    idx = idx[np.argsort(d2[idx])]
    return candidates[idx]


def pad_patch_rows(rows, k_total):
    if len(rows) == 0:
        raise RuntimeError("Cannot pad an empty patch")
    if len(rows) >= k_total:
        return rows[:k_total]
    reps = np.resize(np.arange(len(rows)), k_total - len(rows))
    return np.vstack([rows, rows[reps]]).astype(np.float32)


def build_patch_summary(rows, centre_row):
    rel_pos = rows[:, 0:3] - centre_row[0:3][None, :]
    rel_vel = rows[:, 3:6] - centre_row[3:6][None, :]
    dist = np.linalg.norm(rel_pos, axis=1).astype(np.float32)
    speed = np.linalg.norm(rows[:, 3:6], axis=1).astype(np.float32)
    L = rows[:, 10:13]
    Lmag = np.linalg.norm(L, axis=1) + 1e-6
    Lhat = L / Lmag[:, None]
    align = float(np.linalg.norm(np.mean(Lhat, axis=0)))

    # position anisotropy
    Xc = rel_pos - rel_pos.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(rows), 1)
    evals = np.linalg.eigvalsh(cov).astype(np.float32)
    evals = np.sort(np.maximum(evals, 1e-8))
    anis_ratio = float(evals[-1] / evals[0])
    flatness = float(evals[1] / evals[-1])

    sigma_v = float(np.std(rows[:, 3:6]))
    sigma_vr = float(np.std(rows[:, 7]))
    mean_dist = float(np.mean(dist))
    kth_dist = float(dist[min(len(dist)-1, max(0, len(dist)//2))])
    max_dist = float(np.max(dist))
    centre_speed = max(float(centre_row[8]), 1e-6)
    mean_speed_ratio = float(np.mean(speed / centre_speed))
    mean_abs_dvr = float(np.mean(np.abs(rows[:, 7] - centre_row[7])))
    mean_vt = float(np.mean(rows[:, 9]))
    mean_lmag = float(np.mean(rows[:, 13]))
    centre_lmag = max(float(centre_row[13]), 1e-6)
    lmag_ratio = mean_lmag / centre_lmag

    out = np.array([
        mean_dist,
        kth_dist,
        max_dist,
        sigma_v,
        sigma_vr,
        anis_ratio,
        flatness,
        align,
        mean_speed_ratio,
        mean_abs_dvr,
        mean_vt,
        lmag_ratio,
    ], dtype=np.float32)
    return out


def build_point_features(patch_rows, centre_row, stats):
    """
    16 per-point features:
      rel_pos(3), rel_vel(3), abs_extra_norm(4), rel_L(3), abs_Lmag_norm(1), dist(1), speed_ratio(1)
    """
    rel_pos = (patch_rows[:, 0:3] - centre_row[0:3][None, :]) / stats["pos_scale"][None, :]
    rel_vel = (patch_rows[:, 3:6] - centre_row[3:6][None, :]) / stats["vel_std"][None, :]
    abs_extra = (patch_rows[:, 6:10] - stats["extra_mean"][None, :]) / stats["extra_std"][None, :]
    rel_L = (patch_rows[:, 10:13] - centre_row[10:13][None, :]) / stats["L_std"][None, :]
    abs_Lmag = (patch_rows[:, 13:14] - stats["Lmag_mean"][None, :]) / stats["Lmag_std"][None, :]

    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True).astype(np.float32)
    centre_speed = max(float(centre_row[8]), 1e-6)
    speed_ratio = (patch_rows[:, 8:9] / centre_speed).astype(np.float32)

    feats = np.hstack([rel_pos, rel_vel, abs_extra, rel_L, abs_Lmag, dist, speed_ratio]).astype(np.float32)
    return feats


def augment_point_features(point_feats, rng, pos_jitter_std, vel_jitter_std, point_dropout_prob, feature_dropout_prob):
    """
    Mild train-time patch augmentation in the already-normalised feature space.

    Feature layout from build_point_features:
      0:3   rel_pos
      3:6   rel_vel
      6:10  abs_extra_norm
      10:13 rel_L
      13:14 abs_Lmag_norm
      14:15 dist
      15:16 speed_ratio

    These augmentations are only applied to training samples. Validation and
    inference remain untouched.
    """
    feats = np.asarray(point_feats, dtype=np.float32).copy()
    k = len(feats)

    # Random point ordering. Pooling should mostly be order-invariant, but this
    # prevents the attention branch from learning accidental neighbour-rank quirks.
    if k > 1:
        perm = rng.permutation(k)
        feats = feats[perm]

    # Small Gaussian jitter in normalised relative position and velocity features.
    if pos_jitter_std > 0:
        feats[:, 0:3] += rng.normal(0.0, pos_jitter_std, size=feats[:, 0:3].shape).astype(np.float32)
    if vel_jitter_std > 0:
        feats[:, 3:6] += rng.normal(0.0, vel_jitter_std, size=feats[:, 3:6].shape).astype(np.float32)

    # Randomly replace a few neighbour feature rows with the patch mean. This is
    # gentler than zeroing rows and makes the model less dependent on one or two
    # highly informative neighbours.
    if point_dropout_prob > 0 and k > 1:
        drop = rng.random(k) < point_dropout_prob
        if np.any(drop) and np.any(~drop):
            feats[drop] = feats[~drop].mean(axis=0, keepdims=True)

    # Mild feature dropout across all point rows. Keep distance/speed ratio intact.
    if feature_dropout_prob > 0:
        cols = np.arange(0, 14)
        drop_cols = rng.random(len(cols)) < feature_dropout_prob
        if np.any(drop_cols):
            feats[:, cols[drop_cols]] = 0.0

    return feats.astype(np.float32, copy=False)


def build_mixed_patch(stream_arr, stream_index, bg_arr, bg_index, centre_row, k_total, base_box_size,
                      include_stream=True, include_bg=True):
    centre_pos = centre_row[0:3]
    ranked = None

    for fac in BOX_EXPAND_FACTORS:
        box = base_box_size * fac
        parts = []
        if include_stream and stream_arr is not None and len(stream_arr) > 0:
            idx_s = stream_index.query_box(centre_pos, box)
            if len(idx_s) > 0:
                parts.append(stream_arr[idx_s])
        if include_bg and bg_arr is not None and len(bg_arr) > 0:
            idx_b = bg_index.query_box(centre_pos, box)
            if len(idx_b) > 0:
                parts.append(bg_arr[idx_b])
        if len(parts) == 0:
            continue
        cand = np.vstack([centre_row[None, :], *parts]).astype(np.float32)
        ranked = rank_candidates_knn(cand, centre_row, k_total + 8)

        cspace = patch_space_coords(ranked)
        c0 = patch_space_coords(centre_row[None, :])[0]
        d2 = np.sum((cspace - c0[None, :]) ** 2, axis=1)
        zero = np.where(d2 == 0)[0]
        if len(zero) > 1:
            keep = np.ones(len(ranked), dtype=bool)
            keep[zero[1:]] = False
            ranked = ranked[keep]
        if len(ranked) >= k_total:
            break

    if ranked is None or len(ranked) == 0:
        ranked = centre_row[None, :].astype(np.float32)
    rows = pad_patch_rows(ranked, k_total)
    return rows

# ============================================================
# DATASET
# ============================================================

class MixedPointPatchDataset(Dataset):
    def __init__(self, stream_arrays, bg_array, stats, n_stream_centres, n_bg_centres,
                 patch_k, base_box_size, negative_overlay_prob, seed,
                 augment=False, augment_pos_jitter_std=0.0, augment_vel_jitter_std=0.0,
                 augment_point_dropout_prob=0.0, augment_feature_dropout_prob=0.0):
        self.stream_arrays = [np.asarray(a, dtype=np.float32) for a in stream_arrays if len(a) > 0]
        self.bg_array = np.asarray(bg_array, dtype=np.float32)
        self.stats = stats
        self.n_stream_centres = int(n_stream_centres)
        self.n_bg_centres = int(n_bg_centres)
        self.patch_k = int(patch_k)
        self.base_box_size = float(base_box_size)
        self.negative_overlay_prob = float(negative_overlay_prob)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.augment_pos_jitter_std = float(augment_pos_jitter_std)
        self.augment_vel_jitter_std = float(augment_vel_jitter_std)
        self.augment_point_dropout_prob = float(augment_point_dropout_prob)
        self.augment_feature_dropout_prob = float(augment_feature_dropout_prob)
        self.total = self.n_stream_centres + self.n_bg_centres

        if len(self.stream_arrays) == 0:
            raise RuntimeError("No stream arrays available")
        if len(self.bg_array) == 0:
            raise RuntimeError("No background array available")

        t0 = now()
        self.stream_indices = [SpatialCellIndex(a, self.base_box_size) for a in self.stream_arrays]
        self.bg_index = SpatialCellIndex(self.bg_array, self.base_box_size)
        print(f"Built spatial indices in {fmt(now() - t0)}")
        self._build_specs()

    def _balanced_stream_file_choices(self, rng, n):
        n_files = len(self.stream_arrays)
        base = n // n_files
        rem = n % n_files
        file_ids = []
        for i in range(n_files):
            count = base + (1 if i < rem else 0)
            file_ids.extend([i] * count)
        file_ids = np.asarray(file_ids, dtype=np.int64)
        if len(file_ids) == 0:
            return file_ids
        rng.shuffle(file_ids)
        return file_ids

    def _build_specs(self):
        rng = np.random.default_rng(self.seed)
        specs = []
        file_ids = self._balanced_stream_file_choices(rng, self.n_stream_centres)
        for fidx in file_ids:
            arr = self.stream_arrays[int(fidx)]
            cidx = int(rng.integers(0, len(arr)))
            specs.append((1, int(fidx), cidx))

        for _ in range(self.n_bg_centres):
            cidx = int(rng.integers(0, len(self.bg_array)))
            overlay = -1
            if rng.random() < self.negative_overlay_prob:
                overlay = int(rng.integers(0, len(self.stream_arrays)))
            specs.append((0, overlay, cidx))
        self.specs = specs

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        label, sid, cidx = self.specs[idx]
        if label == 1:
            stream_arr = self.stream_arrays[sid]
            stream_idx = self.stream_indices[sid]
            centre = stream_arr[cidx]
            rows = build_mixed_patch(stream_arr, stream_idx, self.bg_array, self.bg_index,
                                     centre, self.patch_k, self.base_box_size,
                                     include_stream=True, include_bg=True)
        else:
            centre = self.bg_array[cidx]
            if sid >= 0:
                stream_arr = self.stream_arrays[sid]
                stream_idx = self.stream_indices[sid]
            else:
                stream_arr = None
                stream_idx = None
            rows = build_mixed_patch(stream_arr, stream_idx, self.bg_array, self.bg_index,
                                     centre, self.patch_k, self.base_box_size,
                                     include_stream=(sid >= 0), include_bg=True)

        point_feats = build_point_features(rows, centre, self.stats)
        if self.augment:
            rng = np.random.default_rng(self.seed + 1000003 * int(idx))
            point_feats = augment_point_features(
                point_feats,
                rng,
                self.augment_pos_jitter_std,
                self.augment_vel_jitter_std,
                self.augment_point_dropout_prob,
                self.augment_feature_dropout_prob,
            )
        patch_summary = build_patch_summary(rows, centre)
        return (
            torch.from_numpy(point_feats),
            torch.from_numpy(patch_summary),
            torch.tensor(label, dtype=torch.long),
        )

# ============================================================
# MODEL
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PatchSummaryMLP(nn.Module):
    def __init__(self, in_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PointCloudHybridNet(nn.Module):
    def __init__(self, point_dim=16, summary_dim=12, dropout=DROPOUT):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBNReLU(point_dim, 64),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 256),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1),
        )
        self.summary_mlp = PatchSummaryMLP(summary_dim)
        self.fc1 = nn.Linear(256 * 3 + 64, 512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, point_x, summary_x, return_features=False):
        point_x = point_x.transpose(1, 2).contiguous()  # [B, C, K]
        h = self.feat(point_x)                          # [B, 256, K]
        h_max = torch.amax(h, dim=2)
        h_mean = torch.mean(h, dim=2)
        a = torch.softmax(self.attn(h), dim=2)
        h_attn = torch.sum(h * a, dim=2)
        patch_embed768 = torch.cat([h_max, h_mean, h_attn], dim=1)

        summary64 = self.summary_mlp(summary_x)
        fusion = torch.cat([patch_embed768, summary64], dim=1)

        hidden512 = F.relu(self.fc1(fusion), inplace=False)
        hidden512 = self.drop1(hidden512)
        latent256 = F.relu(self.fc2(hidden512), inplace=False)
        latent256 = self.drop2(latent256)
        logits = self.fc3(latent256)

        if not return_features:
            return logits
        return {
            "logits": logits,
            "patch_embed768": patch_embed768,
            "summary64": summary64,
            "hidden512": hidden512,
            "latent256": latent256,
        }

# ============================================================
# TRAIN / EVAL
# ============================================================


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    train_mode=True,
    epoch=0,
    live_logger=None,
    log_every=25,
):

    if train_mode:
        model.train()
        phase = "train"
    else:
        model.eval()
        phase = "val"

    all_true = []
    all_pred = []
    all_prob = []
    metric_acc = BinaryMetricAccumulator()
    loss_sum = 0.0
    n_seen = 0

    for batch_i, (point_x, summary_x, yb) in enumerate(loader, start=1):
        point_x = point_x.to(DEVICE, non_blocking=True)
        summary_x = summary_x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            # With USE_AMP=False, autocast_context() is a dummy context.
            # This keeps the same code path but avoids float16 instability.
            with autocast_context():
                logits = model(point_x, summary_x)
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                loss_value = float("nan")
                try:
                    loss_value = float(loss.detach().cpu().item())
                except Exception:
                    pass
                if live_logger is not None:
                    live_logger.write({
                        "epoch": int(epoch),
                        "phase": phase,
                        "batch": int(batch_i),
                        "total_batches": int(len(loader)),
                        "status": "nonfinite_loss",
                        "loss": loss_value,
                        "running_loss": float(loss_sum / max(n_seen, 1)),
                        "batch_true_pos_frac": float("nan"),
                        "batch_pred_pos_frac": float("nan"),
                        "batch_prob_mean": float("nan"),
                        "batch_prob_min": float("nan"),
                        "batch_prob_max": float("nan"),
                        "grad_norm": float("nan"),
                        "n_seen": int(n_seen),
                    })
                raise FloatingPointError(
                    f"Non-finite loss detected | phase={phase} epoch={epoch} "
                    f"batch={batch_i} loss={loss_value}"
                )

            grad_norm = 0.0

            if train_mode:
                # AMP is intentionally disabled by default for stability.
                # Therefore scaler should usually be None, but keep this robust.
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        if not torch.all(torch.isfinite(g)):
                            if live_logger is not None:
                                live_logger.write({
                                    "epoch": int(epoch),
                                    "phase": phase,
                                    "batch": int(batch_i),
                                    "total_batches": int(len(loader)),
                                    "status": "nonfinite_gradient",
                                    "loss": float(loss.detach().cpu().item()),
                                    "running_loss": float(loss_sum / max(n_seen, 1)),
                                    "batch_true_pos_frac": float("nan"),
                                    "batch_pred_pos_frac": float("nan"),
                                    "batch_prob_mean": float("nan"),
                                    "batch_prob_min": float("nan"),
                                    "batch_prob_max": float("nan"),
                                    "grad_norm": float("nan"),
                                    "n_seen": int(n_seen),
                                })
                            raise FloatingPointError(
                                f"Non-finite gradient detected | epoch={epoch} batch={batch_i}"
                            )
                        grad_sq += float(torch.sum(g.float() ** 2).detach().cpu())

                grad_norm = grad_sq ** 0.5

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        prob = torch.softmax(logits.detach(), dim=1)[:, 1]
        pred = torch.argmax(logits.detach(), dim=1)

        bsz = int(point_x.size(0))
        loss_val = float(loss.detach().cpu().item())
        loss_sum += loss_val * bsz
        n_seen += bsz

        y_true_np = yb.detach().cpu().numpy()
        y_pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()

        metric_acc.update(y_true_np, y_pred_np)
        all_true.append(y_true_np)
        all_pred.append(y_pred_np)
        all_prob.append(prob_np)

        if live_logger is not None and (
            batch_i == 1 or batch_i % max(int(log_every), 1) == 0 or batch_i == len(loader)
        ):
            live_logger.write({
                "epoch": int(epoch),
                "phase": phase,
                "batch": int(batch_i),
                "total_batches": int(len(loader)),
                "status": "ok",
                "loss": float(loss_val),
                "running_loss": float(loss_sum / max(n_seen, 1)),
                "batch_true_pos_frac": float(np.mean(y_true_np == 1)),
                "batch_pred_pos_frac": float(np.mean(y_pred_np == 1)),
                "batch_prob_mean": float(np.mean(prob_np)),
                "batch_prob_min": float(np.min(prob_np)),
                "batch_prob_max": float(np.max(prob_np)),
                "grad_norm": float(grad_norm),
                "n_seen": int(n_seen),
            })

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    out = metric_acc.compute()
    out["loss"] = loss_sum / max(n_seen, 1)
    return out, y_true, y_pred, y_prob

# ============================================================
# INFERENCE HELPERS
# ============================================================

def build_infer_patch_tensors(X_full, centres_rows, stats, inds):
    if inds.ndim != 2:
        raise ValueError(f"inds must be 2D, got shape {inds.shape}")
    if np.any(inds < 0) or np.any(inds >= len(X_full)):
        bad = inds[(inds < 0) | (inds >= len(X_full))][:10]
        raise IndexError(f"Neighbour indices out of bounds for X_full with size {len(X_full)}; sample bad indices={bad}")

    neigh = X_full[inds]
    centres = np.ascontiguousarray(centres_rows, dtype=np.float32)
    if len(centres) != len(neigh):
        raise ValueError(f"centre/neighbour batch mismatch: {len(centres)} vs {len(neigh)}")

    rel_pos = (neigh[:, :, 0:3] - centres[:, None, 0:3]) / stats["pos_scale"][None, None, :]
    rel_vel = (neigh[:, :, 3:6] - centres[:, None, 3:6]) / stats["vel_std"][None, None, :]
    abs_extra = (neigh[:, :, 6:10] - stats["extra_mean"][None, None, :]) / stats["extra_std"][None, None, :]
    rel_L = (neigh[:, :, 10:13] - centres[:, None, 10:13]) / stats["L_std"][None, None, :]
    abs_Lmag = (neigh[:, :, 13:14] - stats["Lmag_mean"][None, None, :]) / stats["Lmag_std"][None, None, :]
    dist = np.linalg.norm(rel_pos, axis=2, keepdims=True).astype(np.float32)
    centre_speed = np.maximum(centres[:, None, 8:9], 1e-6)
    speed_ratio = neigh[:, :, 8:9] / centre_speed
    point_feats = np.concatenate([rel_pos, rel_vel, abs_extra, rel_L, abs_Lmag, dist, speed_ratio], axis=2).astype(np.float32)
    if point_feats.shape[2] != 16:
        raise RuntimeError(f"Expected 16 point features, got {point_feats.shape[2]}")

    # patch summaries batch
    B = len(centres)
    summary = np.empty((B, 12), dtype=np.float32)
    for i in range(B):
        summary[i] = build_patch_summary(neigh[i], centres[i])

    return np.ascontiguousarray(point_feats), np.ascontiguousarray(summary)


# ============================================================
# INFERENCE
# ============================================================

def infer_full_dataset(model, X_test8, stats, threshold, infer_chunk_size, patch_k,
                       save_latent256=True, save_patch_embed768=False,
                       save_hidden512=False, aux_feature_dtype="float16"):
    model.eval()
    X_test = enrich_features(X_test8)
    knn = FastSelfKNN(X_test[:, 0:3])

    n = len(X_test)
    probs = np.zeros(n, dtype=np.float32)
    logit_stream = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    feature_norm = np.zeros(n, dtype=np.float32)
    patch_summary12 = np.zeros((n, 12), dtype=np.float32)

    aux_dtype = np.float16 if str(aux_feature_dtype).lower() == "float16" else np.float32
    latent256 = np.zeros((n, 256), dtype=aux_dtype) if save_latent256 else None
    patch_embed768 = np.zeros((n, 768), dtype=aux_dtype) if save_patch_embed768 else None
    hidden512 = np.zeros((n, 512), dtype=aux_dtype) if save_hidden512 else None

    t0 = now()
    print("Running self-KNN inference on test snapshot...")

    for s in range(0, n, infer_chunk_size):
        e = min(s + infer_chunk_size, n)
        centres_rows = X_test[s:e]
        centres_xyz = centres_rows[:, 0:3]
        inds = np.asarray(knn.query(centres_xyz, k=patch_k + 4, batch=FAISS_QUERY_BATCH), dtype=np.int64)

        if inds.ndim != 2:
            raise RuntimeError(f"KNN query returned invalid shape {inds.shape}")
        if inds.shape[1] < 1:
            raise RuntimeError("KNN query returned zero neighbours")

        q_idx = np.arange(s, e, dtype=np.int64)
        fixed_inds = np.empty((e - s, patch_k), dtype=np.int64)

        for i in range(e - s):
            row = inds[i]
            row = row[row != q_idx[i]]
            if row.size == 0:
                row_full = np.asarray(knn.query(centres_xyz[i:i+1], k=patch_k + 8, batch=FAISS_QUERY_BATCH), dtype=np.int64)[0]
                row = row_full[row_full != q_idx[i]]
            if row.size == 0:
                raise RuntimeError(f"No valid non-self neighbours found for inference row {s + i}")
            if row.size < patch_k:
                row = np.resize(row, patch_k)
            else:
                row = row[:patch_k]
            fixed_inds[i] = row

        point_feats, patch_summary = build_infer_patch_tensors(X_test, centres_rows, stats, fixed_inds)
        point_x = torch.from_numpy(point_feats).to(DEVICE, non_blocking=True)
        summary_x = torch.from_numpy(patch_summary).to(DEVICE, non_blocking=True)

        with torch.no_grad():
            with autocast_context():
                out = model(point_x, summary_x, return_features=True)
                logits = out["logits"]
                p = torch.softmax(logits, dim=1)[:, 1]

        logits_np = logits.detach().cpu().numpy().astype(np.float32)
        probs[s:e] = p.detach().cpu().numpy().astype(np.float32)
        logit_stream[s:e] = logits_np[:, 1]
        margin[s:e] = logits_np[:, 1] - logits_np[:, 0]
        patch_summary12[s:e] = patch_summary

        lat_np = out["latent256"].detach().cpu().numpy().astype(np.float32)
        feature_norm[s:e] = np.linalg.norm(lat_np, axis=1).astype(np.float32)

        if save_latent256:
            latent256[s:e] = lat_np.astype(aux_dtype, copy=False)
        if save_patch_embed768:
            patch_embed768[s:e] = out["patch_embed768"].detach().cpu().numpy().astype(aux_dtype, copy=False)
        if save_hidden512:
            hidden512[s:e] = out["hidden512"].detach().cpu().numpy().astype(aux_dtype, copy=False)

        print(f"  processed {format_int(e)}/{format_int(n)}")

    pred = (probs >= float(threshold)).astype(np.int8)
    print(f"Inference finished in {fmt(now() - t0)} | stream fraction={pred.mean():.4f}")
    return {
        "probs": probs,
        "pred": pred,
        "logit_stream": logit_stream,
        "margin": margin,
        "feature_norm": feature_norm,
        "patch_summary12": patch_summary12,
        "latent256": latent256,
        "patch_embed768": patch_embed768,
        "hidden512": hidden512,
    }

# ============================================================
# SAVE / LOAD META
# ============================================================

def stats_to_jsonable(stats):
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


def stats_from_jsonable(obj):
    return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}

# ============================================================
# MAIN TRAIN / INFER
# ============================================================

def train_main(args):
    ensure_dir(args.out_dir)

    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPUs:", N_GPUS)
    print("Using device:", DEVICE)

    pools = build_training_pools(args)
    stats = pools.train_stats

    model = PointCloudHybridNet(point_dim=16, summary_dim=12).to(DEVICE)
    model = maybe_wrap_dataparallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = torch.tensor([float(args.background_loss_weight), 1.0], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    print(f"CrossEntropy class weights: background={class_weights[0].item():.3f}, stream={class_weights[1].item():.3f}")
    scaler = make_grad_scaler()

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, CKPT_NAME)
    meta_path = os.path.join(args.out_dir, META_NAME)
    log_path = os.path.join(args.out_dir, LOG_NAME)
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = CSVLogger(log_path)

    live_log_path = os.path.join(args.out_dir, LIVE_LOG_NAME)
    if os.path.exists(live_log_path):
        os.remove(live_log_path)
    live_logger = CSVLogger(live_log_path)

    threshold_log_path = os.path.join(args.out_dir, THRESHOLD_LOG_NAME)
    if os.path.exists(threshold_log_path):
        os.remove(threshold_log_path)
    threshold_logger = CSVLogger(threshold_log_path)

    for epoch in range(args.epochs):
        print("\n================================================")
        print(f"Epoch {epoch + 1:03d}/{args.epochs}")
        print("================================================")
        t0 = now()

        train_ds = MixedPointPatchDataset(
            pools.train_stream_arrays, pools.train_bg, stats,
            n_stream_centres=args.train_stream_centres,
            n_bg_centres=args.train_back_centres,
            patch_k=args.patch_k,
            base_box_size=args.local_box_size,
            negative_overlay_prob=args.negative_overlay_stream_prob,
            seed=RANDOM_SEED + epoch,
            augment=args.use_patch_augmentation,
            augment_pos_jitter_std=args.augment_pos_jitter_std,
            augment_vel_jitter_std=args.augment_vel_jitter_std,
            augment_point_dropout_prob=args.augment_point_dropout_prob,
            augment_feature_dropout_prob=args.augment_feature_dropout_prob,
        )
        val_ds = MixedPointPatchDataset(
            pools.val_stream_arrays, pools.val_bg, stats,
            n_stream_centres=args.val_stream_centres,
            n_bg_centres=args.val_back_centres,
            patch_k=args.patch_k,
            base_box_size=args.local_box_size,
            negative_overlay_prob=args.negative_overlay_stream_prob,
            seed=RANDOM_SEED + 1000 + epoch,
            augment=False,
        )

        train_dl = build_dataloader(train_ds, args.batch_size, True, True, args.num_workers)
        val_dl   = build_dataloader(val_ds, args.batch_size, False, False, args.num_workers)

        tr, _, _, _ = run_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            scaler,
            train_mode=True,
            epoch=epoch + 1,
            live_logger=live_logger,
            log_every=args.live_log_every,
        )
        va, y_true, y_pred, y_prob = run_epoch(
            model,
            val_dl,
            optimizer,
            criterion,
            scaler,
            train_mode=False,
            epoch=epoch + 1,
            live_logger=live_logger,
            log_every=args.live_log_every,
        )

        print(
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} prec={tr['precision']:.4f} "
            f"rec={tr['recall']:.4f} f1={tr['f1']:.4f} iou={tr['iou']:.4f} "
            f"dice={tr['dice']:.4f} mcc={tr['mcc']:.4f}"
        )
        print(
            f"val   loss={va['loss']:.4f} acc={va['acc']:.4f} prec={va['precision']:.4f} "
            f"rec={va['recall']:.4f} f1={va['f1']:.4f} iou={va['iou']:.4f} "
            f"dice={va['dice']:.4f} mcc={va['mcc']:.4f}"
        )

        val_threshold_rows = compute_threshold_metrics(y_true, y_prob, VALIDATION_THRESHOLDS)
        print("validation threshold metrics:")
        print("  thr    acc    prec   rec    f1     iou    mcc    pred+frac")
        for row in val_threshold_rows:
            print(
                f"  {row['threshold']:.3f}  {row['acc']:.4f} {row['precision']:.4f} "
                f"{row['recall']:.4f} {row['f1']:.4f} {row['iou']:.4f} "
                f"{row['mcc']:.4f} {row['pred_pos_frac']:.4f}"
            )
            threshold_logger.write({
                "epoch": epoch + 1,
                "threshold": float(row["threshold"]),
                "val_acc": float(row["acc"]),
                "val_precision": float(row["precision"]),
                "val_recall": float(row["recall"]),
                "val_specificity": float(row["specificity"]),
                "val_f1": float(row["f1"]),
                "val_iou": float(row["iou"]),
                "val_dice": float(row["dice"]),
                "val_mcc": float(row["mcc"]),
                "val_balanced_acc": float(row["balanced_acc"]),
                "val_pred_pos_frac": float(row["pred_pos_frac"]),
                "val_true_pos_frac": float(row["true_pos_frac"]),
                "tp": int(row["tp"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            })

        epoch_time = now() - t0
        logger.write({
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "train_precision": float(tr["precision"]),
            "train_recall": float(tr["recall"]),
            "train_specificity": float(tr["specificity"]),
            "train_f1": float(tr["f1"]),
            "train_iou": float(tr["iou"]),
            "train_dice": float(tr["dice"]),
            "train_mcc": float(tr["mcc"]),
            "train_balanced_acc": float(tr["balanced_acc"]),
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "val_precision": float(va["precision"]),
            "val_recall": float(va["recall"]),
            "val_specificity": float(va["specificity"]),
            "val_f1": float(va["f1"]),
            "val_iou": float(va["iou"]),
            "val_dice": float(va["dice"]),
            "val_mcc": float(va["mcc"]),
            "val_balanced_acc": float(va["balanced_acc"]),
            "epoch_seconds": float(epoch_time),
        })
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print(f"epoch time: {fmt(epoch_time)}")

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            ckpt = {
                "model_state": get_model_state_dict(model),
                "stats": stats_to_jsonable(stats),
                "args": vars(args),
                "best_val_f1": float(best_f1),
                "patch_k": int(args.patch_k),
                "summary_dim": 12,
                "point_dim": 16,
            }
            torch.save(ckpt, best_path)
            with open(meta_path, "w") as f:
                json.dump({
                    "stats": stats_to_jsonable(stats),
                    "best_val_f1": float(best_f1),
                    "train_stream_files": [base_stem(p) for p in pools.stream_train_files],
                    "val_stream_files": [base_stem(p) for p in pools.stream_val_files],
                    "patch_k": int(args.patch_k),
                    "local_box_size": float(args.local_box_size),
                    "patch_space": args.patch_space,
                    "patch_vel_weight": float(args.patch_vel_weight),
                    "negative_overlay_stream_prob": float(args.negative_overlay_stream_prob),
                    "background_loss_weight": float(args.background_loss_weight),
                    "use_patch_augmentation": bool(args.use_patch_augmentation),
                    "augment_pos_jitter_std": float(args.augment_pos_jitter_std),
                    "augment_vel_jitter_std": float(args.augment_vel_jitter_std),
                    "augment_point_dropout_prob": float(args.augment_point_dropout_prob),
                    "augment_feature_dropout_prob": float(args.augment_feature_dropout_prob),
                    "summary_features": [
                        "mean_dist", "kth_dist", "max_dist", "sigma_v", "sigma_vr",
                        "anis_ratio", "flatness", "L_align", "mean_speed_ratio",
                        "mean_abs_dvr", "mean_vt", "lmag_ratio"
                    ],
                    "notes": "Hybrid point-cloud patch classifier: point features + explicit patch summaries.",
                    "training_log": LOG_NAME,
                    "threshold_log": THRESHOLD_LOG_NAME,
                    "validation_thresholds": list(VALIDATION_THRESHOLDS),
                }, f, indent=2)
            print(f"Saved new best model to {best_path}")

    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
    print(f"Outputs written to: {args.out_dir}")


def infer_main(args):
    ensure_dir(args.out_dir)
    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(args.out_dir, CKPT_NAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stats = stats_from_jsonable(ckpt["stats"])
    patch_k = int(ckpt.get("patch_k", PATCH_K))

    model = PointCloudHybridNet(point_dim=16, summary_dim=12).to(DEVICE)
    load_model_state_dict(model, ckpt["model_state"])
    model.eval()

    test_dataset_path = resolve_test_dataset_path(args.test_dataset)
    X_test8, mass, ids = load_test_snapshot(test_dataset_path)
    prefix = base_stem(test_dataset_path)

    infer_out = infer_full_dataset(
        model=model,
        X_test8=X_test8,
        stats=stats,
        threshold=args.threshold,
        infer_chunk_size=args.infer_chunk_size,
        patch_k=patch_k,
        save_latent256=True,
        save_patch_embed768=True,
        save_hidden512=True,
        aux_feature_dtype=args.aux_feature_dtype,
    )

    pos = X_test8[:, 0:3].astype(np.float32)
    vel = X_test8[:, 3:6].astype(np.float32)
    r_vr = X_test8[:, 6:8].astype(np.float32)


    np.save(os.path.join(args.out_dir, f"{prefix}_pos.npy"), pos)
    np.save(os.path.join(args.out_dir, f"{prefix}_vel.npy"), vel)
    np.save(os.path.join(args.out_dir, f"{prefix}_r_vr.npy"), r_vr)
    np.save(os.path.join(args.out_dir, f"{prefix}_mass.npy"), mass.astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_ids.npy"), ids.astype(np.int64))
    np.save(os.path.join(args.out_dir, f"{prefix}_prob_stream.npy"), infer_out["probs"].astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_logit_stream.npy"), infer_out["logit_stream"].astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_margin.npy"), infer_out["margin"].astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_feature_norm.npy"), infer_out["feature_norm"].astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_patch_summary12.npy"), infer_out["patch_summary12"].astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{prefix}_pred_label.npy"), infer_out["pred"].astype(np.int8))


    if infer_out["latent256"] is not None:
        np.save(os.path.join(args.out_dir, f"{prefix}_latent256.npy"), infer_out["latent256"])
    if infer_out["patch_embed768"] is not None:
        np.save(os.path.join(args.out_dir, f"{prefix}_patch_embed768.npy"), infer_out["patch_embed768"])
    if infer_out["hidden512"] is not None:
        np.save(os.path.join(args.out_dir, f"{prefix}_hidden512.npy"), infer_out["hidden512"])

    print("\nSaved rich outputs to:", args.out_dir)

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid point-cloud stream classifier")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train", help="Train the hybrid point-cloud model")
    train_parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    train_parser.add_argument("--tv_root", type=str, default=TV_ROOT,
                              help="Prepared training/validation root")
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    train_parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    train_parser.add_argument("--background_loss_weight", type=float, default=BACKGROUND_LOSS_WEIGHT,
                              help="Class weight for background in CrossEntropyLoss; stream weight remains 1.0")
    train_parser.add_argument("--use_patch_augmentation", type=int, default=int(USE_PATCH_AUGMENTATION),
                              help="1=enable mild train-only patch feature augmentations, 0=disable")
    train_parser.add_argument("--augment_pos_jitter_std", type=float, default=AUGMENT_POS_JITTER_STD)
    train_parser.add_argument("--augment_vel_jitter_std", type=float, default=AUGMENT_VEL_JITTER_STD)
    train_parser.add_argument("--augment_point_dropout_prob", type=float, default=AUGMENT_POINT_DROPOUT_PROB)
    train_parser.add_argument("--augment_feature_dropout_prob", type=float, default=AUGMENT_FEATURE_DROPOUT_PROB)
    train_parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT)
    train_parser.add_argument("--live_log_every", type=int, default=25,
                              help="Write batch-level live training diagnostics every N batches")

    train_parser.add_argument("--patch_k", type=int, default=PATCH_K)
    train_parser.add_argument("--local_box_size", type=float, default=LOCAL_BOX_SIZE)
    train_parser.add_argument("--negative_overlay_stream_prob", type=float, default=NEGATIVE_OVERLAY_STREAM_PROB)

    train_parser.add_argument("--val_stream_file_fraction", type=float, default=VAL_STREAM_FILE_FRACTION)
    train_parser.add_argument("--val_background_particle_fraction", type=float, default=VAL_BACKGROUND_PARTICLE_FRACTION)

    train_parser.add_argument("--train_stream_ref_cap", type=int, default=TRAIN_STREAM_REF_CAP)
    train_parser.add_argument("--train_back_ref_cap", type=int, default=TRAIN_BACK_REF_CAP)
    train_parser.add_argument("--val_stream_ref_cap", type=int, default=VAL_STREAM_REF_CAP)
    train_parser.add_argument("--val_back_ref_cap", type=int, default=VAL_BACK_REF_CAP)

    train_parser.add_argument("--train_stream_centres", type=int, default=TRAIN_STREAM_CENTRES)
    train_parser.add_argument("--train_back_centres", type=int, default=TRAIN_BACK_CENTRES)
    train_parser.add_argument("--val_stream_centres", type=int, default=VAL_STREAM_CENTRES)
    train_parser.add_argument("--val_back_centres", type=int, default=VAL_BACK_CENTRES)

    train_parser.add_argument("--stream_sampling_mode", type=str, default=STREAM_SAMPLING_MODE,
                              choices=["equal_files", "sqrt_size", "proportional"])
    train_parser.add_argument("--background_sampling_mode", type=str, default=BACKGROUND_SAMPLING_MODE,
                              choices=["equal_files", "sqrt_size", "proportional"])
    train_parser.add_argument("--full_background_small_files", type=int, default=0,
                              help="1=load all non-large background files fully and only subsample large synthetic files; 0=original capped sampling across all background files")
    train_parser.add_argument("--background_large_file_count", type=int, default=2,
                              help="Number of largest background files per split to treat as large synthetic files when --full_background_small_files=1")
    train_parser.add_argument("--large_background_file_names", type=str, default="",
                              help="Optional comma-separated basenames for large synthetic backgrounds. If empty, uses the largest N files per split.")
    train_parser.add_argument("--patch_space", type=str, default=PATCH_SPACE, choices=["xyz", "xyzv"])
    train_parser.add_argument("--patch_vel_weight", type=float, default=PATCH_VEL_WEIGHT)

    infer_parser = subparsers.add_parser("infer", help="Run inference on a test dataset")
    infer_parser.add_argument("test_dataset", type=str, help="Test dataset filename in TEST_DIR or full path (.hdf5/.h5/.npy)")
    infer_parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    infer_parser.add_argument("--ckpt_path", type=str, default="")
    infer_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    infer_parser.add_argument("--infer_chunk_size", type=int, default=INFER_CHUNK_SIZE)
    infer_parser.add_argument("--aux_feature_dtype", type=str, default="float16", choices=["float16", "float32"])

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return

    if hasattr(args, "use_patch_augmentation"):
        args.use_patch_augmentation = bool(args.use_patch_augmentation)
    if hasattr(args, "full_background_small_files"):
        args.full_background_small_files = bool(args.full_background_small_files)
    if hasattr(args, "tv_root"):
        configure_training_validation_root(args.tv_root)

    globals()["PATCH_SPACE"] = getattr(args, "patch_space", PATCH_SPACE)
    globals()["PATCH_VEL_WEIGHT"] = getattr(args, "patch_vel_weight", PATCH_VEL_WEIGHT)

    if args.mode == "train":
        train_main(args)
    elif args.mode == "infer":
        infer_main(args)

if __name__ == "__main__":
    main()
