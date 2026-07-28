import os
import time
import argparse
import json
import numpy as np
import h5py
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)

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
OUT_DIR    = os.path.join("outputs", "voxel_cnn")

TEST_POS_SCALE = 1
TEST_PART_TYPE = "PartType4"
BACKGROUND_PART_TYPE = "PartType1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ------------------------------------------------------------
# Data / compute settings
# ------------------------------------------------------------
USE_MEMMAP = True
USE_ANGULAR_MOMENTUM = True
USE_AMP = True
USE_DATA_PARALLEL = True       # used for training only
PIN_MEMORY = True

MAX_STREAM_TRAIN_POINTS = 1_200_000
MAX_BACK_TRAIN_POINTS   = 1_200_000

STREAM_SAMPLING_MODE = "proportional"      # equal_files / sqrt_size / proportional
BACKGROUND_SAMPLING_MODE = "proportional"  # equal_files / sqrt_size / proportional
FULL_BACKGROUND_SMALL_FILES = False
BACKGROUND_LARGE_FILE_COUNT = 2
LARGE_BACKGROUND_FILE_NAMES = ""

BRICK_SIZE = 128.0
GRID_SIZE = 32
TRAIN_BRICKS_PER_EPOCH = 8000
VAL_BRICKS = 1000
MIN_PARTICLES_PER_BRICK = 1024

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------
BATCH_SIZE = 12
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ------------------------------------------------------------
# CPU worker settings
# ------------------------------------------------------------
AVAILABLE_CPUS = 24
NUM_WORKERS_DEFAULT = min(12, max(2, AVAILABLE_CPUS - 4))

# ------------------------------------------------------------
# Output settings
# ------------------------------------------------------------
DEFAULT_THRESHOLD = 0.50
VOXEL_CKPT = "voxel_best.pt"
VOXEL_META = "voxel_meta.json"
VOXEL_LOG = "voxel_training_log.csv"

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
# COMPAT HELPERS
# ============================================================

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
    def __init__(self, keep_probs=False):
        self.keep_probs = bool(keep_probs)
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.n = 0
        self.probs = [] if self.keep_probs else None
        self.targets = [] if self.keep_probs else None

    def update(self, y_true, y_pred, y_prob=None):
        y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
        y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)

        self.tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.tn += int(np.sum((y_true == 0) & (y_pred == 0)))
        self.fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        self.n += int(y_true.size)

        if self.keep_probs and y_prob is not None:
            self.targets.append(y_true.copy())
            self.probs.append(np.asarray(y_prob, dtype=np.float32).reshape(-1))

    def compute(self):
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        n = max(self.n, 1)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        acc = (tp + tn) / n
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)

        prod = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        prod = np.float64(max(float(prod), 1.0))
        denom = float(np.sqrt(prod))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        balanced_acc = 0.5 * (recall + specificity)
        pred_pos_frac = (tp + fp) / n
        true_pos_frac = (tp + fn) / n

        out = {
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "iou": float(iou),
            "dice": float(dice),
            "mcc": float(mcc),
            "balanced_acc": float(balanced_acc),
            "pred_pos_frac": float(pred_pos_frac),
            "true_pos_frac": float(true_pos_frac),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
        return out


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if y_true.size > 0:
            if np.all(y_true == 1) and np.all(y_pred == 1):
                tp = int(y_true.size)
            elif np.all(y_true == 0) and np.all(y_pred == 0):
                tn = int(y_true.size)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    acc = float(np.mean(y_true == y_pred))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0
    try:
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        balanced_acc = 0.5 * (recall + specificity)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "mcc": float(mcc),
        "balanced_acc": float(balanced_acc),
        "pred_pos_frac": float(np.mean(y_pred == 1)),
        "true_pos_frac": float(np.mean(y_true == 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


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
    raise FileNotFoundError("Test dataset not found: {} (also checked {})".format(path_or_name, candidate))

def load_npy(path):
    if USE_MEMMAP:
        return np.load(path, mmap_mode="r")
    return np.load(path)

def recompute_r_vr_from_pos_vel(pos, vel):
    r = np.linalg.norm(pos, axis=1)
    r_safe = np.where(r == 0, 1e-8, r)
    vr = np.sum(pos * vel, axis=1) / r_safe
    return np.column_stack([r, vr]).astype(np.float32)

def build_dataloader(dataset, batch_size, shuffle, drop_last, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
        drop_last=drop_last
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)

def maybe_wrap_dataparallel(model):
    if USE_DATA_PARALLEL and DEVICE == "cuda" and N_GPUS > 1:
        print("Wrapping model in DataParallel across {} GPUs".format(N_GPUS))
        model = nn.DataParallel(model)
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
    """Update the prepared train/validation directory used by training.

    This matches the pointcloud-style --tv_root behaviour. Expected layout:
      tv_root/training/streams
      tv_root/training/background
      tv_root/validation/streams
      tv_root/validation/background
    """
    global TV_ROOT, TRAIN_STREAM_DIR, VAL_STREAM_DIR, TRAIN_BACK_DIR, VAL_BACK_DIR
    global STREAM_DIR, BACK_DIR

    TV_ROOT = os.path.abspath(str(tv_root))
    TRAIN_STREAM_DIR = os.path.join(TV_ROOT, "training", "streams")
    VAL_STREAM_DIR   = os.path.join(TV_ROOT, "validation", "streams")
    TRAIN_BACK_DIR   = os.path.join(TV_ROOT, "training", "background")
    VAL_BACK_DIR     = os.path.join(TV_ROOT, "validation", "background")

    # Backwards-compatible aliases
    STREAM_DIR = TRAIN_STREAM_DIR
    BACK_DIR   = TRAIN_BACK_DIR


def validate_training_validation_root():
    """Fail early if --tv_root does not match the expected split directory layout."""
    required = [TRAIN_STREAM_DIR, VAL_STREAM_DIR, TRAIN_BACK_DIR, VAL_BACK_DIR]
    missing = [d for d in required if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            "Training/validation root is missing required directories:\n  "
            + "\n  ".join(missing)
            + "\nExpected layout: <tv_root>/training|validation/streams|background"
        )

def load_stream_file(path):
    arr = load_npy(path)
    arr = ascontig_f32(arr)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError("Stream file must have shape (N, >=8): {}".format(path))
    return arr[:, :8]

def load_background_npy(path):
    arr = load_npy(path)
    arr = ascontig_f32(arr)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError("Background npy must have shape (N, >=8): {}".format(path))
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
    elif ext in [".hdf5", ".h5"]:
        return load_background_hdf5(path, part_type=part_type)
    raise ValueError("Unsupported background file format: {}".format(path))

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

    elif ext == ".npy":
        arr = ascontig_f32(load_npy(path))
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError("Test npy must have shape (N, >=8): {}".format(path))
        X = arr[:, :8].astype(np.float32)
        centre = X[:, 0:3].mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
        X[:, 0:3] -= centre
        mass = np.ones(len(X), dtype=np.float32)
        ids = np.arange(len(X), dtype=np.int64)
        return X, mass, ids

    raise ValueError("Unsupported test dataset format: {}".format(path))

# ============================================================
# FEATURES
# ============================================================

def add_angular_momentum_features(X):
    if not USE_ANGULAR_MOMENTUM:
        return np.asarray(X, dtype=np.float32)

    X = np.asarray(X, dtype=np.float32)
    pos = X[:, 0:3]
    vel = X[:, 3:6]
    L = np.cross(pos, vel)
    Lmag = np.linalg.norm(L, axis=1, keepdims=True)
    return np.hstack([X, L, Lmag]).astype(np.float32)

def compute_tangential_velocity(X8):
    """Return per-particle tangential speed v_t = sqrt(|v|^2 - v_r^2)."""
    X8 = np.asarray(X8, dtype=np.float32)
    vel = X8[:, 3:6]
    vr = X8[:, 7:8]
    speed2 = np.sum(vel * vel, axis=1, keepdims=True)
    vt2 = np.maximum(speed2 - vr * vr, 0.0)
    return np.sqrt(vt2).astype(np.float32)

def voxel_input_channels():
    # count + mean velocity(3) + mean [r, vr](2) + mean vt(1) + optional L(3), Lmag(1)
    return 11 if USE_ANGULAR_MOMENTUM else 7

def _sample_file_rows(arr, n, rng):
    if n >= len(arr):
        return np.asarray(arr, dtype=np.float32)
    idx = rng.choice(len(arr), size=int(n), replace=False)
    return np.asarray(arr[idx], dtype=np.float32)


def _sampling_counts(cap, sizes, mode="proportional", min_per_nonzero=1):
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
        raise ValueError("Unknown sampling mode: {}".format(mode))

    weights /= weights.sum()
    raw = cap * weights
    counts = np.floor(raw).astype(np.int64)
    counts = np.minimum(counts, sizes.astype(np.int64))

    leftover = int(cap - counts.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    for j in order:
        if leftover <= 0:
            break
        if counts[j] < sizes[j]:
            counts[j] += 1
            leftover -= 1

    eligible = np.where(sizes >= min_per_nonzero)[0]
    for j in eligible:
        if counts[j] == 0 and cap > 0:
            donor = int(np.argmax(counts))
            if counts[donor] > min_per_nonzero:
                counts[donor] -= 1
                counts[j] += 1
    return counts.astype(np.int64)

def _parse_large_background_names(names):
    if names is None:
        return set()
    names = str(names).strip()
    if not names:
        return set()
    return set(x.strip() for x in names.split(",") if x.strip())

def _background_counts(cap, paths, sizes, mode="proportional", full_small_files=False,
                       large_file_count=2, large_file_names=""):
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if not full_small_files:
        return _sampling_counts(cap, sizes_arr, mode=mode, min_per_nonzero=1)

    n_files = len(paths)
    if n_files == 0:
        return np.zeros(0, dtype=np.int64)

    large_mask = np.zeros(n_files, dtype=bool)
    explicit_names = _parse_large_background_names(large_file_names)
    if explicit_names:
        for i, path in enumerate(paths):
            b = os.path.basename(path)
            stem = base_stem(path)
            if b in explicit_names or stem in explicit_names:
                large_mask[i] = True
        if not np.any(large_mask):
            raise ValueError("No --large_background_file_names matched the available background files")
    else:
        n_large = min(int(large_file_count), n_files)
        if n_large > 0:
            large_mask[np.argsort(-sizes_arr)[:n_large]] = True

    counts = sizes_arr.copy()
    large_idx = np.where(large_mask)[0]
    if len(large_idx) > 0:
        counts[large_idx] = _sampling_counts(
            cap, sizes_arr[large_idx], mode=mode, min_per_nonzero=1
        )
    return counts.astype(np.int64)


def build_train_candidate_arrays(max_stream=MAX_STREAM_TRAIN_POINTS, max_back=MAX_BACK_TRAIN_POINTS, split="train",
                                 stream_sampling_mode=STREAM_SAMPLING_MODE,
                                 background_sampling_mode=BACKGROUND_SAMPLING_MODE,
                                 full_background_small_files=FULL_BACKGROUND_SMALL_FILES,
                                 background_large_file_count=BACKGROUND_LARGE_FILE_COUNT,
                                 large_background_file_names=LARGE_BACKGROUND_FILE_NAMES):
    """
    Build candidate particle arrays from the explicit train/validation split.
    Stream files remain split at file level; particle caps only limit the candidate pool size.
    Background particles are sampled proportionally across assigned background files.
    """
    rng = np.random.default_rng(RANDOM_SEED + (0 if split in ("train", "training") else 1000))

    stream_files = list_stream_files(split)
    back_files = list_background_files(split)

    if len(stream_files) == 0:
        d = TRAIN_STREAM_DIR if split in ("train", "training") else VAL_STREAM_DIR
        raise RuntimeError("No stream files found in {}".format(d))
    if len(back_files) == 0:
        d = TRAIN_BACK_DIR if split in ("train", "training") else VAL_BACK_DIR
        raise RuntimeError("No background files found in {}".format(d))

    stream_sizes = [load_stream_file(p).shape[0] for p in stream_files]
    back_sizes = [load_background_file(p).shape[0] for p in back_files]

    stream_counts = _sampling_counts(max_stream, stream_sizes, mode=stream_sampling_mode, min_per_nonzero=32)
    back_counts = _background_counts(
        max_back, back_files, back_sizes, mode=background_sampling_mode,
        full_small_files=full_background_small_files,
        large_file_count=background_large_file_count,
        large_file_names=large_background_file_names,
    )

    print("{} sampling | stream_mode={} background_mode={} full_small_bg={}".format(
        split, stream_sampling_mode, background_sampling_mode, bool(full_background_small_files)
    ))
    if full_background_small_files:
        print("{} background sampling plan:".format(split))
        for path, size, cnt in sorted(zip(back_files, back_sizes, back_counts), key=lambda x: -int(x[1])):
            mode_txt = "FULL" if int(cnt) >= int(size) else "SAMPLED"
            print("  {} | size={:,} | load={:,} | {}".format(base_stem(path), int(size), int(cnt), mode_txt))

    stream_parts = []
    back_parts = []

    print(f"Loading {split} stream candidate pool from {len(stream_files)} whole-file split streams...")
    for path, n in zip(stream_files, stream_counts):
        arr = load_stream_file(path)
        stream_parts.append(_sample_file_rows(arr, int(n), rng))

    print(f"Loading {split} background candidate pool from {len(back_files)} split background files...")
    for path, n in zip(back_files, back_counts):
        arr = load_background_file(path)
        back_parts.append(_sample_file_rows(arr, int(n), rng))

    X_stream = np.vstack(stream_parts).astype(np.float32)
    X_back   = np.vstack(back_parts).astype(np.float32)

    print(f"{split} candidate points | stream={len(X_stream):,} background={len(X_back):,}")
    return X_stream, X_back

def compute_voxel_stats_from_dirs(sample_cap=1_500_000):
    """Compute feature normalization stats from the training split only."""
    rng = np.random.default_rng(RANDOM_SEED)

    stream_files = list_stream_files("train")
    back_files = list_background_files("train")

    if len(stream_files) == 0:
        raise RuntimeError("No training stream files found in {}".format(TRAIN_STREAM_DIR))
    if len(back_files) == 0:
        raise RuntimeError("No training background files found in {}".format(TRAIN_BACK_DIR))

    parts = []

    n_stream_each = max(1, (sample_cap // 2) // len(stream_files))
    n_back_each   = max(1, (sample_cap // 2) // len(back_files))

    for path in stream_files:
        arr = load_stream_file(path)
        n = min(len(arr), n_stream_each)
        idx = rng.choice(len(arr), size=n, replace=False)
        parts.append(np.asarray(arr[idx], dtype=np.float32))

    for path in back_files:
        arr = load_background_file(path)
        n = min(len(arr), n_back_each)
        idx = rng.choice(len(arr), size=n, replace=False)
        parts.append(np.asarray(arr[idx], dtype=np.float32))

    X = np.vstack(parts).astype(np.float32)
    X = add_angular_momentum_features(X)

    stats = {}
    stats["vel_mean"] = np.mean(X[:, 3:6], axis=0).astype(np.float32)
    stats["vel_std"] = np.std(X[:, 3:6], axis=0).astype(np.float32) + 1e-6
    stats["extra_mean"] = np.mean(X[:, 6:8], axis=0).astype(np.float32)
    stats["extra_std"] = np.std(X[:, 6:8], axis=0).astype(np.float32) + 1e-6
    vt = compute_tangential_velocity(X[:, :8])
    stats["vt_mean"] = np.mean(vt, axis=0).astype(np.float32)
    stats["vt_std"] = np.std(vt, axis=0).astype(np.float32) + 1e-6

    if USE_ANGULAR_MOMENTUM:
        stats["L_mean"] = np.mean(X[:, 8:11], axis=0).astype(np.float32)
        stats["L_std"] = np.std(X[:, 8:11], axis=0).astype(np.float32) + 1e-6
        stats["Lmag_mean"] = np.mean(X[:, 11:12], axis=0).astype(np.float32)
        stats["Lmag_std"] = np.std(X[:, 11:12], axis=0).astype(np.float32) + 1e-6

    return stats

# ============================================================
# FAST TRAINING BRICK INDEX
# ============================================================

class SpatialCellIndex(object):
    def __init__(self, X, cell_size):
        self.X = np.asarray(X, dtype=np.float32)
        self.cell_size = float(cell_size)
        self.pos = self.X[:, 0:3]
        self.min_corner = self.pos.min(axis=0)
        cell_ids = np.floor((self.pos - self.min_corner) / self.cell_size).astype(np.int32)

        self.cell_map = {}
        for i in range(len(cell_ids)):
            key = (int(cell_ids[i, 0]), int(cell_ids[i, 1]), int(cell_ids[i, 2]))
            if key not in self.cell_map:
                self.cell_map[key] = []
            self.cell_map[key].append(i)

        for key in self.cell_map:
            self.cell_map[key] = np.asarray(self.cell_map[key], dtype=np.int64)

    def query_brick(self, centre, brick_size):
        lo = centre - 0.5 * brick_size
        hi = centre + 0.5 * brick_size

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
            return np.empty(0, dtype=np.int64), lo, hi

        idx = np.concatenate(candidates)
        pos = self.pos[idx]
        mask = np.all((pos >= lo) & (pos < hi), axis=1)
        return idx[mask], lo, hi

# ============================================================
# VOXELISATION
# ============================================================

def voxelize_brick(X, y, lo, hi, stats, grid_size=32):
    X = add_angular_momentum_features(X)

    pos = X[:, 0:3]
    vel = X[:, 3:6]
    extra = X[:, 6:8]
    vt = compute_tangential_velocity(X[:, :8])

    span = np.maximum(hi - lo, 1e-6)
    normed = (pos - lo) / span
    ijk = np.floor(normed * grid_size).astype(np.int32)
    ijk = np.clip(ijk, 0, grid_size - 1)

    C = voxel_input_channels()
    voxel = np.zeros((C, grid_size, grid_size, grid_size), dtype=np.float32)

    count_all = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    count_stream = np.zeros_like(count_all)
    count_bg = np.zeros_like(count_all)

    sum_v = np.zeros((3, grid_size, grid_size, grid_size), dtype=np.float32)
    sum_extra = np.zeros((2, grid_size, grid_size, grid_size), dtype=np.float32)
    sum_vt = np.zeros((1, grid_size, grid_size, grid_size), dtype=np.float32)

    vel_norm = (vel - stats["vel_mean"]) / stats["vel_std"]
    extra_norm = (extra - stats["extra_mean"]) / stats["extra_std"]
    vt_norm = (vt - stats["vt_mean"]) / stats["vt_std"]

    if USE_ANGULAR_MOMENTUM:
        L = X[:, 8:11]
        Lmag = X[:, 11:12]
        L_norm = (L - stats["L_mean"]) / stats["L_std"]
        Lmag_norm = (Lmag - stats["Lmag_mean"]) / stats["Lmag_std"]

        sum_L = np.zeros((3, grid_size, grid_size, grid_size), dtype=np.float32)
        sum_Lmag = np.zeros((1, grid_size, grid_size, grid_size), dtype=np.float32)

    for n, (i, j, k) in enumerate(ijk):
        count_all[i, j, k] += 1.0
        if y[n] == 1:
            count_stream[i, j, k] += 1.0
        else:
            count_bg[i, j, k] += 1.0

        sum_v[:, i, j, k] += vel_norm[n]
        sum_extra[:, i, j, k] += extra_norm[n]
        sum_vt[:, i, j, k] += vt_norm[n]

        if USE_ANGULAR_MOMENTUM:
            sum_L[:, i, j, k] += L_norm[n]
            sum_Lmag[:, i, j, k] += Lmag_norm[n]

    nonzero = count_all > 0
    voxel[0] = np.log1p(count_all)

    for c in range(3):
        tmp = np.zeros_like(count_all)
        tmp[nonzero] = sum_v[c][nonzero] / count_all[nonzero]
        voxel[1 + c] = tmp

    for c in range(2):
        tmp = np.zeros_like(count_all)
        tmp[nonzero] = sum_extra[c][nonzero] / count_all[nonzero]
        voxel[4 + c] = tmp

    tmp = np.zeros_like(count_all)
    tmp[nonzero] = sum_vt[0][nonzero] / count_all[nonzero]
    voxel[6] = tmp

    if USE_ANGULAR_MOMENTUM:
        for c in range(3):
            tmp = np.zeros_like(count_all)
            tmp[nonzero] = sum_L[c][nonzero] / count_all[nonzero]
            voxel[7 + c] = tmp

        tmp = np.zeros_like(count_all)
        tmp[nonzero] = sum_Lmag[0][nonzero] / count_all[nonzero]
        voxel[10] = tmp

    target = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
    target[count_stream > count_bg] = 1

    return voxel, target, ijk, count_all

# ============================================================
# DATASET
# ============================================================

class BrickDataset(Dataset):
    def __init__(self, stream_arr, bg_arr, stats, brick_size, grid_size, n_bricks, min_particles):
        self.stream_arr = np.asarray(stream_arr, dtype=np.float32)
        self.bg_arr = np.asarray(bg_arr, dtype=np.float32)
        self.stats = stats
        self.brick_size = brick_size
        self.grid_size = grid_size
        self.n_bricks = n_bricks
        self.min_particles = min_particles
        self.rng = np.random.default_rng(RANDOM_SEED)

        cell_size = brick_size
        self.stream_index = SpatialCellIndex(self.stream_arr, cell_size)
        self.bg_index = SpatialCellIndex(self.bg_arr, cell_size)

        ns = min(len(self.stream_arr), 300000)
        nb = min(len(self.bg_arr), 300000)

        idx_s = self.rng.choice(len(self.stream_arr), size=ns, replace=False)
        idx_b = self.rng.choice(len(self.bg_arr), size=nb, replace=False)

        self.candidate_centres = np.vstack([
            np.asarray(self.stream_arr[idx_s, 0:3], dtype=np.float32),
            np.asarray(self.bg_arr[idx_b, 0:3], dtype=np.float32)
        ])

    def __len__(self):
        return self.n_bricks

    def __getitem__(self, idx):
        for _ in range(50):
            centre = self.candidate_centres[self.rng.integers(0, len(self.candidate_centres))]

            idx_s, lo, hi = self.stream_index.query_brick(centre, self.brick_size)
            idx_b, _, _ = self.bg_index.query_brick(centre, self.brick_size)

            if len(idx_s) + len(idx_b) < self.min_particles:
                continue

            Xs = self.stream_arr[idx_s]
            Xb = self.bg_arr[idx_b]

            X = np.vstack([Xs, Xb]).astype(np.float32)
            y = np.concatenate([
                np.ones(len(Xs), dtype=np.int64),
                np.zeros(len(Xb), dtype=np.int64)
            ])

            voxel, target, _, _ = voxelize_brick(X, y, lo, hi, self.stats, self.grid_size)
            return torch.from_numpy(voxel), torch.from_numpy(target)

        C = voxel_input_channels()
        voxel = np.zeros((C, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        target = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int64)
        return torch.from_numpy(voxel), torch.from_numpy(target)

# ============================================================
# MODEL
# ============================================================

class ResBlock3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(c_out)

        if c_in != c_out:
            self.skip = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + s, inplace=True)
        return x

class VoxelStreamNet(nn.Module):
    def __init__(self, in_channels):
        super(VoxelStreamNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock3D(24, 32)
        self.block2 = ResBlock3D(32, 48)
        self.block3 = ResBlock3D(48, 48)
        self.head = nn.Conv3d(48, 2, kernel_size=1)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        feat_top = self.block3(x)
        logits = self.head(feat_top)

        if return_features:
            return logits, {"feat_top": feat_top}
        return logits

# ============================================================
# TRAIN / EVAL
# ============================================================

def print_binary_metrics(name, y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    print("\n{} metrics @ threshold={:.2f}".format(name, threshold))
    print("Precision: {:.4f}".format(p))
    print("Recall:    {:.4f}".format(r))
    print("F1:        {:.4f}".format(f))
    print(classification_report(y_true, pred, digits=4, zero_division=0))
    return pred


def run_epoch(model, loader, optimizer, criterion, scaler, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0
    metric_acc = BinaryMetricAccumulator(keep_probs=False)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with autocast_context():
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

        y_true_np = y_batch.detach().cpu().numpy().reshape(-1)
        y_pred_np = pred.detach().cpu().numpy().reshape(-1)
        metric_acc.update(y_true_np, y_pred_np)

        total_loss += float(loss.detach().item())
        n_batches += 1

    out = metric_acc.compute()
    out["loss"] = total_loss / max(n_batches, 1)
    return out

# ============================================================
# INFERENCE
# ============================================================

def infer_full_dataset(model, X_test, stats, threshold=0.5, stride_fraction=0.5, edge_weight_power=2.0):
    """
    Overlapping, shifted-brick voxel inference.

    Instead of assigning each particle to exactly one non-overlapping brick, this sweeps
    the test set with a stride smaller than BRICK_SIZE. Particles can be seen in several
    brick contexts. Predictions are edge-weighted so particles near a brick centre count
    more than particles near a brick boundary. This reduces hard cut-off artefacts at
    brick edges.
    """
    model.eval()

    X_test = np.asarray(X_test, dtype=np.float32)
    xyz = X_test[:, 0:3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    step = float(BRICK_SIZE) * float(stride_fraction)
    if step <= 0:
        raise ValueError("stride_fraction must produce a positive step")

    half = 0.5 * float(BRICK_SIZE)
    index = SpatialCellIndex(X_test, BRICK_SIZE)

    xs = np.arange(mins[0] + half, maxs[0] + half + 1e-6, step, dtype=np.float32)
    ys = np.arange(mins[1] + half, maxs[1] + half + 1e-6, step, dtype=np.float32)
    zs = np.arange(mins[2] + half, maxs[2] + half + 1e-6, step, dtype=np.float32)

    prob_accum = np.zeros(len(X_test), dtype=np.float64)
    logit_accum = np.zeros(len(X_test), dtype=np.float64)
    margin_accum = np.zeros(len(X_test), dtype=np.float64)
    density_accum = np.zeros(len(X_test), dtype=np.float64)
    energy_accum = np.zeros(len(X_test), dtype=np.float64)
    weight_sum = np.zeros(len(X_test), dtype=np.float64)
    coverage = np.zeros(len(X_test), dtype=np.float32)

    n_centres = len(xs) * len(ys) * len(zs)
    done = 0
    used = 0

    with torch.no_grad():
        for cx in xs:
            for cy in ys:
                for cz in zs:
                    centre = np.array([cx, cy, cz], dtype=np.float32)
                    idx = index.query_brick(centre, BRICK_SIZE)[0]
                    done += 1

                    if len(idx) == 0:
                        continue

                    Xb = X_test[idx]
                    y_dummy = np.zeros(len(Xb), dtype=np.int64)
                    lo = centre - half
                    hi = centre + half

                    voxel, _, ijk, count_all = voxelize_brick(Xb, y_dummy, lo, hi, stats, GRID_SIZE)
                    x_t = torch.from_numpy(voxel[None]).to(DEVICE, non_blocking=True)

                    with autocast_context():
                        logits, aux = model(x_t, return_features=True)
                        prob_grid = torch.softmax(logits, dim=1)[0, 1]
                        margin_grid = logits[0, 1] - logits[0, 0]
                        feat_top = aux["feat_top"][0]
                        feat_energy_grid = torch.norm(feat_top, dim=0)

                    prob_grid = prob_grid.detach().cpu().numpy()
                    logit_grid = logits[0, 1].detach().cpu().numpy()
                    margin_grid = margin_grid.detach().cpu().numpy()
                    feat_energy_grid = feat_energy_grid.detach().cpu().numpy()

                    p = prob_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    lg = logit_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    mg = margin_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    den = count_all[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    eng = feat_energy_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

                    rel = np.max(np.abs((xyz[idx] - centre[None, :]) / max(half, 1e-6)), axis=1)
                    rel = np.clip(rel, 0.0, 1.0)
                    w = (1.0 - rel) ** float(edge_weight_power)
                    w = np.maximum(w, 1e-3).astype(np.float64)

                    prob_accum[idx] += p * w
                    logit_accum[idx] += lg * w
                    margin_accum[idx] += mg * w
                    density_accum[idx] += den * w
                    energy_accum[idx] += eng * w
                    weight_sum[idx] += w
                    coverage[idx] += 1.0
                    used += 1

                    if done % 100 == 0:
                        print("Processed voxel centres: {}/{} | used {}".format(done, n_centres, used))

    valid = weight_sum > 0
    probs = np.zeros(len(X_test), dtype=np.float32)
    logits_stream = np.zeros(len(X_test), dtype=np.float32)
    margin = np.zeros(len(X_test), dtype=np.float32)
    voxel_density = np.zeros(len(X_test), dtype=np.float32)
    feature_energy = np.zeros(len(X_test), dtype=np.float32)

    probs[valid] = (prob_accum[valid] / weight_sum[valid]).astype(np.float32)
    logits_stream[valid] = (logit_accum[valid] / weight_sum[valid]).astype(np.float32)
    margin[valid] = (margin_accum[valid] / weight_sum[valid]).astype(np.float32)
    voxel_density[valid] = (density_accum[valid] / weight_sum[valid]).astype(np.float32)
    feature_energy[valid] = (energy_accum[valid] / weight_sum[valid]).astype(np.float32)

    pred = (probs >= threshold).astype(np.int64)
    if np.any(valid):
        print(
            "Overlapped voxel inference complete | centres={} | used={} | mean coverage={:.2f}".format(
                n_centres, used, float(coverage[valid].mean())
            )
        )
    else:
        print("Overlapped voxel inference complete but no valid particle coverage was produced.")

    return probs, logits_stream, margin, voxel_density, feature_energy, pred

# ============================================================
# SAVE
# ============================================================

def save_particle_outputs(
    out_dir,
    prefix,
    X,
    probs,
    logits,
    margins,
    pred,
    feature_metric_1,
    feature_metric_1_name,
    feature_metric_2=None,
    feature_metric_2_name=None,
    latent64=None,
    mass=None,
    ids=None,
):
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "{}_pos.npy".format(prefix)), X[:, 0:3].astype(np.float32))
    np.save(os.path.join(out_dir, "{}_vel.npy".format(prefix)), X[:, 3:6].astype(np.float32))
    np.save(os.path.join(out_dir, "{}_r_vr.npy".format(prefix)), X[:, 6:8].astype(np.float32))

    np.save(os.path.join(out_dir, "{}_prob_stream.npy".format(prefix)), probs.astype(np.float32))
    np.save(os.path.join(out_dir, "{}_logit_stream.npy".format(prefix)), logits.astype(np.float32))
    np.save(os.path.join(out_dir, "{}_margin.npy".format(prefix)), margins.astype(np.float32))
    np.save(os.path.join(out_dir, "{}_pred_label.npy".format(prefix)), pred.astype(np.int8))

    np.save(
        os.path.join(out_dir, "{}_{}.npy".format(prefix, feature_metric_1_name)),
        feature_metric_1.astype(np.float32)
    )

    if feature_metric_2 is not None and feature_metric_2_name is not None:
        np.save(
            os.path.join(out_dir, "{}_{}.npy".format(prefix, feature_metric_2_name)),
            feature_metric_2.astype(np.float32)
        )

    if latent64 is not None:
        np.save(os.path.join(out_dir, "{}_latent64.npy".format(prefix)), latent64.astype(np.float32))

    if mass is not None:
        np.save(os.path.join(out_dir, "{}_mass.npy".format(prefix)), mass.astype(np.float32))

    if ids is not None:
        np.save(os.path.join(out_dir, "{}_ids.npy".format(prefix)), ids.astype(np.int64))

# ============================================================
# TRAIN / INFER
# ============================================================

def train_main(args):
    ensure_dir(args.out_dir)

    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPUs:", N_GPUS)
    print("Using device:", DEVICE)
    print("Training/validation root:", TV_ROOT)
    print("Train streams:", TRAIN_STREAM_DIR)
    print("Validation streams:", VAL_STREAM_DIR)
    print("Train background:", TRAIN_BACK_DIR)
    print("Validation background:", VAL_BACK_DIR)
    print("Computing voxel stats...")
    stats = compute_voxel_stats_from_dirs()

    print("Building sampled training arrays...")
    train_stream, train_bg = build_train_candidate_arrays(
        max_stream=args.max_stream_train_points,
        max_back=args.max_back_train_points,
        split="train",
        stream_sampling_mode=args.stream_sampling_mode,
        background_sampling_mode=args.background_sampling_mode,
        full_background_small_files=args.full_background_small_files,
        background_large_file_count=args.background_large_file_count,
        large_background_file_names=args.large_background_file_names,
    )

    print("Building sampled validation arrays...")
    val_stream, val_bg = build_train_candidate_arrays(
        max_stream=args.val_stream_points,
        max_back=args.val_back_points,
        split="val",
        stream_sampling_mode=args.stream_sampling_mode,
        background_sampling_mode=args.background_sampling_mode,
        full_background_small_files=args.full_background_small_files,
        background_large_file_count=args.background_large_file_count,
        large_background_file_names=args.large_background_file_names,
    )

    print("Building train/val datasets...")
    train_ds = BrickDataset(
        train_stream, train_bg, stats,
        brick_size=args.brick_size,
        grid_size=args.grid_size,
        n_bricks=args.train_bricks,
        min_particles=args.min_particles_per_brick
    )
    val_ds = BrickDataset(
        val_stream, val_bg, stats,
        brick_size=args.brick_size,
        grid_size=args.grid_size,
        n_bricks=args.val_bricks,
        min_particles=args.min_particles_per_brick
    )

    train_dl = build_dataloader(train_ds, args.batch_size, True, True, args.num_workers)
    val_dl = build_dataloader(val_ds, args.batch_size, False, False, args.num_workers)

    in_channels = voxel_input_channels()
    model = VoxelStreamNet(in_channels=in_channels).to(DEVICE)
    model = maybe_wrap_dataparallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler()

    best_val = np.inf
    best_path = os.path.join(args.out_dir, VOXEL_CKPT)
    log_path = os.path.join(args.out_dir, VOXEL_LOG)
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = CSVLogger(log_path)

    print("Training...")
    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = run_epoch(model, train_dl, optimizer, criterion, scaler, train=True)
        val_metrics = run_epoch(model, val_dl, optimizer, criterion, scaler, train=False)

        dt = time.time() - t0
        logger.write({
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_specificity": float(train_metrics["specificity"]),
            "train_f1": float(train_metrics["f1"]),
            "train_iou": float(train_metrics["iou"]),
            "train_dice": float(train_metrics["dice"]),
            "train_mcc": float(train_metrics["mcc"]),
            "train_balanced_acc": float(train_metrics["balanced_acc"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_f1": float(val_metrics["f1"]),
            "val_iou": float(val_metrics["iou"]),
            "val_dice": float(val_metrics["dice"]),
            "val_mcc": float(val_metrics["mcc"]),
            "val_balanced_acc": float(val_metrics["balanced_acc"]),
            "epoch_seconds": float(dt),
        })
        print(
            "Epoch {:03d}/{} | "
            "train_loss={:.6f} acc={:.4f} f1={:.4f} iou={:.4f} dice={:.4f} mcc={:.4f} | "
            "val_loss={:.6f} acc={:.4f} f1={:.4f} iou={:.4f} dice={:.4f} mcc={:.4f} | "
            "time={:.1f}s".format(
                epoch + 1, args.epochs,
                train_metrics["loss"], train_metrics["acc"], train_metrics["f1"], train_metrics["iou"], train_metrics["dice"], train_metrics["mcc"],
                val_metrics["loss"], val_metrics["acc"], val_metrics["f1"], val_metrics["iou"], val_metrics["dice"], val_metrics["mcc"],
                dt
            )
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": get_model_state_dict(model),
                    "stats": stats,
                    "use_angular_momentum": USE_ANGULAR_MOMENTUM,
                    "brick_size": args.brick_size,
                    "grid_size": args.grid_size,
                    "in_channels": int(in_channels),
                    "voxel_feature_channels": ["log_count", "mean_vx", "mean_vy", "mean_vz", "mean_r", "mean_vr", "mean_vt"] + (["mean_Lx", "mean_Ly", "mean_Lz", "mean_Lmag"] if USE_ANGULAR_MOMENTUM else []),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "tv_root": TV_ROOT,
                },
                best_path
            )

    meta_path = os.path.join(args.out_dir, VOXEL_META)
    with open(meta_path, "w") as f:
        json.dump({"brick_size": args.brick_size, "grid_size": args.grid_size, "use_angular_momentum": USE_ANGULAR_MOMENTUM, "in_channels": int(voxel_input_channels()), "voxel_feature_channels": ["log_count", "mean_vx", "mean_vy", "mean_vz", "mean_r", "mean_vr", "mean_vt"] + (["mean_Lx", "mean_Ly", "mean_Lz", "mean_Lmag"] if USE_ANGULAR_MOMENTUM else []), "training_log": VOXEL_LOG, "tv_root": TV_ROOT, "rich_output_files": ["*_pos.npy","*_vel.npy","*_r_vr.npy","*_mass.npy","*_ids.npy","*_prob_stream.npy","*_logit_stream.npy","*_margin.npy","*_pred_label.npy","*_voxel_density.npy","*_feature_energy.npy"]}, f, indent=2)
    print("\nSaved trained model to: {}".format(best_path))
    print("Saved metadata to: {}".format(meta_path))

def infer_main(args):
    global USE_ANGULAR_MOMENTUM, BRICK_SIZE, GRID_SIZE

    ensure_dir(args.out_dir)

    ckpt_path = args.ckpt_path if getattr(args, "ckpt_path", "") else os.path.join(args.out_dir, VOXEL_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(ckpt_path))

    print("Loading trained checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    stats = ckpt["stats"]
    USE_ANGULAR_MOMENTUM = ckpt["use_angular_momentum"]
    BRICK_SIZE = ckpt["brick_size"]
    GRID_SIZE = ckpt["grid_size"]

    in_channels = voxel_input_channels()

    # IMPORTANT:
    # Do NOT wrap in DataParallel for inference here.
    # infer_full_dataset processes one brick at a time (batch size = 1),
    # and DataParallel can fail when splitting batch=1 across 2 GPUs.
    model = VoxelStreamNet(in_channels=in_channels).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset_path = resolve_test_dataset_path(args.test_dataset)
    print("Loading test dataset...")
    X_test, mass_test, ids_test = load_test_snapshot(
        test_dataset_path,
        part_type=TEST_PART_TYPE,
        pos_scale=TEST_POS_SCALE
    )

    print("Running inference...")
    probs, logits, margins, voxel_density, feature_energy, pred = infer_full_dataset(
        model,
        X_test,
        stats,
        threshold=args.threshold,
        stride_fraction=args.infer_stride_fraction,
        edge_weight_power=args.edge_weight_power
    )

    prefix = base_stem(test_dataset_path)
    print("Saving outputs...")
    save_particle_outputs(
        out_dir=args.out_dir,
        prefix=prefix,
        X=X_test,
        probs=probs,
        logits=logits,
        margins=margins,
        pred=pred,
        feature_metric_1=voxel_density,
        feature_metric_1_name="voxel_density",
        feature_metric_2=feature_energy,
        feature_metric_2_name="feature_energy",
        latent64=None,
        mass=mass_test,
        ids=ids_test,
    )

    print("\nSaved to: {}".format(args.out_dir))

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Voxel stream classifier")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train", help="Train the voxel model")
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    train_parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT)
    train_parser.add_argument("--brick_size", type=float, default=BRICK_SIZE)
    train_parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    train_parser.add_argument("--train_bricks", type=int, default=TRAIN_BRICKS_PER_EPOCH)
    train_parser.add_argument("--val_bricks", type=int, default=VAL_BRICKS)
    train_parser.add_argument("--min_particles_per_brick", type=int, default=MIN_PARTICLES_PER_BRICK)
    train_parser.add_argument("--max_stream_train_points", type=int, default=MAX_STREAM_TRAIN_POINTS)
    train_parser.add_argument("--max_back_train_points", type=int, default=MAX_BACK_TRAIN_POINTS)
    train_parser.add_argument("--val_stream_points", "--val_stream_ref_cap", dest="val_stream_points", type=int, default=MAX_STREAM_TRAIN_POINTS,
                              help="Validation stream candidate cap. Defaults to training stream cap for backwards compatibility.")
    train_parser.add_argument("--val_back_points", "--val_back_ref_cap", dest="val_back_points", type=int, default=MAX_BACK_TRAIN_POINTS,
                              help="Validation background candidate cap. Defaults to training background cap for backwards compatibility.")
    train_parser.add_argument("--stream_sampling_mode", type=str, default=STREAM_SAMPLING_MODE,
                              choices=["equal_files", "sqrt_size", "proportional"])
    train_parser.add_argument("--background_sampling_mode", type=str, default=BACKGROUND_SAMPLING_MODE,
                              choices=["equal_files", "sqrt_size", "proportional"])
    train_parser.add_argument("--full_background_small_files", type=int, default=int(FULL_BACKGROUND_SMALL_FILES),
                              help="1=load all non-large background files fully and only subsample large synthetic files; 0=capped sampling across all background files")
    train_parser.add_argument("--background_large_file_count", type=int, default=BACKGROUND_LARGE_FILE_COUNT,
                              help="Number of largest background files per split to treat as large synthetic files when --full_background_small_files=1")
    train_parser.add_argument("--large_background_file_names", type=str, default=LARGE_BACKGROUND_FILE_NAMES,
                              help="Optional comma-separated basenames for large synthetic backgrounds. If empty, uses the largest N files per split.")
    train_parser.add_argument("--tv_root", type=str, default=TV_ROOT,
                              help="Prepared training/validation root")
    train_parser.add_argument("--out_dir", type=str, default=OUT_DIR)

    infer_parser = subparsers.add_parser("infer", help="Run inference on a test dataset")
    infer_parser.add_argument("test_dataset", type=str, help="Test dataset filename in TEST_DIR or full path (.hdf5/.h5/.npy)")
    infer_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    infer_parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    infer_parser.add_argument("--ckpt_path", type=str, default="",
                              help="Explicit path to voxel_best.pt. If omitted, uses --out_dir/voxel_best.pt")
    infer_parser.add_argument("--infer_stride_fraction", type=float, default=0.5,
                              help="Stride as a fraction of brick size for overlapped inference. 0.5 gives 50% overlap.")
    infer_parser.add_argument("--edge_weight_power", type=float, default=2.0,
                              help="Down-weight brick edges during overlapped inference. 0 gives equal weighting.")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    if hasattr(args, "full_background_small_files"):
        args.full_background_small_files = bool(args.full_background_small_files)

    if args.mode == "train" and hasattr(args, "tv_root"):
        configure_training_validation_root(args.tv_root)
        validate_training_validation_root()

    if args.mode == "train":
        train_main(args)
    elif args.mode == "infer":
        infer_main(args)

if __name__ == "__main__":
    main()
