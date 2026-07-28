#!/usr/bin/env python3
"""GPU stellar-stream classifier using FAISS features, CuPy, and XGBoost.

What this script does:
- TRAIN: builds pointcloud-style scalar/patch-summary features for all training stream .npy files + background .hdf5 or .npy files, trains XGBoost, saves model+metadata.
- INFER: loads a test snapshot (filename only, assumed in same directory as this script),
         computes features, predicts probabilities, applies prior correction (optional),
         selects stream particles by TOP-FRACTION (default) or thresholding,
         and saves outputs with a prefix equal to the snapshot filename stem.

Background training files can be:
    * .hdf5 Gadget snapshots
    * .npy arrays in format [x,y,z,vx,vy,vz,radii,vradii]

Example:
  python xgboost.py train --chunk 500000
  python xgboost.py infer testdataset.hdf5 --top_frac 0.03
"""

import os
import json
import time
import gc
import argparse
import numpy as np
import cupy as cp
import xgboost as xgb
from xgboost.callback import TrainingCallback
import h5py
import faiss
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

# ============================================================
# TRAINING DATA PATHS (edit to your locations)
# ============================================================

TV_DIR = os.path.join("data", "training_validation")
TRAIN_STREAM_DIR = os.path.join(TV_DIR, "training", "streams")
VAL_STREAM_DIR   = os.path.join(TV_DIR, "validation", "streams")
TRAIN_BACK_DIR   = os.path.join(TV_DIR, "training", "background")
VAL_BACK_DIR     = os.path.join(TV_DIR, "validation", "background")
TEST_DIR = os.path.join("data", "test")
OUT_DIR = os.path.join("outputs", "xgboost")

# ============================================================
# OUTPUT FILE NAMES (saved in the script directory by default)
# ============================================================

OUTPUT_FOLDERNAME = "xgboost_outputs"
MODEL_FILENAME = "stream_classifier.json"
META_FILENAME = "feature_metadata.json"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# feature settings
K = 48
DEFAULT_CHUNK = 100_000
PATCH_VEL_WEIGHT = 0.05

# units: test snapshot is Mpc, training streams are kpc -> multiply by 1000 to match kpc
TEST_POS_SCALE = 1

# FAISS policy
MIN_IVF_POINTS = 200_000
IVF_TRAIN_CAP  = 2_000_000
IVF_NLIST_CAP  = 16384
IVF_NPROBE_CAP = 64
FAISS_SEARCH_BATCH = 100_000

# classification / priors
TRAIN_STREAM_PRIOR = 0.5
TEST_STREAM_PRIOR_DEFAULT = 0.07

USE_PRIOR_CORRECTION_DEFAULT = True  # can disable via CLI
USE_TOP_FRACTION_DEFAULT = True      # can disable via CLI
TOP_FRACTION_DEFAULT = 0.07          # can override via CLI
VAL_FRACTION = 0.15
XGB_LOG_FILENAME = "xgboost_training_log.csv"
XGB_EVAL_HISTORY = "xgboost_eval_history.csv"

# ============================================================
# Helpers
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


class XGBEvalLogger(TrainingCallback):
    def __init__(self, logger):
        self.logger = logger

    def after_iteration(self, model, epoch, evals_log):
        row = {"round": int(epoch)}
        for dataset_name, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                row[f"{dataset_name}_{metric_name}"] = float(values[-1])
        self.logger.write(row)
        return False


def compute_binary_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    y_pred = (probs >= float(threshold)).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

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
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, probs))
    except Exception:
        auprc = float("nan")

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
        "auc": float(auc),
        "auprc": float(auprc),
        "pred_pos_frac": float(np.mean(y_pred == 1)),
        "true_pos_frac": float(np.mean(y_true == 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def now() -> float:
    return time.perf_counter()

def fmt(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    if sec < 3600:
        return f"{sec/60:.2f}m"
    return f"{sec/3600:.2f}h"

def ascontig_f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def base_stem(path_or_fname: str) -> str:
    return os.path.splitext(os.path.basename(path_or_fname))[0]


def resolve_test_dataset_path(path_or_name: str, test_dir: str = None) -> str:
    search_dir = TEST_DIR if test_dir is None else test_dir

    if os.path.isabs(path_or_name) or os.path.exists(path_or_name):
        return path_or_name

    candidate = os.path.join(search_dir, path_or_name)

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Test dataset not found: {path_or_name} "
        f"(also checked {candidate})"
    )

# ============================================================
# FAISS GPU index builders (robust)
# ============================================================

def build_faiss_flat_gpu(pos_np: np.ndarray):
    pos_np = ascontig_f32(pos_np)
    d = pos_np.shape[1]
    index_cpu = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(index_cpu)
    gpu_index.add(pos_np)
    return gpu_index, {"type": "flat", "N": int(pos_np.shape[0])}

def build_faiss_ivf_gpu(pos_np: np.ndarray,
                        train_cap: int = IVF_TRAIN_CAP,
                        seed: int = RANDOM_SEED):
    """
    Build IVF directly on GPU 0 (more robust than CPU->all_gpus IVF in some builds).
    """
    pos_np = ascontig_f32(pos_np)
    N, d = pos_np.shape

    rng = np.random.default_rng(seed)
    train_size = min(N, train_cap)
    if train_size < N:
        train_idx = rng.choice(N, size=train_size, replace=False)
        train_data = pos_np[train_idx]
    else:
        train_data = pos_np

    ntrain = train_data.shape[0]

    nlist = int(np.clip(np.sqrt(N), 2048, IVF_NLIST_CAP))
    nlist = min(nlist, ntrain)
    if nlist < 64 or ntrain < 2048:
        raise RuntimeError(f"IVF not viable: N={N}, ntrain={ntrain}, nlist={nlist}")

    nprobe = int(np.clip(nlist // 256, 8, IVF_NPROBE_CAP))
    nprobe = min(nprobe, nlist)

    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFFlatConfig()
    config.device = 0
    config.useFloat16 = False

    quantizer = faiss.IndexFlatL2(d)
    index_gpu = faiss.GpuIndexIVFFlat(res, quantizer, d, nlist, faiss.METRIC_L2, config)

    index_gpu.train(train_data)
    index_gpu.add(pos_np)
    index_gpu.nprobe = nprobe

    return index_gpu, {
        "type": "ivf_gpu0",
        "N": int(N),
        "ntrain": int(ntrain),
        "nlist": int(nlist),
        "nprobe": int(nprobe)
    }

def build_faiss_index_adaptive(pos_np: np.ndarray):
    pos_np = ascontig_f32(pos_np)
    N = pos_np.shape[0]

    if N < MIN_IVF_POINTS:
        t0 = now()
        idx, meta = build_faiss_flat_gpu(pos_np)
        meta["build_s"] = now() - t0
        return idx, meta

    t0 = now()
    try:
        idx, meta = build_faiss_ivf_gpu(pos_np)
        meta["build_s"] = now() - t0
        return idx, meta
    except Exception as e:
        print(f"[FAISS] IVF build failed ({type(e).__name__}: {e}). Falling back to Flat.")
        t1 = now()
        idx, meta = build_faiss_flat_gpu(pos_np)
        meta["build_s"] = now() - t1
        meta["fallback_reason"] = str(e)
        return idx, meta

# ============================================================
# FAISS search wrapper (batched)
# ============================================================

def faiss_search_batched(index, queries_np: np.ndarray, k: int, batch: int = FAISS_SEARCH_BATCH):
    queries_np = ascontig_f32(queries_np)
    nq = queries_np.shape[0]
    D_all = np.empty((nq, k), dtype=np.float32)
    I_all = np.empty((nq, k), dtype=np.int64)

    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        q = queries_np[s:e]
        D, I = index.search(q, k)
        D_all[s:e] = D
        I_all[s:e] = I

    return D_all, I_all

# ============================================================
# Features
# ============================================================

def recompute_scalar_phase_features(pos_gpu, vel_gpu):
    """
    Return centre-particle scalar features only.

    These intentionally exclude raw x,y,z,vx,vy,vz so the classifier cannot learn
    simulation-location shortcuts. They match the scalar quantities used by the
    pointcloud code: r, vr, speed, vt, and |L|.
    """
    r = cp.linalg.norm(pos_gpu, axis=1)
    speed = cp.linalg.norm(vel_gpu, axis=1)
    vr = cp.sum(pos_gpu * vel_gpu, axis=1) / cp.maximum(r, 1e-8)
    vt = cp.sqrt(cp.maximum(speed * speed - vr * vr, 0.0))
    L = cp.cross(pos_gpu, vel_gpu)
    L_mag = cp.linalg.norm(L, axis=1)
    return r, vr, speed, vt, L, L_mag


XGB_FEATURE_NAMES = [
    # centre-particle scalar features; no raw position or velocity components
    "centre_r",
    "centre_vr",
    "centre_speed",
    "centre_vt",
    "centre_lmag",

    # pointcloud-style local patch summaries
    "patch_mean_dist",
    "patch_max_dist",
    "patch_sigma_v",
    "patch_sigma_vr",
    "patch_anis_ratio",
    "patch_flatness",
    "patch_L_align",
    "patch_mean_speed_ratio",
    "patch_mean_abs_dvr",
    "patch_mean_vt",
    "patch_lmag_ratio",
]


def compute_features_gpu(pos: np.ndarray, vel: np.ndarray, k: int = K, chunk: int = DEFAULT_CHUNK):
    """
    Build XGBoost features from pointcloud-relevant scalar + local patch quantities.

    Excluded from the model input:
      - raw x,y,z
      - raw vx,vy,vz
      - raw Lx,Ly,Lz orientation components

    Included:
      - centre scalar phase-space features: r, vr, speed, vt, |L|
      - the same 12 patch-summary quantities used by the pointcloud model:
        mean_dist, max_dist, sigma_v, sigma_vr, anis_ratio, flatness,
        L_align, mean_speed_ratio, mean_abs_dvr, mean_vt, lmag_ratio
    """
    t_total = now()

    pos = ascontig_f32(pos)
    vel = ascontig_f32(vel)
    N = pos.shape[0]
    meta = {
        "N": int(N),
        "k": int(k),
        "chunk": int(chunk),
        "patch_space": "xyzv",
        "patch_vel_weight": float(PATCH_VEL_WEIGHT),
        "feature_names": list(XGB_FEATURE_NAMES),
    }

    # upload once
    t0 = now()
    pos_gpu = cp.asarray(pos)
    vel_gpu = cp.asarray(vel)
    features = cp.zeros((N, len(XGB_FEATURE_NAMES)), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    meta["gpu_upload_s"] = now() - t0

    # centre scalar features only, no exact position/velocity components
    t0 = now()
    r, vr, speed, vt, L, L_mag = recompute_scalar_phase_features(pos_gpu, vel_gpu)
    features[:, 0] = r
    features[:, 1] = vr
    features[:, 2] = speed
    features[:, 3] = vt
    features[:, 4] = L_mag
    cp.cuda.Stream.null.synchronize()
    meta["global_s"] = now() - t0

    # Build kNN in the same phase-space spirit as pointcloud: xyz + scaled velocity.
    # The resulting features are still scalar summaries; the raw coordinates are not
    # passed to XGBoost.
    t0 = now()
    patch_coords = np.hstack([pos, vel * np.float32(PATCH_VEL_WEIGHT)]).astype(np.float32, copy=False)
    index, idx_meta = build_faiss_index_adaptive(patch_coords)
    meta["faiss"] = idx_meta
    meta["index_s"] = now() - t0

    t_knn = 0.0
    t_local = 0.0

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        query = patch_coords[start:end]

        # batched search in xyzv patch-space
        t0 = now()
        D, I = faiss_search_batched(index, query, k=k, batch=FAISS_SEARCH_BATCH)
        t_knn += now() - t0

        # local patch features on GPU
        t0 = now()
        I_gpu = cp.asarray(I)
        pos_n = pos_gpu[I_gpu]                 # [B, K, 3]
        vel_n = vel_gpu[I_gpu]                 # [B, K, 3]

        centre_pos = pos_gpu[start:end][:, None, :]
        centre_vel = vel_gpu[start:end][:, None, :]
        centre_vr = vr[start:end]
        centre_speed = cp.maximum(speed[start:end], 1e-6)
        centre_lmag = cp.maximum(L_mag[start:end], 1e-6)

        rel_pos = pos_n - centre_pos
        dist = cp.linalg.norm(rel_pos, axis=2)
        mean_dist = cp.mean(dist, axis=1)
        max_dist = cp.max(dist, axis=1)

        # Match pointcloud sigma_v: std over all local velocity components.
        sigma_v = cp.std(vel_n, axis=(1, 2))

        r_n = cp.linalg.norm(pos_n, axis=2)
        speed_n = cp.linalg.norm(vel_n, axis=2)
        vr_n = cp.sum(pos_n * vel_n, axis=2) / cp.maximum(r_n, 1e-8)
        vt_n = cp.sqrt(cp.maximum(speed_n * speed_n - vr_n * vr_n, 0.0))
        sigma_vr = cp.std(vr_n, axis=1)

        # Position anisotropy using covariance eigenvalues, matching pointcloud logic.
        Xc = rel_pos - cp.mean(rel_pos, axis=1, keepdims=True)
        cov = cp.matmul(cp.swapaxes(Xc, 1, 2), Xc) / max(int(k), 1)
        evals = cp.linalg.eigvalsh(cov)
        evals = cp.sort(cp.maximum(evals, 1e-8), axis=1)
        anis_ratio = evals[:, -1] / evals[:, 0]
        flatness = evals[:, 1] / evals[:, -1]

        L_n = cp.cross(pos_n, vel_n)
        Lmag_n = cp.linalg.norm(L_n, axis=2)
        L_hat = L_n / cp.maximum(Lmag_n[..., None], 1e-6)
        L_align = cp.linalg.norm(cp.mean(L_hat, axis=1), axis=1)

        mean_speed_ratio = cp.mean(speed_n / centre_speed[:, None], axis=1)
        mean_abs_dvr = cp.mean(cp.abs(vr_n - centre_vr[:, None]), axis=1)
        mean_vt = cp.mean(vt_n, axis=1)
        lmag_ratio = cp.mean(Lmag_n, axis=1) / centre_lmag

        features[start:end, 5] = mean_dist
        features[start:end, 6] = max_dist
        features[start:end, 7] = sigma_v
        features[start:end, 8] = sigma_vr
        features[start:end, 9] = anis_ratio
        features[start:end, 10] = flatness
        features[start:end, 11] = L_align
        features[start:end, 12] = mean_speed_ratio
        features[start:end, 13] = mean_abs_dvr
        features[start:end, 14] = mean_vt
        features[start:end, 15] = lmag_ratio

        cp.cuda.Stream.null.synchronize()
        t_local += now() - t0

        del I_gpu, pos_n, vel_n, centre_pos, centre_vel, rel_pos, dist
        del r_n, speed_n, vr_n, vt_n, Xc, cov, evals, L_n, Lmag_n, L_hat
        gc.collect()

        print(f"[FEAT] {start:,}-{end:,} | knn={fmt(t_knn)} local={fmt(t_local)} idx={idx_meta.get('type')}")

    meta["knn_s"] = t_knn
    meta["local_s"] = t_local

    t0 = now()
    X_out = cp.asnumpy(features)
    cp.cuda.Stream.null.synchronize()
    meta["copyback_s"] = now() - t0
    meta["total_feature_s"] = now() - t_total
    return X_out, meta


# ============================================================
# Prior correction + selection
# ============================================================

def prior_correct_probs(p: np.ndarray, pi_train: float, pi_test: float, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    prior_odds_train = pi_train / (1 - pi_train)
    prior_odds_test  = pi_test / (1 - pi_test)
    odds_adj = odds * (prior_odds_test / prior_odds_train)
    return odds_adj / (1 + odds_adj)

def select_predictions(probs: np.ndarray,
                       use_top_fraction: bool,
                       top_fraction: float,
                       threshold: float = 0.5):
    if use_top_fraction:
        top_fraction = float(np.clip(top_fraction, 1e-6, 1.0))
        cut = np.quantile(probs, 1.0 - top_fraction)
        pred = probs >= cut
        return pred, {"mode": "top_fraction", "top_fraction": top_fraction, "cut": float(cut)}
    else:
        pred = probs >= float(threshold)
        return pred, {"mode": "threshold", "threshold": float(threshold)}

# ============================================================
# Loading
# ============================================================

def load_stream_npy(path: str):
    data = ascontig_f32(np.load(path))
    return data[:, :3], data[:, 3:6]

def load_background_npy(path: str):
    """
    Background .npy format:
    [x, y, z, vx, vy, vz, radii, vradii]
    Only x,y,z,vx,vy,vz are used here.
    """
    data = ascontig_f32(np.load(path))
    return data[:, :3], data[:, 3:6]

def load_background_hdf5(path: str, part_type: str = "PartType1"):
    with h5py.File(path, "r") as f:
        pos = np.array(f[part_type]["Coordinates"][:], dtype=np.float32, order="C")
        vel = np.array(f[part_type]["Velocities"][:], dtype=np.float32, order="C")
    return pos, vel

def load_background(path: str, part_type: str = "PartType1"):
    """
    Accept background files in either:
    - .hdf5  -> Gadget snapshot format
    - .npy   -> [x,y,z,vx,vy,vz,radii,vradii]
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".hdf5":
        return load_background_hdf5(path, part_type=part_type)
    elif ext == ".npy":
        return load_background_npy(path)
    else:
        raise ValueError(f"Unsupported background file format: {path}")

def load_test_snapshot(path: str, part_type: str = "PartType4", pos_scale: float = TEST_POS_SCALE):
    ext = os.path.splitext(path)[1].lower()

    # -----------------------------
    # HDF5 case
    # -----------------------------
    if ext in [".hdf5", ".h5"]:
        with h5py.File(path, "r") as f:
            pos = np.array(f[part_type]["Coordinates"][:], dtype=np.float32, order="C")
            _, idx = np.unique(pos, axis=0, return_index=True)
            idx = np.sort(idx)

            pos *= np.float32(pos_scale)
            vel = np.array(f[part_type]["Velocities"][:], dtype=np.float32, order="C")

            # Use unit masses when the optional mass dataset is absent.
            if "Mass" in f[part_type]:
                mass = np.array(f[part_type]["Mass"][:], dtype=np.float32, order="C")
            else:
                print("[WARN] Mass not found in dataset → using unit masses")
                mass = np.ones(len(pos), dtype=np.float32)
            if "ParticleIDs" in f[part_type]:
                ids = np.array(f[part_type]["ParticleIDs"][:], dtype=np.int64, order="C")
            else:
                ids = np.arange(len(pos), dtype=np.int64)

            pos = pos[idx]
            vel = vel[idx]
            mass = mass[idx]
            ids = ids[idx]

            centre = pos.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
            pos -= centre

        return pos, vel, mass, ids

    # -----------------------------
    # NPY case
    # -----------------------------
    elif ext == ".npy":
        arr = np.load(path).astype(np.float32)

        pos = arr[:, 0:3]
        vel = arr[:, 3:6]

        # NPY inputs do not contain mass, so use unit masses.
        mass = np.ones(len(pos), dtype=np.float32)
        ids = np.arange(len(pos), dtype=np.int64)

        centre = pos.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
        pos -= centre

        return pos, vel, mass, ids

    else:
        raise ValueError(f"Unsupported test snapshot format: {path}")


def save_feature_importance_outputs(model, feature_names, out_dir):
    """Save XGBoost feature importance using multiple built-in importance types."""
    os.makedirs(out_dir, exist_ok=True)
    booster = model.get_booster()
    importance_types = ["gain", "weight", "cover", "total_gain", "total_cover"]

    rows = []
    by_type = {}
    for imp_type in importance_types:
        raw = booster.get_score(importance_type=imp_type)
        vals = []
        for i, name in enumerate(feature_names):
            val = float(raw.get(f"f{i}", 0.0))
            vals.append(val)
        total = sum(vals)
        by_type[imp_type] = {name: vals[i] for i, name in enumerate(feature_names)}
        for i, name in enumerate(feature_names):
            row = next((r for r in rows if r["feature"] == name), None)
            if row is None:
                row = {"feature_index": int(i), "feature": name}
                rows.append(row)
            row[imp_type] = vals[i]
            row[f"{imp_type}_norm"] = vals[i] / total if total > 0 else 0.0

    # Sort by normalized gain first because it is usually the most interpretable.
    rows = sorted(rows, key=lambda r: r.get("gain_norm", 0.0), reverse=True)

    csv_path = os.path.join(out_dir, "xgboost_feature_importance.csv")
    import csv
    fieldnames = ["feature_index", "feature"]
    for imp_type in importance_types:
        fieldnames.extend([imp_type, f"{imp_type}_norm"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = os.path.join(out_dir, "xgboost_feature_importance.json")
    with open(json_path, "w") as f:
        json.dump({
            "feature_names": list(feature_names),
            "importance_by_type": by_type,
            "ranked_by_gain": rows,
            "notes": "gain/total_gain are usually the best first diagnostic; weight counts split frequency and can overvalue high-cardinality features.",
        }, f, indent=2)

    print(f"[TRAIN] saved feature importance CSV: {csv_path}")
    print(f"[TRAIN] saved feature importance JSON: {json_path}")
    return rows


# ============================================================
# Train
# ============================================================


def _iter_data_files(data_dir, suffixes=(".npy", ".hdf5", ".h5")):
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Missing data directory: {data_dir}")
    return [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(suffixes)]


def build_feature_set(stream_dir: str, back_dir: str, chunk: int, split_name: str):
    X_list, y_list = [], []

    stream_files = _iter_data_files(stream_dir, suffixes=(".npy",))
    back_files = _iter_data_files(back_dir, suffixes=(".npy", ".hdf5", ".h5"))

    if len(stream_files) == 0:
        raise RuntimeError(f"No stream files found in {stream_dir}")
    if len(back_files) == 0:
        raise RuntimeError(f"No background files found in {back_dir}")

    print(f"[{split_name}] stream files: {len(stream_files)}")
    print(f"[{split_name}] background files: {len(back_files)}")

    for path in stream_files:
        print(f"[{split_name}] stream {os.path.basename(path)}")
        pos, vel = load_stream_npy(path)
        X, meta = compute_features_gpu(pos, vel, k=K, chunk=chunk)
        print(f"  feature={fmt(meta['total_feature_s'])} knn={fmt(meta['knn_s'])} faiss={meta['faiss']}")
        X_list.append(X)
        y_list.append(np.ones(len(X), dtype=np.float32))

    for path in back_files:
        print(f"[{split_name}] background {os.path.basename(path)}")
        pos, vel = load_background(path, part_type="PartType1")
        X, meta = compute_features_gpu(pos, vel, k=K, chunk=chunk)
        print(f"  feature={fmt(meta['total_feature_s'])} knn={fmt(meta['knn_s'])} faiss={meta['faiss']}")
        X_list.append(X)
        y_list.append(np.zeros(len(X), dtype=np.float32))

    X_all = np.vstack(X_list).astype(np.float32, copy=False)
    y_all = np.concatenate(y_list).astype(np.float32, copy=False)
    print(f"[{split_name}] X={X_all.shape} stream_frac={y_all.mean():.4f}")
    return X_all, y_all, stream_files, back_files


def train_model(model_path: str, meta_path: str, chunk: int = DEFAULT_CHUNK):
    t_total = now()

    X_train, y_train, train_stream_files, train_back_files = build_feature_set(
        TRAIN_STREAM_DIR, TRAIN_BACK_DIR, chunk, "TRAIN"
    )
    X_val, y_val, val_stream_files, val_back_files = build_feature_set(
        VAL_STREAM_DIR, VAL_BACK_DIR, chunk, "VAL"
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        scale_pos_weight=1.0,
        eval_metric=["logloss", "aucpr", "auc"],
        early_stopping_rounds=25,
        random_state=RANDOM_SEED,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    log_path = os.path.join(os.path.dirname(model_path), XGB_LOG_FILENAME)
    hist_path = os.path.join(os.path.dirname(model_path), XGB_EVAL_HISTORY)
    for pth in [log_path, hist_path]:
        if os.path.exists(pth):
            os.remove(pth)
    t0 = now()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    fit_time = now() - t0
    print(f"[TRAIN] fit_time={fmt(fit_time)}")

    # Write eval history after fitting. This avoids the older XGBoost sklearn API
    # error: XGBClassifier.fit() got an unexpected keyword argument 'callbacks'.
    try:
        evals = model.evals_result()
        hist_logger = CSVLogger(hist_path)
        n_rounds = 0
        for _dataset_name, _metrics in evals.items():
            for _metric_name, _values in _metrics.items():
                n_rounds = max(n_rounds, len(_values))
        for r in range(n_rounds):
            row = {"round": int(r)}
            for dataset_name, metrics in evals.items():
                for metric_name, values in metrics.items():
                    if r < len(values):
                        row[f"{dataset_name}_{metric_name}"] = float(values[r])
            hist_logger.write(row)
    except Exception as e:
        print(f"[WARN] Could not write XGBoost eval history: {e}")

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]

    train_metrics = compute_binary_metrics_from_probs(y_train, train_probs, threshold=0.5)
    val_metrics = compute_binary_metrics_from_probs(y_val, val_probs, threshold=0.5)

    print("[TRAIN] final train metrics:", train_metrics)
    print("[TRAIN] final val metrics:", val_metrics)

    model.save_model(model_path)
    feature_importance_rows = save_feature_importance_outputs(model, XGB_FEATURE_NAMES, os.path.dirname(model_path))

    logger = CSVLogger(log_path)
    logger.write({
        "fit_seconds": float(fit_time),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "train_loss_final": float(model.evals_result()["validation_0"]["logloss"][-1]),
        "val_loss_final": float(model.evals_result()["validation_1"]["logloss"][-1]),
        "train_acc": float(train_metrics["acc"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_specificity": float(train_metrics["specificity"]),
        "train_f1": float(train_metrics["f1"]),
        "train_iou": float(train_metrics["iou"]),
        "train_dice": float(train_metrics["dice"]),
        "train_mcc": float(train_metrics["mcc"]),
        "train_balanced_acc": float(train_metrics["balanced_acc"]),
        "train_auc": float(train_metrics["auc"]),
        "train_auprc": float(train_metrics["auprc"]),
        "val_acc": float(val_metrics["acc"]),
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_specificity": float(val_metrics["specificity"]),
        "val_f1": float(val_metrics["f1"]),
        "val_iou": float(val_metrics["iou"]),
        "val_dice": float(val_metrics["dice"]),
        "val_mcc": float(val_metrics["mcc"]),
        "val_balanced_acc": float(val_metrics["balanced_acc"]),
        "val_auc": float(val_metrics["auc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "best_score": float(getattr(model, "best_score", np.nan)),
    })

    meta = {
        "K": K,
        "patch_space": "xyzv",
        "patch_vel_weight": float(PATCH_VEL_WEIGHT),
        "chunk": int(chunk),
        "test_pos_scale": TEST_POS_SCALE,
        "train_stream_prior": TRAIN_STREAM_PRIOR,
        "default_test_stream_prior": TEST_STREAM_PRIOR_DEFAULT,
        "feature_order": list(XGB_FEATURE_NAMES),
        "rich_output_files": ["*_pos.npy", "*_vel.npy", "*_r_vr.npy", "*_mass.npy", "*_ids.npy", "*_prob_stream.npy", "*_logit_stream.npy", "*_margin.npy", "*_pred_label.npy", "*_xgb_features.npy"],
        "training_data_dirs": {
            "train_stream": TRAIN_STREAM_DIR,
            "train_background": TRAIN_BACK_DIR,
            "val_stream": VAL_STREAM_DIR,
            "val_background": VAL_BACK_DIR,
        },
        "training_files": {
            "train_stream": [os.path.basename(p) for p in train_stream_files],
            "train_background": [os.path.basename(p) for p in train_back_files],
            "val_stream": [os.path.basename(p) for p in val_stream_files],
            "val_background": [os.path.basename(p) for p in val_back_files],
        },
        "training_log": XGB_LOG_FILENAME,
        "eval_history_log": XGB_EVAL_HISTORY,
        "feature_importance_csv": "xgboost_feature_importance.csv",
        "feature_importance_json": "xgboost_feature_importance.json",
        "top_features_by_gain": feature_importance_rows[:10],
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "best_score": float(getattr(model, "best_score", np.nan)),
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[TRAIN] saved model+meta. total={fmt(now() - t_total)}")

# ============================================================
# Infer
# ============================================================

def run_inference(model_path: str,
                  meta_path: str,
                  output_dir: str,
                  snapshot_path: str,
                  out_prefix: str,
                  chunk: int,
                  pi_test: float,
                  use_prior_correction: bool,
                  use_top_fraction: bool,
                  top_fraction: float,
                  threshold: float):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run `train` first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run `train` first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cuda", tree_method="hist", predictor="gpu_predictor")

    t0 = now()
    pos, vel, mass, ids = load_test_snapshot(snapshot_path, part_type="PartType4", pos_scale=TEST_POS_SCALE)
    print(f"[INFER] load_time={fmt(now()-t0)} N={len(pos):,}")

    t0 = now()
    X_test, feat_meta = compute_features_gpu(pos, vel, k=K, chunk=chunk)
    print(f"[INFER] feature_time={fmt(now()-t0)} knn={fmt(feat_meta['knn_s'])} faiss={feat_meta['faiss']}")

    t0 = now()
    probs = model.get_booster().inplace_predict(X_test)
    print(f"[INFER] predict_time={fmt(now()-t0)}")
    print("[INFER] probs percentiles:", np.percentile(probs, [50, 80, 90, 95, 99, 99.5, 99.9]))

    if use_prior_correction:
        probs = prior_correct_probs(probs, pi_train=TRAIN_STREAM_PRIOR, pi_test=pi_test)
        print("[INFER] probs_adj percentiles:", np.percentile(probs, [50, 80, 90, 95, 99, 99.5, 99.9]))

    raw_margin = model.get_booster().inplace_predict(X_test, predict_type="margin")
    pred, sel_meta = select_predictions(
        probs,
        use_top_fraction=use_top_fraction,
        top_fraction=top_fraction,
        threshold=threshold
    )
    print(f"[INFER] selection={sel_meta} stream_frac={pred.mean():.4f}")

    os.makedirs(output_dir, exist_ok=True)

    r = np.linalg.norm(pos, axis=1).astype(np.float32)
    vr = (np.sum(vel * pos, axis=1) / np.maximum(r, 1e-8)).astype(np.float32)
    margins = np.asarray(raw_margin, dtype=np.float32)
    logits_stream = margins.astype(np.float32)

    np.save(os.path.join(output_dir, f"{out_prefix}_pos.npy"), pos.astype(np.float32))
    np.save(os.path.join(output_dir, f"{out_prefix}_vel.npy"), vel.astype(np.float32))
    np.save(os.path.join(output_dir, f"{out_prefix}_r_vr.npy"), np.column_stack([r, vr]).astype(np.float32))
    np.save(os.path.join(output_dir, f"{out_prefix}_mass.npy"), mass.astype(np.float32))
    np.save(os.path.join(output_dir, f"{out_prefix}_ids.npy"), ids.astype(np.int64))

    np.save(os.path.join(output_dir, f"{out_prefix}_prob_stream.npy"), np.asarray(probs, dtype=np.float32))
    np.save(os.path.join(output_dir, f"{out_prefix}_logit_stream.npy"), logits_stream)
    np.save(os.path.join(output_dir, f"{out_prefix}_margin.npy"), margins)
    np.save(os.path.join(output_dir, f"{out_prefix}_pred_label.npy"), pred.astype(np.int8))
    np.save(os.path.join(output_dir, f"{out_prefix}_xgb_features.npy"), X_test.astype(np.float32))

    with open(os.path.join(output_dir, f"{out_prefix}_feature_names.json"), "w") as f:
        json.dump({"feature_order": list(XGB_FEATURE_NAMES)}, f, indent=2)

    print(f"[INFER] saved rich outputs with prefix: {out_prefix}")

# ============================================================
# Main (argparse)
# ============================================================

def add_path_args(p):
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--test_dir", type=str, default=TEST_DIR)

    p.add_argument("--train_stream_dir", type=str, default=TRAIN_STREAM_DIR)
    p.add_argument("--val_stream_dir", type=str, default=VAL_STREAM_DIR)
    p.add_argument("--train_back_dir", type=str, default=TRAIN_BACK_DIR)
    p.add_argument("--val_back_dir", type=str, default=VAL_BACK_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Stream Classifier (FAISS + XGBoost)"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------
    p_train = subparsers.add_parser("train", help="Train and save model")

    p_train.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help="Feature chunk size"
    )

    add_path_args(p_train)

    # ------------------------------------------------------------
    # INFER
    # ------------------------------------------------------------
    p_infer = subparsers.add_parser("infer", help="Run inference on a test snapshot")

    p_infer.add_argument(
        "snapshot",
        type=str,
        help="Test snapshot filename in TEST_DIR or full path (.hdf5/.h5/.npy)"
    )

    p_infer.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help="Feature chunk size"
    )

    # Pointcloud-style aliases / explicit checkpoint paths for standardized inference.
    # These do not change the trained model format; they only let inference load a
    # model+metadata from a different directory than --out_dir.
    p_infer.add_argument(
        "--infer_chunk_size",
        type=int,
        default=None,
        help="Alias for --chunk, matching the pointcloud inference interface"
    )

    p_infer.add_argument(
        "--ckpt_path",
        "--model_path",
        dest="ckpt_path",
        type=str,
        default="",
        help="Explicit path to stream_classifier.json. If omitted, uses --out_dir/stream_classifier.json"
    )

    p_infer.add_argument(
        "--meta_path",
        type=str,
        default="",
        help="Explicit path to feature_metadata.json. If omitted, uses --out_dir/feature_metadata.json"
    )

    p_infer.add_argument(
        "--pi_test",
        type=float,
        default=TEST_STREAM_PRIOR_DEFAULT,
        help="Assumed stream fraction in test snapshot"
    )

    p_infer.add_argument(
        "--top_frac",
        type=float,
        default=TOP_FRACTION_DEFAULT,
        help="Top fraction selected as stream"
    )

    p_infer.add_argument(
        "--no_prior_correction",
        action="store_true",
        help="Disable prior correction"
    )

    p_infer.add_argument(
        "--no_top_fraction",
        action="store_true",
        help="Disable top-fraction selection and use threshold instead"
    )

    p_infer.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used if --no_top_fraction is set"
    )

    add_path_args(p_infer)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Apply directory arguments globally
    # ------------------------------------------------------------
    globals()["OUT_DIR"] = args.out_dir
    globals()["TEST_DIR"] = args.test_dir

    globals()["TRAIN_STREAM_DIR"] = args.train_stream_dir
    globals()["VAL_STREAM_DIR"] = args.val_stream_dir
    globals()["TRAIN_BACK_DIR"] = args.train_back_dir
    globals()["VAL_BACK_DIR"] = args.val_back_dir

    out_dir = args.out_dir
    model_path = os.path.join(out_dir, MODEL_FILENAME)
    meta_path = os.path.join(out_dir, META_FILENAME)
    output_dir = out_dir

    # For inference, allow model/metadata to live somewhere other than --out_dir.
    # This keeps existing trained models compatible and lets --out_dir be only the
    # destination for inference products.
    if args.mode == "infer":
        if getattr(args, "ckpt_path", ""):
            model_path = args.ckpt_path
        if getattr(args, "meta_path", ""):
            meta_path = args.meta_path
        if getattr(args, "infer_chunk_size", None) is not None:
            args.chunk = int(args.infer_chunk_size)

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------
    if args.mode == "train":
        train_model(
            model_path=model_path,
            meta_path=meta_path,
            chunk=args.chunk
        )
        return

    # ------------------------------------------------------------
    # INFER
    # ------------------------------------------------------------
    if args.mode == "infer":
        snapshot_path = resolve_test_dataset_path(
            args.snapshot,
            test_dir=args.test_dir
        )

        if not (0.0 < args.top_frac <= 1.0):
            raise ValueError(f"--top_frac must be in (0,1], got {args.top_frac}")

        if not (0.0 < args.pi_test < 1.0):
            raise ValueError(f"--pi_test must be in (0,1), got {args.pi_test}")

        if not (0.0 < args.threshold < 1.0):
            raise ValueError(f"--threshold must be in (0,1), got {args.threshold}")

        out_prefix = base_stem(snapshot_path)

        run_inference(
            model_path=model_path,
            meta_path=meta_path,
            output_dir=output_dir,
            snapshot_path=snapshot_path,
            out_prefix=out_prefix,
            chunk=args.chunk,
            pi_test=args.pi_test,
            use_prior_correction=(not args.no_prior_correction),
            use_top_fraction=(not args.no_top_fraction),
            top_fraction=args.top_frac,
            threshold=args.threshold,
        )
        return

if __name__ == "__main__":
    main()
