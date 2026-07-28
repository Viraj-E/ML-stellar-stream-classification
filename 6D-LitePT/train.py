
"""Training entry point for 6D-LitePT.

Usage:
  python train.py train [options]
"""

import argparse
import json
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR

import config as C
import config as _config_module
from data import (
    autocast_context,
    build_split_candidate_arrays,
    build_train_val_candidate_arrays,
    compute_feature_stats,
    configure_training_validation_root,
    ensure_dir,
    fmt,
    geometry_report,
    make_grad_scaler,
    now,
    get_model_feature_names,
    resolve_feature_set,
    save_json,
    stats_to_jsonable,
    xlogger,
    FEATURE_GROUPS,
)
from dataset import LitePTChunkDataset, build_dataloader
from losses import (
    BatchBalancedCrossEntropyPointLoss,
    CrossEntropyPointLoss,
    FTNMTPointLoss,
    HybridCEFTNMTPointLoss,
)
from network import LitePTBinarySeg


# ----------------------------- optimizer/scheduler ---------------------------
def build_optimizer(model, lr, weight_decay, name="adam"):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer must be adam, adamw, or radam")


def build_scheduler(optimizer, schedule, warmup_epochs, total_epochs, cosine_t0, cosine_t_mult):
    schedule = schedule.lower()
    if schedule == "none":
        return None
    if warmup_epochs < 1:
        raise ValueError("warmup_epochs must be >= 1")
    warm = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    if schedule == "warmup":
        tail = ConstantLR(optimizer, factor=1.0, total_iters=max(1, total_epochs - warmup_epochs))
    elif schedule == "warmup_cosine":
        tail = CosineAnnealingWarmRestarts(optimizer, T_0=cosine_t0, T_mult=cosine_t_mult)
    else:
        raise ValueError("schedule must be none, warmup, or warmup_cosine")
    return SequentialLR(optimizer, schedulers=[warm, tail], milestones=[warmup_epochs])


def build_loss(name):
    name = str(name).lower()
    if name == "ftnmt":
        return FTNMTPointLoss(depth=C.FTNMT_DEPTH, num_classes=2)
    if name == "ce":
        return CrossEntropyPointLoss()
    if name in ("weighted_ce", "balanced_ce"):
        return BatchBalancedCrossEntropyPointLoss()
    if name in ("hybrid", "hybrid_ce_ftnmt", "ce_ftnmt"):
        return HybridCEFTNMTPointLoss(
            depth=C.FTNMT_DEPTH,
            num_classes=2,
            ce_weight=float(getattr(C, "HYBRID_CE_WEIGHT", 0.5)),
            ftnmt_weight=float(getattr(C, "HYBRID_FTNMT_WEIGHT", 0.5)),
        )
    raise ValueError("loss_name must be ftnmt, ce, weighted_ce, or hybrid_ce_ftnmt")


# ----------------------------- metrics ---------------------------------------
def _stats_from_counts(tp, tn, fp, fn):
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    n = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    acc = (tp + tn) / n
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    miou = 0.5 * (iou + bg_iou)
    dice = f1
    denom = float(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_acc": float(0.5 * (recall + specificity)),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "bg_iou": float(bg_iou),
        "miou": float(miou),
        "mcc": float(mcc),
        "true_pos_frac": float((tp + fn) / n),
        "pred_pos_frac": float((tp + fp) / n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class ThresholdAccumulator:
    def __init__(self, thresholds):
        self.thresholds = tuple(float(t) for t in thresholds)
        self._threshold_tensor = None
        self._tp = self._tn = self._fp = self._fn = None

    def _ensure_device(self, device):
        if self._threshold_tensor is not None and self._threshold_tensor.device == device:
            return
        self._threshold_tensor = torch.tensor(self.thresholds, dtype=torch.float32, device=device)
        shape = (len(self.thresholds),)
        self._tp = torch.zeros(shape, dtype=torch.long, device=device)
        self._tn = torch.zeros(shape, dtype=torch.long, device=device)
        self._fp = torch.zeros(shape, dtype=torch.long, device=device)
        self._fn = torch.zeros(shape, dtype=torch.long, device=device)

    def update(self, probs, target):
        probs = probs.detach().float()
        target = target.detach().bool()
        self._ensure_device(probs.device)
        pred = probs.unsqueeze(0) >= self._threshold_tensor[:, None]
        target = target.unsqueeze(0)
        self._tp += (pred & target).sum(dim=1)
        self._tn += ((~pred) & (~target)).sum(dim=1)
        self._fp += (pred & (~target)).sum(dim=1)
        self._fn += ((~pred) & target).sum(dim=1)

    def compute(self):
        if self._tp is None:
            return {float(t): _stats_from_counts(0, 0, 0, 0) for t in self.thresholds}
        tp = self._tp.detach().cpu().tolist()
        tn = self._tn.detach().cpu().tolist()
        fp = self._fp.detach().cpu().tolist()
        fn = self._fn.detach().cpu().tolist()
        return {
            float(t): _stats_from_counts(tp[i], tn[i], fp[i], fn[i])
            for i, t in enumerate(self.thresholds)
        }


def _move_batch_to_device(batch):
    return {k: v.to(C.DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _iter_device_batches(loader):
    if not (C.DEVICE == "cuda" and torch.cuda.is_available() and bool(getattr(C, "CUDA_PREFETCH", False))):
        for batch in loader:
            yield _move_batch_to_device(batch)
        return

    stream = torch.cuda.Stream()
    iterator = iter(loader)
    next_batch = None

    def preload():
        nonlocal next_batch
        try:
            batch = next(iterator)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = _move_batch_to_device(batch)

    preload()
    while next_batch is not None:
        torch.cuda.current_stream().wait_stream(stream)
        batch = next_batch
        for value in batch.values():
            if torch.is_tensor(value):
                value.record_stream(torch.cuda.current_stream())
        preload()
        yield batch


def run_epoch(model, loader, optimizer, scaler, criterion, train_mode=True, thresholds=C.EVAL_THRESHOLDS, phase="train", progress_every=5):
    model.train() if train_mode else model.eval()
    total_loss, n_batches = None, 0
    epoch_t0 = now()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    metrics = ThresholdAccumulator(thresholds)
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    phase_label = str(phase)
    for batch_idx, batch in enumerate(_iter_device_batches(loader), start=1):
        amp_ctx = autocast_context() if train_mode else nullcontext()
        with torch.set_grad_enabled(train_mode):
            with amp_ctx:
                logits, _ = model(batch, return_dense=False)
                loss = criterion(logits, batch["segment"])
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        metrics.update(torch.softmax(logits.detach(), dim=1)[:, 1], batch["segment"].detach())
        loss_detached = loss.detach().float()
        total_loss = loss_detached if total_loss is None else total_loss + loss_detached
        n_batches += 1
        if progress_every and (batch_idx % progress_every == 0 or (total_batches is not None and batch_idx == total_batches)):
            avg_loss = float((total_loss / max(n_batches, 1)).item()) if total_loss is not None else 0.0
            suffix = f"/{total_batches}" if total_batches is not None else ""
            gpu = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_reserved() / 1024**3
                gpu = f" gpu_alloc={alloc:.2f}GiB gpu_reserved={reserved:.2f}GiB gpu_peak_reserved={peak:.2f}GiB"
            print(f"{phase_label}: batch {batch_idx}{suffix} avg_loss={avg_loss:.4f} elapsed={now() - epoch_t0:.1f}s{gpu}", flush=True)
    by_thr = metrics.compute()
    main = dict(by_thr.get(0.5, by_thr[float(thresholds[0])]))
    main["loss"] = float((total_loss / max(n_batches, 1)).item()) if total_loss is not None else 0.0
    main["by_threshold"] = by_thr
    return main


def _flatten_epoch_row(prefix, stats, thresholds):
    row = {f"{prefix}_loss": stats["loss"]}
    s05 = stats["by_threshold"].get(0.5, stats["by_threshold"][float(thresholds[0])])
    for k in (
        "acc", "precision", "recall", "specificity", "balanced_acc", "f1", "dice", "iou", "bg_iou", "miou", "mcc",
        "true_pos_frac", "pred_pos_frac", "tp", "tn", "fp", "fn",
    ):
        row[f"{prefix}_{k}"] = s05[k]
    for t in thresholds:
        key = str(float(t)).replace(".", "p")
        s = stats["by_threshold"][float(t)]
        for k in (
            "acc", "precision", "recall", "specificity", "balanced_acc", "f1", "dice", "iou", "mcc",
            "true_pos_frac", "pred_pos_frac", "tp", "tn", "fp", "fn",
        ):
            row[f"{prefix}_thr{key}_{k}"] = s[k]
    return row


def _geometry_gate(X_stream_tr, X_back_tr):
    stream_geo = geometry_report("train_stream", X_stream_tr)
    back_geo = geometry_report("train_background", X_back_tr)
    stream_nonspherical = stream_geo["axis_ratio"] >= float(C.STREAM_NONSPHERICAL_AXIS_RATIO_MIN)
    background_spherical = back_geo["axis_ratio"] <= float(C.BACKGROUND_SPHERICAL_AXIS_RATIO_MAX)
    enable_relative = bool(C.SYNTH_RELATIVE_STREAM_ROTATION and stream_nonspherical and background_spherical)
    print("Geometry check for relative stream rotations:")
    print(f"  stream axis_ratio={stream_geo['axis_ratio']:.4f} non_spherical={stream_nonspherical}")
    print(f"  background axis_ratio={back_geo['axis_ratio']:.4f} spherical_enough={background_spherical}")
    print(f"  relative stream rotation enabled={enable_relative}")
    return enable_relative, {"stream": stream_geo, "background": back_geo, "relative_stream_rotation_enabled": enable_relative}


def evaluate_split(name, args, ckpt_path, stats, in_channels, thresholds, max_stream, max_back):
    print("\n" + "=" * 60)
    print(f"{name.upper()} EVALUATION: {ckpt_path}")
    print("=" * 60)
    X_stream, X_back = build_split_candidate_arrays(
        name,
        max_stream=max_stream,
        max_back=max_back,
        stream_sampling_mode=args.stream_sampling_mode,
        background_sampling_mode=args.background_sampling_mode,
        full_background_small_files=args.full_background_small_files,
        background_large_file_count=args.background_large_file_count,
        large_background_file_names=args.large_background_file_names,
    )
    n_chunks = args.test_chunks if name in ("test", "testing") else args.val_chunks
    ds = LitePTChunkDataset(
        X_stream, X_back, stats, args.chunk_size, args.grid_size,
        n_chunks, args.min_particles_per_chunk, args.n_valid_centres,
        args.feature_set, train_mode=False,
    )
    eval_batch_size = int(getattr(args, "eval_batch_size", 0)) or int(args.batch_size)
    dl = build_dataloader(ds, eval_batch_size, False, False, 0)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LitePTBinarySeg(in_channels=in_channels).to(C.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    criterion = build_loss(args.loss_name)
    model.eval()
    out = run_epoch(model, dl, optimizer=None, scaler=None, criterion=criterion, train_mode=False, thresholds=thresholds, progress_every=args.progress_every)
    return out


# ----------------------------- training --------------------------------------
def train_main(args):
    if int(args.batch_size) < 2:
        raise ValueError("Batch size must be at least 2 because LitePT uses BatchNorm1d")
    if int(args.batch_size) > 64:
        raise ValueError("Batch size must be no larger than 64 for this run")
    if int(getattr(args, "eval_batch_size", 0)) < 0:
        raise ValueError("Eval batch size must be non-negative; use 0 to mirror batch_size")
    C.PREFETCH_FACTOR = max(1, int(args.prefetch_factor))
    C.PIN_MEMORY = bool(int(args.pin_memory))
    C.CUDA_PREFETCH = bool(int(args.cuda_prefetch))
    C.CACHE_MATERIALIZED_SAMPLES = bool(int(args.materialized_cache))
    C.MATERIALIZED_CACHE_MAX_ITEMS = int(args.materialized_cache_max_items)
    C.CACHE_SPATIAL_QUERIES = bool(int(args.cache_spatial_queries))
    C.SPATIAL_QUERY_CACHE_MAX_CENTRES = int(args.spatial_query_cache_max_centres)
    C.MAX_RAW_PARTICLES_PER_CHUNK = int(args.max_raw_particles_per_chunk)
    C.TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK = int(args.train_max_native_voxels_per_chunk)
    C.EVAL_MAX_NATIVE_VOXELS_PER_CHUNK = int(args.eval_max_native_voxels_per_chunk)
    C.MAX_NATIVE_VOXELS_PER_CHUNK = C.TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK
    C.ACTIVATION_CHECKPOINTING = bool(int(args.activation_checkpointing))
    C.RAW_SUBSAMPLE_MODE = str(args.raw_subsample_mode)
    C.RAW_SUBSAMPLE_TARGET_STREAM_FRAC = float(args.raw_subsample_target_stream_frac)
    C.LABEL_REDUCE = str(args.label_reduce)
    C.MODEL_PRESET = str(args.model_preset)
    C.AUGMENT_TRAIN = bool(int(args.augment_train))
    C.AUGMENT_UNIT_SCALE = bool(int(args.augment_unit_scale))
    C.AUGMENT_ROTATION = bool(int(args.augment_rotation))
    C.SYNTH_RELATIVE_STREAM_ROTATION = bool(int(args.relative_stream_rotation))
    C.HYBRID_CE_WEIGHT = float(args.hybrid_ce_weight)
    C.HYBRID_FTNMT_WEIGHT = float(args.hybrid_ftnmt_weight)
    ensure_dir(args.out_dir)
    config_src = _config_module.__file__
    if config_src.endswith(".pyc"):
        config_src = config_src[:-1]
    shutil.copy2(config_src, os.path.join(args.out_dir, "litept_config_snapshot.py"))
    log = xlogger(os.path.join(args.out_dir, C.LOG_NAME))

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:  ", C.DEVICE)
    thresholds = tuple(float(x) for x in args.eval_thresholds.split(","))
    if 0.5 not in thresholds:
        thresholds = (0.5,) + thresholds
    print("Monitoring threshold: 0.5")
    print("Logged threshold sweep:", thresholds)

    t0 = now()
    configure_training_validation_root(args.tv_root)
    X_stream_tr, X_back_tr, X_stream_va, X_back_va = build_train_val_candidate_arrays(
        max_stream_train=args.train_stream_ref_cap,
        max_back_train=args.train_back_ref_cap,
        max_stream_val=args.val_stream_ref_cap,
        max_back_val=args.val_back_ref_cap,
        stream_sampling_mode=args.stream_sampling_mode,
        background_sampling_mode=args.background_sampling_mode,
        full_background_small_files=args.full_background_small_files,
        background_large_file_count=args.background_large_file_count,
        large_background_file_names=args.large_background_file_names,
    )
    print(
        f"Built candidate pools in {fmt(now() - t0)} | "
        f"train stream={len(X_stream_tr):,} back={len(X_back_tr):,} | "
        f"val stream={len(X_stream_va):,} back={len(X_back_va):,}"
    )
    print(f"TV root: {args.tv_root}")

    enable_relative, geometry = _geometry_gate(X_stream_tr, X_back_tr)
    stats = compute_feature_stats(X_stream_tr, X_back_tr, feature_set=args.feature_set)
    feat_names = get_model_feature_names(args.feature_set)
    in_channels = len(feat_names)
    print(f"Feature set: {args.feature_set!r} -> {in_channels} channels: {feat_names}")
    print(f"Dimensionless inputs: {C.DIMENSIONLESS_INPUTS}")
    print(f"Activation checkpointing: {C.ACTIVATION_CHECKPOINTING}")
    eval_cap_desc = "uncapped" if C.EVAL_MAX_NATIVE_VOXELS_PER_CHUNK <= 0 else str(C.EVAL_MAX_NATIVE_VOXELS_PER_CHUNK)
    print(f"Native voxel caps: train={C.TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK} eval={eval_cap_desc}")
    print(f"Loss: {args.loss_name}, depth={C.FTNMT_DEPTH}")

    train_ds = LitePTChunkDataset(
        X_stream_tr, X_back_tr, stats, args.chunk_size, args.grid_size,
        args.train_chunks_per_epoch, args.min_particles_per_chunk,
        args.n_valid_centres, args.feature_set, train_mode=True,
        enable_relative_stream_rotation=enable_relative,
    )
    val_ds = LitePTChunkDataset(
        X_stream_va, X_back_va, stats, args.chunk_size, args.grid_size,
        args.val_chunks, args.min_particles_per_chunk,
        args.n_valid_centres, args.feature_set, train_mode=False,
    )
    train_shuffle = not bool(getattr(C, "CACHE_MATERIALIZED_SAMPLES", False))
    train_dl = build_dataloader(train_ds, args.batch_size, train_shuffle, True, args.num_workers)
    eval_batch_size = int(args.eval_batch_size) or int(args.batch_size)
    print(f"Batch sizes: train={args.batch_size} eval={eval_batch_size}")
    val_dl = build_dataloader(val_ds, eval_batch_size, False, False, args.num_workers)

    model = LitePTBinarySeg(in_channels=in_channels).to(C.DEVICE)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.optimizer)
    scheduler = build_scheduler(optimizer, args.lr_schedule, args.warmup_epochs, args.epochs, args.cosine_t0, args.cosine_t_mult)
    scaler = make_grad_scaler()
    criterion = build_loss(args.loss_name)

    best_mcc = -1.0
    best_loss = float("inf")
    best_mcc_path = os.path.join(args.out_dir, C.CKPT_NAME)
    best_loss_path = os.path.join(args.out_dir, C.CKPT_BEST_LOSS_NAME)
    meta_path = os.path.join(args.out_dir, C.META_NAME)
    epoch_dir = os.path.join(args.out_dir, C.CKPT_EPOCH_DIR)
    ensure_dir(epoch_dir)

    start_epoch = 0
    if args.resume_from:
        resume_path = os.path.abspath(args.resume_from)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume_ckpt["model_state"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if scheduler is not None:
            if resume_ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            else:
                for _ in range(int(resume_ckpt.get("epoch", 0))):
                    scheduler.step()
        if scaler is not None and resume_ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_mcc = float(resume_ckpt.get("best_val_mcc", resume_ckpt.get("val_mcc", best_mcc)))
        best_loss = float(resume_ckpt.get("best_val_loss", resume_ckpt.get("val_loss", best_loss)))
        print(
            f"Resumed from {resume_path} at completed epoch {start_epoch}; "
            f"best_mcc={best_mcc:.4f} best_loss={best_loss:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        print("\n" + "=" * 48 + f"\nEpoch {epoch + 1:03d}/{args.epochs}\n" + "=" * 48)
        epoch_t0 = now()
        current_lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_dl, optimizer, scaler, criterion, train_mode=True, thresholds=thresholds, phase=f"train epoch {epoch + 1:03d}", progress_every=args.progress_every)
        va = run_epoch(model, val_dl, optimizer, scaler, criterion, train_mode=False, thresholds=thresholds, phase=f"val epoch {epoch + 1:03d}", progress_every=args.progress_every)
        if scheduler is not None:
            scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        row = dict(
            epoch=epoch + 1,
            lr=current_lr,
            next_lr=next_lr,
            feature_set=args.feature_set,
            model_preset=C.MODEL_PRESET,
            loss_name=args.loss_name,
            label_reduce=C.LABEL_REDUCE,
            dimensionless_inputs=int(C.DIMENSIONLESS_INPUTS),
            augment_train=int(C.AUGMENT_TRAIN),
            relative_stream_rotation=int(enable_relative),
            tv_root=args.tv_root,
            stream_sampling_mode=args.stream_sampling_mode,
            background_sampling_mode=args.background_sampling_mode,
            full_background_small_files=int(args.full_background_small_files),
            background_large_file_count=args.background_large_file_count,
        )
        row.update(_flatten_epoch_row("train", tr, thresholds))
        row.update(_flatten_epoch_row("val", va, thresholds))
        row["seconds"] = now() - epoch_t0
        log.write(row)

        print(f"LR   : {current_lr:.2e}" + (f" -> {next_lr:.2e}" if scheduler is not None else ""))
        print(
            f"Train@0.5: loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1={tr['f1']:.4f} "
            f"IoU={tr['iou']:.4f} MCC={tr['mcc']:.4f} true_pos_frac={tr['true_pos_frac']:.4f} "
            f"pred_pos_frac={tr['pred_pos_frac']:.4f}"
        )
        print(
            f"Val@0.5  : loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['f1']:.4f} "
            f"IoU={va['iou']:.4f} MCC={va['mcc']:.4f} true_pos_frac={va['true_pos_frac']:.4f} "
            f"pred_pos_frac={va['pred_pos_frac']:.4f}"
        )
        print("Val cutoffs:")
        for t in thresholds:
            s = va["by_threshold"][float(t)]
            print(
                f"  thr={t:.3f} acc={s['acc']:.4f} prec={s['precision']:.4f} rec={s['recall']:.4f} "
                f"f1={s['f1']:.4f} iou={s['iou']:.4f} mcc={s['mcc']:.4f} pred_pos_frac={s['pred_pos_frac']:.4f}"
            )

        ckpt = dict(
            epoch=epoch + 1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            scaler_state=scaler.state_dict() if scaler is not None else None,
            best_val_mcc=best_mcc,
            best_val_loss=best_loss,
            stats=stats_to_jsonable(stats),
            args=vars(args),
            feature_set=args.feature_set,
            feature_names=feat_names,
            in_channels=in_channels,
            model_preset=C.MODEL_PRESET,
            loss_name=args.loss_name,
            ftnmt_depth=C.FTNMT_DEPTH,
            label_reduce=C.LABEL_REDUCE,
            dimensionless_inputs=bool(C.DIMENSIONLESS_INPUTS),
            augmentation=dict(
                augment_train=bool(C.AUGMENT_TRAIN),
                unit_scale=bool(C.AUGMENT_UNIT_SCALE),
                rotation=bool(C.AUGMENT_ROTATION),
                relative_stream_rotation=bool(enable_relative),
                geometry=geometry,
            ),
            eval_thresholds=list(thresholds),
            val_loss=va["loss"],
            val_mcc=va["mcc"],
            val_by_threshold=va["by_threshold"],
        )
        torch.save(ckpt, os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}.pt"))
        if va["mcc"] > best_mcc:
            best_mcc = va["mcc"]
            ckpt["best_val_mcc"] = best_mcc
            torch.save(ckpt, best_mcc_path)
            print(f"Saved new best-MCC checkpoint -> {best_mcc_path} (MCC@0.5={best_mcc:.4f})")
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            ckpt["best_val_loss"] = best_loss
            torch.save(ckpt, best_loss_path)
            print(f"Saved new best-loss checkpoint -> {best_loss_path} (loss={best_loss:.4f})")
        save_json(meta_path, dict(
            best_val_mcc=best_mcc,
            best_val_loss=best_loss,
            stats=stats_to_jsonable(stats),
            args=vars(args),
            feature_set=args.feature_set,
            feature_names=feat_names,
            in_channels=in_channels,
            model_preset=C.MODEL_PRESET,
            loss_name=args.loss_name,
            ftnmt_depth=C.FTNMT_DEPTH,
            label_reduce=C.LABEL_REDUCE,
            dimensionless_inputs=bool(C.DIMENSIONLESS_INPUTS),
            augmentation=ckpt["augmentation"],
            eval_thresholds=list(thresholds),
        ))

    if os.path.exists(best_mcc_path):
        te = evaluate_split(
            "test", args, best_mcc_path, stats, in_channels, thresholds,
            max_stream=args.test_stream_ref_cap,
            max_back=args.test_back_ref_cap,
        )
        path = os.path.join(args.out_dir, "litept_test_metrics.dat")
        test_log = xlogger(path)
        test_row = dict(split="testing", checkpoint=best_mcc_path, selection_metric="val_mcc")
        test_row.update(_flatten_epoch_row("test", te, thresholds))
        test_log.write(test_row)
        print(f"[TEST] @0.5: loss={te['loss']:.4f} f1={te['f1']:.4f} IoU={te['iou']:.4f} MCC={te['mcc']:.4f}")
        print("[TEST] wrote:", path)

    print(f"\nTraining finished | best val MCC@0.5={best_mcc:.4f} | best val loss={best_loss:.4f}")
    print("Log:", os.path.join(args.out_dir, C.LOG_NAME))


def build_argparser():
    p = argparse.ArgumentParser(description="LitePT phase-context astronomy segmentation")
    sub = p.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out_dir", type=str, default=C.OUT_DIR)
    tr.add_argument("--tv_root", type=str, default=C.TV_ROOT)
    tr.add_argument("--train_stream_ref_cap", type=int, default=C.TRAIN_STREAM_REF_CAP)
    tr.add_argument("--train_back_ref_cap", type=int, default=C.TRAIN_BACK_REF_CAP)
    tr.add_argument("--val_stream_ref_cap", type=int, default=C.VAL_STREAM_REF_CAP)
    tr.add_argument("--val_back_ref_cap", type=int, default=C.VAL_BACK_REF_CAP)
    tr.add_argument("--test_stream_ref_cap", type=int, default=C.TEST_STREAM_REF_CAP)
    tr.add_argument("--test_back_ref_cap", type=int, default=C.TEST_BACK_REF_CAP)
    tr.add_argument("--stream_sampling_mode", type=str, default=C.STREAM_SAMPLING_MODE, choices=["equal_files", "sqrt_size", "proportional"])
    tr.add_argument("--background_sampling_mode", type=str, default=C.BACKGROUND_SAMPLING_MODE, choices=["equal_files", "sqrt_size", "proportional"])
    tr.add_argument("--full_background_small_files", type=int, default=int(C.FULL_BACKGROUND_SMALL_FILES))
    tr.add_argument("--background_large_file_count", type=int, default=C.BACKGROUND_LARGE_FILE_COUNT)
    tr.add_argument("--large_background_file_names", type=str, default=C.LARGE_BACKGROUND_FILE_NAMES)
    tr.add_argument("--chunk_size", type=float, default=C.CHUNK_SIZE)
    tr.add_argument("--grid_size", type=float, default=C.GRID_SIZE)
    tr.add_argument("--train_chunks_per_epoch", type=int, default=C.TRAIN_CHUNKS_PER_EPOCH)
    tr.add_argument("--val_chunks", type=int, default=C.VAL_CHUNKS)
    tr.add_argument("--test_chunks", type=int, default=C.TEST_CHUNKS)
    tr.add_argument("--min_particles_per_chunk", type=int, default=C.MIN_PARTICLES_PER_CHUNK)
    tr.add_argument("--n_valid_centres", type=int, default=C.N_VALID_CENTRES)
    tr.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    tr.add_argument("--eval_batch_size", type=int, default=0, help="Validation/test batch size; 0 mirrors --batch_size")
    tr.add_argument("--epochs", type=int, default=C.EPOCHS)
    tr.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    tr.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    tr.add_argument("--optimizer", type=str, default=C.OPTIMIZER, choices=["adam", "adamw", "radam"])
    tr.add_argument("--lr_schedule", type=str, default=C.LR_SCHEDULE, choices=["none", "warmup", "warmup_cosine"])
    tr.add_argument("--warmup_epochs", type=int, default=C.WARMUP_EPOCHS)
    tr.add_argument("--cosine_t0", type=int, default=C.COSINE_T0)
    tr.add_argument("--cosine_t_mult", type=int, default=C.COSINE_T_MULT)
    tr.add_argument("--num_workers", type=int, default=C.NUM_WORKERS_DEFAULT)
    tr.add_argument("--progress_every", type=int, default=getattr(C, "PROGRESS_EVERY", 5))
    tr.add_argument("--prefetch_factor", type=int, default=C.PREFETCH_FACTOR)
    tr.add_argument("--pin_memory", type=int, default=int(C.PIN_MEMORY))
    tr.add_argument("--cuda_prefetch", type=int, default=int(getattr(C, "CUDA_PREFETCH", 0)))
    tr.add_argument("--materialized_cache", type=int, default=int(getattr(C, "CACHE_MATERIALIZED_SAMPLES", False)))
    tr.add_argument("--materialized_cache_max_items", type=int, default=int(getattr(C, "MATERIALIZED_CACHE_MAX_ITEMS", 0)))
    tr.add_argument("--cache_spatial_queries", type=int, default=int(getattr(C, "CACHE_SPATIAL_QUERIES", False)))
    tr.add_argument("--spatial_query_cache_max_centres", type=int, default=int(getattr(C, "SPATIAL_QUERY_CACHE_MAX_CENTRES", 0)))
    tr.add_argument("--max_raw_particles_per_chunk", type=int, default=int(getattr(C, "MAX_RAW_PARTICLES_PER_CHUNK", 0)))
    tr.add_argument("--train_max_native_voxels_per_chunk", type=int, default=int(getattr(C, "TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK", getattr(C, "MAX_NATIVE_VOXELS_PER_CHUNK", 0))))
    tr.add_argument("--eval_max_native_voxels_per_chunk", type=int, default=int(getattr(C, "EVAL_MAX_NATIVE_VOXELS_PER_CHUNK", 0)))
    tr.add_argument("--activation_checkpointing", type=int, default=int(getattr(C, "ACTIVATION_CHECKPOINTING", False)))
    tr.add_argument("--raw_subsample_mode", type=str, default=getattr(C, "RAW_SUBSAMPLE_MODE", "balanced"), choices=["balanced", "proportional", "target_prior"])
    tr.add_argument("--raw_subsample_target_stream_frac", type=float, default=float(getattr(C, "RAW_SUBSAMPLE_TARGET_STREAM_FRAC", 0.10)))
    tr.add_argument("--label_reduce", type=str, default=C.LABEL_REDUCE, choices=["majority", "any_positive"])
    tr.add_argument("--model_preset", type=str, default=C.MODEL_PRESET, choices=["native_tiny", "native_base", "native_large"])
    tr.add_argument("--augment_train", type=int, default=int(C.AUGMENT_TRAIN))
    tr.add_argument("--augment_unit_scale", type=int, default=int(C.AUGMENT_UNIT_SCALE))
    tr.add_argument("--augment_rotation", type=int, default=int(C.AUGMENT_ROTATION))
    tr.add_argument("--relative_stream_rotation", type=int, default=int(C.SYNTH_RELATIVE_STREAM_ROTATION))
    tr.add_argument("--hybrid_ce_weight", type=float, default=float(getattr(C, "HYBRID_CE_WEIGHT", 0.5)))
    tr.add_argument("--hybrid_ftnmt_weight", type=float, default=float(getattr(C, "HYBRID_FTNMT_WEIGHT", 0.5)))
    tr.add_argument("--resume_from", type=str, default="")
    tr.add_argument("--feature_set", type=str, default=C.FEATURE_SET, help="Predefined: " + ", ".join(FEATURE_GROUPS) + "; or comma-separated custom list")
    tr.add_argument("--loss_name", type=str, default=C.LOSS_NAME, choices=["ftnmt", "ce", "weighted_ce", "hybrid_ce_ftnmt", "hybrid"])
    tr.add_argument("--eval_thresholds", type=str, default=",".join(map(str, C.EVAL_THRESHOLDS)))
    return p


def main():
    args = build_argparser().parse_args()
    if hasattr(args, "full_background_small_files"):
        args.full_background_small_files = bool(args.full_background_small_files)
    if args.mode == "train":
        train_main(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
