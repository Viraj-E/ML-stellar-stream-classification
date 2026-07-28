
"""Dataset and DataLoader helpers for the LitePT phase-context run."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config as C
from data import (
    apply_physical_augmentations,
    fmt,
    get_context_feature_names,
    make_litept_inputs,
    now,
    resolve_feature_set,
)


class SpatialCellIndex:
    def __init__(self, X, cell_size):
        self.X = np.asarray(X, dtype=np.float32)
        self.cell_size = float(cell_size)
        self.pos = self.X[:, 0:3]
        self.min_corner = self.pos.min(axis=0)
        cell_ids = np.floor((self.pos - self.min_corner) / self.cell_size).astype(np.int32)
        dims = cell_ids.max(axis=0) + 1
        flat = (
            cell_ids[:, 0].astype(np.int64) * dims[1] * dims[2]
            + cell_ids[:, 1].astype(np.int64) * dims[2]
            + cell_ids[:, 2].astype(np.int64)
        )
        order = np.argsort(flat)
        flat_sorted = flat[order]
        breaks = np.concatenate([[0], np.where(np.diff(flat_sorted) != 0)[0] + 1, [len(flat)]])
        self.dims = dims
        self.cell_map = {}
        for k in range(len(breaks) - 1):
            key_flat = int(flat_sorted[breaks[k]])
            iz = key_flat % dims[2]
            rem = key_flat // dims[2]
            iy = rem % dims[1]
            ix = rem // dims[1]
            self.cell_map[(int(ix), int(iy), int(iz))] = order[breaks[k]:breaks[k + 1]]

    def query_box(self, centre, box_size):
        lo = centre - 0.5 * box_size
        hi = centre + 0.5 * box_size
        lo_id = np.floor((lo - self.min_corner) / self.cell_size).astype(np.int32)
        hi_id = np.floor((hi - self.min_corner) / self.cell_size).astype(np.int32)
        candidates = []
        for ix in range(lo_id[0], hi_id[0] + 1):
            for iy in range(lo_id[1], hi_id[1] + 1):
                for iz in range(lo_id[2], hi_id[2] + 1):
                    val = self.cell_map.get((int(ix), int(iy), int(iz)))
                    if val is not None:
                        candidates.append(val)
        if not candidates:
            return np.empty(0, dtype=np.int64)
        idx = np.concatenate(candidates)
        pos = self.pos[idx]
        return idx[np.all((pos >= lo) & (pos < hi), axis=1)]


def _append_voxel_context_features(
    feat_s,
    feat,
    counts,
    feature_names,
    context_names,
    *,
    flat=None,
    occupied=None,
    inverse=None,
    minlength=None,
):
    """Append cheap per-voxel phase-space summaries to LitePT features."""
    context_names = [] if context_names is None else list(context_names)
    if not context_names:
        return feat_s.astype(np.float32, copy=False)

    feature_names = [] if feature_names is None else list(feature_names)
    col = {name: i for i, name in enumerate(feature_names)}
    counts = np.asarray(counts, dtype=np.float64)
    denom = np.maximum(counts, 1.0)

    if flat is not None:
        if occupied is None or minlength is None:
            raise ValueError("direct context aggregation requires occupied and minlength")

        def group_sum(values):
            return np.bincount(
                flat, weights=np.asarray(values, dtype=np.float64), minlength=int(minlength)
            )[occupied]
    elif inverse is not None:
        if minlength is None:
            raise ValueError("inverse context aggregation requires minlength")

        def group_sum(values):
            return np.bincount(
                inverse, weights=np.asarray(values, dtype=np.float64), minlength=int(minlength)
            )
    else:
        raise ValueError("context aggregation requires flat or inverse")

    def group_std(name):
        values = feat[:, col[name]].astype(np.float64)
        mean = group_sum(values) / denom
        mean2 = group_sum(values * values) / denom
        return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

    extras = []
    for name in context_names:
        if name == "voxel_log_count":
            extras.append(np.log1p(counts))
        elif name == "voxel_sigma_v":
            idx = [col[axis] for axis in ("vx", "vy", "vz")]
            values = feat[:, idx].astype(np.float64)
            sum_v = sum(group_sum(values[:, j]) for j in range(3))
            sum_v2 = sum(group_sum(values[:, j] * values[:, j]) for j in range(3))
            mean = sum_v / np.maximum(3.0 * counts, 1.0)
            mean2 = sum_v2 / np.maximum(3.0 * counts, 1.0)
            extras.append(np.sqrt(np.maximum(mean2 - mean * mean, 0.0)))
        elif name == "voxel_sigma_vr":
            extras.append(group_std("vr"))
        elif name == "voxel_sigma_speed":
            extras.append(group_std("speed"))
        elif name == "voxel_L_align":
            idx = [col[axis] for axis in ("Lx", "Ly", "Lz")]
            L = feat[:, idx].astype(np.float64)
            L_norm = np.linalg.norm(L, axis=1)
            L_hat = L / np.maximum(L_norm[:, None], 1e-8)
            mean_hat = np.stack([group_sum(L_hat[:, j]) / denom for j in range(3)], axis=1)
            extras.append(np.linalg.norm(mean_hat, axis=1))
        elif name == "voxel_sigma_Lmag":
            extras.append(group_std("Lmag"))
        else:
            raise ValueError(f"Unknown voxel context feature: {name}")

    extra = np.stack(extras, axis=1).astype(np.float32)
    return np.hstack([feat_s.astype(np.float32, copy=False), extra]).astype(np.float32)



def grid_sample_chunk(
    coord, feat, label, grid_size, label_reduce="majority", return_inverse=False,
    feature_names=None, context_names=None, phase_coord=None,
    max_voxels=None, cap_mode=None,
):
    """Voxelize directly in native 6D phase space.

    ``coord`` is retained as xyz/origin diagnostics. ``phase_coord`` is the
    model coordinate and must be [N, 6]. The returned ``coord`` and
    ``grid_coord`` are both 6D, so downstream LitePT cannot silently fall back
    to xyz topology.
    """
    origin_coord = np.asarray(coord, dtype=np.float32)
    feat = np.asarray(feat, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)
    coord6 = origin_coord if phase_coord is None else np.asarray(phase_coord, dtype=np.float32)
    if coord6.ndim != 2 or coord6.shape[1] != int(getattr(C, "NATIVE_COORD_DIM", 6)):
        raise ValueError(f"native 6D voxelization requires [N, 6] coordinates, got {coord6.shape}")
    if len(coord6) != len(origin_coord):
        raise ValueError("native coord6 must have one row per input particle")

    grid_coord = np.floor(coord6 / float(grid_size)).astype(np.int32)
    grid_coord -= grid_coord.min(axis=0, keepdims=True)
    grid_coord_s, inverse, counts = np.unique(
        grid_coord,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    m = len(grid_coord_s)
    counts_f = counts.astype(np.float64)

    coord_s = np.empty((m, coord6.shape[1]), dtype=np.float64)
    for c in range(coord6.shape[1]):
        coord_s[:, c] = np.bincount(inverse, weights=coord6[:, c], minlength=m)
    origin_s = np.empty((m, origin_coord.shape[1]), dtype=np.float64)
    for c in range(origin_coord.shape[1]):
        origin_s[:, c] = np.bincount(inverse, weights=origin_coord[:, c], minlength=m)
    feat_s = np.empty((m, feat.shape[1]), dtype=np.float64)
    for c in range(feat.shape[1]):
        feat_s[:, c] = np.bincount(inverse, weights=feat[:, c], minlength=m)

    coord_s = (coord_s / counts_f[:, None]).astype(np.float32)
    origin_s = (origin_s / counts_f[:, None]).astype(np.float32)
    feat_s = (feat_s / counts_f[:, None]).astype(np.float32)
    feat_s = _append_voxel_context_features(
        feat_s, feat, counts_f, feature_names, context_names,
        inverse=inverse, minlength=m,
    )
    if label_reduce == "majority":
        pos_cnt = np.bincount(inverse, weights=label.astype(np.float64), minlength=m)
        label_s = (pos_cnt >= counts_f * 0.5).astype(np.int64)
    elif label_reduce == "any_positive":
        label_s = np.zeros(m, dtype=np.int64)
        np.maximum.at(label_s, inverse, label)
    else:
        raise ValueError(f"Unknown label_reduce={label_reduce}")
    if max_voxels is None:
        max_voxels = int(getattr(C, "MAX_NATIVE_VOXELS_PER_CHUNK", 0))
    else:
        max_voxels = int(max_voxels)
    if cap_mode is None:
        cap_mode = getattr(C, "VOXEL_CAP_MODE", "hash")
    cap_mode = str(cap_mode).lower()
    if max_voxels > 0 and m > max_voxels:
        if return_inverse:
            raise ValueError("RETURN_INVERSE is incompatible with MAX_NATIVE_VOXELS_PER_CHUNK")
        if cap_mode not in ("hash", "coordinate_hash", "label_independent"):
            raise ValueError(
                f"Unsupported VOXEL_CAP_MODE={cap_mode!r}; only label-independent hash capping is allowed"
            )
        primes = np.array([73856093, 19349663, 83492791, 2654435761, 97531, 314159], dtype=np.uint64)
        h = (grid_coord_s.astype(np.uint64) * primes[:grid_coord_s.shape[1]][None, :]).sum(axis=1)
        keep = np.argsort(h)[:max_voxels]
        keep = np.sort(keep)
        coord_s = coord_s[keep]
        feat_s = feat_s[keep]
        label_s = label_s[keep]
        grid_coord_s = grid_coord_s[keep]
        origin_s = origin_s[keep]

    inverse_out = inverse.astype(np.int64) if return_inverse else None
    return coord_s, feat_s, label_s, grid_coord_s.astype(np.int32), origin_s, inverse_out

def _worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(C.RANDOM_SEED + worker_id + 1)


class LitePTChunkDataset(Dataset):
    def __init__(
        self,
        stream_arr,
        bg_arr,
        stats,
        chunk_size,
        grid_size,
        n_chunks,
        min_particles,
        n_valid_target=None,
        feature_set=None,
        train_mode=False,
        enable_relative_stream_rotation=False,
    ):
        self.stream_arr = np.asarray(stream_arr, dtype=np.float32)
        self.bg_arr = np.asarray(bg_arr, dtype=np.float32)
        self.stats = stats
        self.chunk_size = float(chunk_size)
        self.grid_size = float(grid_size)
        self.n_chunks = int(n_chunks)
        self.min_particles = int(min_particles)
        self.feature_set = feature_set
        self.feature_names = resolve_feature_set(self.feature_set)
        self.context_feature_names = get_context_feature_names(self.feature_set)
        self.train_mode = bool(train_mode)
        self.enable_relative_stream_rotation = bool(enable_relative_stream_rotation)
        default_cap_name = "TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK" if self.train_mode else "EVAL_MAX_NATIVE_VOXELS_PER_CHUNK"
        default_cap = getattr(C, default_cap_name, getattr(C, "MAX_NATIVE_VOXELS_PER_CHUNK", 0))
        self.max_native_voxels_per_chunk = int(default_cap)
        self.voxel_cap_mode = str(getattr(C, "VOXEL_CAP_MODE", "hash")).lower()
        cap_desc = "uncapped" if self.max_native_voxels_per_chunk <= 0 else str(self.max_native_voxels_per_chunk)
        phase = "train" if self.train_mode else "eval"
        print(f"Native voxel cap ({phase}): {cap_desc} mode={self.voxel_cap_mode}")
        self.rng = np.random.default_rng(C.RANDOM_SEED)
        self.cache_materialized_samples = bool(getattr(C, "CACHE_MATERIALIZED_SAMPLES", False))
        max_cache = int(getattr(C, "MATERIALIZED_CACHE_MAX_ITEMS", 0))
        self.materialized_cache_max_items = int(self.n_chunks if max_cache <= 0 else max_cache)
        self._sample_cache = {}

        t0 = now()
        print("Building spatial indices ...")
        self.stream_index = SpatialCellIndex(self.stream_arr, self.chunk_size)
        self.bg_index = SpatialCellIndex(self.bg_arr, self.chunk_size)
        print(f"  indices built in {fmt(now() - t0)}")

        if n_valid_target is None:
            n_valid_target = max(int(getattr(C, "N_VALID_CENTRES", 5000)), int(self.n_chunks))

        t1 = now()
        print(f"Validating centres (target {n_valid_target}) ...")
        self.valid_centres = self._validate_centres_fast(n_valid_target)
        print(f"  found {len(self.valid_centres)} valid centres in {fmt(now() - t1)}")

        if len(self.valid_centres) == 0:
            raise RuntimeError("No valid centres found. Reduce min_particles_per_chunk or increase chunk_size.")
        if len(self.valid_centres) < self.n_chunks:
            print(
                f"Warning: only {len(self.valid_centres)} valid centres found for "
                f"{self.n_chunks} chunks. Centres will be reused with replacement."
            )

        self.cache_spatial_queries = bool(getattr(C, "CACHE_SPATIAL_QUERIES", False) and self.train_mode)
        self.cached_idx_s, self.cached_idx_b = None, None
        self.cache_centre_count = 0
        if self.cache_spatial_queries:
            cache_limit = int(getattr(C, "SPATIAL_QUERY_CACHE_MAX_CENTRES", len(self.valid_centres)))
            if cache_limit <= 0:
                self.cache_spatial_queries = False
                print("Spatial query caching disabled by SPATIAL_QUERY_CACHE_MAX_CENTRES <= 0.")
            else:
                t2 = now()
                self.cache_centre_count = min(len(self.valid_centres), cache_limit)
                print(f"Caching spatial queries for {self.cache_centre_count}/{len(self.valid_centres)} centres ...")
                self.cached_idx_s, self.cached_idx_b = [], []
                total_idx = 0
                for i, centre in enumerate(self.valid_centres[:self.cache_centre_count]):
                    idx_s = self.stream_index.query_box(centre, self.chunk_size)
                    idx_b = self.bg_index.query_box(centre, self.chunk_size)
                    total_idx += len(idx_s) + len(idx_b)
                    self.cached_idx_s.append(idx_s)
                    self.cached_idx_b.append(idx_b)
                    if (i + 1) % 256 == 0 or (i + 1) == self.cache_centre_count:
                        est_gb = total_idx * np.dtype(np.int64).itemsize / 1e9
                        print(f"    {i + 1}/{self.cache_centre_count} cached ... est_index_bytes={est_gb:.2f} GB")
                print(f"  cached in {fmt(now() - t2)}")
        if not self.cache_spatial_queries:
            print("Spatial query caching disabled; chunk indices will be queried lazily.")
        if self.cache_materialized_samples:
            print(
                f"Materialized sample cache enabled; "
                f"max_items_per_worker={self.materialized_cache_max_items}"
            )

    def _validate_centres_fast(self, n_valid_target):
        rng = np.random.default_rng(C.RANDOM_SEED + 1)
        centre_cap = 300_000
        ns = min(len(self.stream_arr), max(1, centre_cap // 10))
        nb = min(len(self.bg_arr), max(1, centre_cap - ns))
        candidates = np.vstack([
            self.stream_arr[rng.choice(len(self.stream_arr), ns, replace=False), 0:3],
            self.bg_arr[rng.choice(len(self.bg_arr), nb, replace=False), 0:3],
        ])
        rng.shuffle(candidates)
        s_pops = {k: len(v) for k, v in self.stream_index.cell_map.items()}
        b_pops = {k: len(v) for k, v in self.bg_index.cell_map.items()}
        valid, cs, mp = [], self.chunk_size, self.min_particles
        for c in candidates:
            lo, hi = c - 0.5 * cs, c + 0.5 * cs
            total = 0
            for pops, idx_obj in ((s_pops, self.stream_index), (b_pops, self.bg_index)):
                lo_id = np.floor((lo - idx_obj.min_corner) / idx_obj.cell_size).astype(int)
                hi_id = np.floor((hi - idx_obj.min_corner) / idx_obj.cell_size).astype(int)
                for ix in range(lo_id[0], hi_id[0] + 1):
                    for iy in range(lo_id[1], hi_id[1] + 1):
                        for iz in range(lo_id[2], hi_id[2] + 1):
                            total += pops.get((ix, iy, iz), 0)
            if total >= mp:
                valid.append(c)
                if len(valid) >= n_valid_target:
                    break
        return np.asarray(valid, dtype=np.float32)

    def __len__(self):
        return self.n_chunks

    def _allocate_raw_cap(self, n_s, n_b, max_raw):
        total = int(n_s) + int(n_b)
        mode = str(getattr(C, "RAW_SUBSAMPLE_MODE", "balanced")).lower()
        if total <= max_raw:
            return int(n_s), int(n_b)

        if mode == "proportional":
            keep_s = int(round(max_raw * (float(n_s) / max(total, 1))))
        elif mode == "target_prior":
            target = float(getattr(C, "RAW_SUBSAMPLE_TARGET_STREAM_FRAC", 0.10))
            target = min(max(target, 0.0), 1.0)
            keep_s = int(round(max_raw * target))
        elif mode == "balanced":
            keep_s = max_raw // 2
        else:
            raise ValueError(f"Unknown RAW_SUBSAMPLE_MODE={mode!r}")

        if n_s > 0:
            keep_s = min(max(1, keep_s), int(n_s))
        else:
            keep_s = 0
        keep_b = min(max_raw - keep_s, int(n_b))
        if n_b > 0 and keep_b <= 0:
            keep_b = 1
            keep_s = min(keep_s, max_raw - keep_b)

        spare = max_raw - keep_s - keep_b
        if spare > 0 and keep_b < n_b:
            extra = min(spare, int(n_b) - keep_b)
            keep_b += extra
            spare -= extra
        if spare > 0 and keep_s < n_s:
            keep_s += min(spare, int(n_s) - keep_s)
        return int(keep_s), int(keep_b)

    def _subsample_raw_indices(self, idx_s, idx_b):
        max_raw = int(getattr(C, "MAX_RAW_PARTICLES_PER_CHUNK", 0))
        if max_raw <= 0 or (len(idx_s) + len(idx_b)) <= max_raw:
            return idx_s, idx_b

        keep_s, keep_b = self._allocate_raw_cap(len(idx_s), len(idx_b), max_raw)
        if keep_s < len(idx_s):
            idx_s = idx_s[self.rng.choice(len(idx_s), size=keep_s, replace=False)]
        if keep_b < len(idx_b):
            idx_b = idx_b[self.rng.choice(len(idx_b), size=keep_b, replace=False)]
        return idx_s, idx_b

    def __getitem__(self, _idx):
        cache_key = int(_idx)
        if self.cache_materialized_samples:
            cached = self._sample_cache.get(cache_key)
            if cached is not None:
                return cached

        if self.cache_spatial_queries:
            i = self.rng.integers(0, self.cache_centre_count)
            centre = self.valid_centres[i]
            idx_s, idx_b = self.cached_idx_s[i], self.cached_idx_b[i]
        else:
            i = self.rng.integers(0, len(self.valid_centres))
            centre = self.valid_centres[i]
            idx_s = self.stream_index.query_box(centre, self.chunk_size)
            idx_b = self.bg_index.query_box(centre, self.chunk_size)
        idx_s, idx_b = self._subsample_raw_indices(idx_s, idx_b)
        Xs = self.stream_arr[idx_s]
        Xb = self.bg_arr[idx_b]

        if self.train_mode and C.AUGMENT_TRAIN:
            Xs, Xb, centre, _ = apply_physical_augmentations(
                Xs, Xb, centre, self.rng,
                enable_relative_stream_rotation=self.enable_relative_stream_rotation,
            )

        X = np.vstack([Xs, Xb])
        y = np.concatenate([
            np.ones(len(Xs), dtype=np.int64),
            np.zeros(len(Xb), dtype=np.int64),
        ])
        coord_xyz, feat, coord6, grid_size_eff = make_litept_inputs(
            X, centre, self.stats, feature_set=self.feature_set, grid_size=self.grid_size
        )
        coord_s, feat_s, y_s, grid_coord_s, origin_coord_s, inverse = grid_sample_chunk(
            coord_xyz, feat, y, grid_size_eff, label_reduce=C.LABEL_REDUCE,
            return_inverse=bool(getattr(C, "RETURN_INVERSE", False)),
            feature_names=self.feature_names,
            context_names=self.context_feature_names,
            phase_coord=coord6,
            max_voxels=self.max_native_voxels_per_chunk,
            cap_mode=self.voxel_cap_mode,
        )
        out = {
            "coord": torch.from_numpy(coord_s).float(),
            "grid_coord": torch.from_numpy(grid_coord_s).int(),
            "origin_coord": torch.from_numpy(origin_coord_s).float(),
            "feat": torch.from_numpy(feat_s).float(),
            "segment": torch.from_numpy(y_s).long(),
        }
        if inverse is not None:
            out["inverse"] = torch.from_numpy(inverse).long()
        if self.cache_materialized_samples and len(self._sample_cache) < self.materialized_cache_max_items:
            self._sample_cache[cache_key] = out
        return out


def collate_litept(batch):
    coords, grid_coords, origin_coords, feats, segments, offsets = [], [], [], [], [], []
    use_inverse = all("inverse" in item for item in batch)
    use_origin = all("origin_coord" in item for item in batch)
    inverses = [] if use_inverse else None
    sampled_cnt = 0
    for item in batch:
        n = item["coord"].shape[0]
        coords.append(item["coord"])
        grid_coords.append(item["grid_coord"])
        if use_origin:
            origin_coords.append(item["origin_coord"])
        feats.append(item["feat"])
        segments.append(item["segment"])
        if use_inverse:
            inverses.append(item["inverse"] + sampled_cnt)
        sampled_cnt += n
        offsets.append(sampled_cnt)
    out = {
        "coord": torch.cat(coords, 0),
        "grid_coord": torch.cat(grid_coords, 0),
        "feat": torch.cat(feats, 0),
        "segment": torch.cat(segments, 0),
        "offset": torch.tensor(offsets, dtype=torch.long),
    }
    if use_origin:
        out["origin_coord"] = torch.cat(origin_coords, 0)
    if use_inverse:
        out["inverse"] = torch.cat(inverses, 0)
    return out


def build_dataloader(dataset, batch_size, shuffle, drop_last, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(C.PIN_MEMORY and C.DEVICE == "cuda"),
        drop_last=drop_last,
        collate_fn=collate_litept,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(C, "PREFETCH_FACTOR", 2))
    return DataLoader(**kwargs)
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
