import os
import sys

import numpy as np
import torch

import config as C
from data import make_litept_inputs
from dataset import LitePTChunkDataset, collate_litept, grid_sample_chunk
from network import LitePTBinarySeg


def _controlled_voxel_check():
    origin = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    coord6 = np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    feat = np.ones((2, 3), dtype=np.float32)
    label = np.array([0, 1], dtype=np.int64)
    coord_s, feat_s, label_s, grid_s, origin_s, inverse = grid_sample_chunk(
        origin,
        feat,
        label,
        grid_size=0.5,
        label_reduce="any_positive",
        return_inverse=True,
        phase_coord=coord6,
    )
    xyz_grid = np.floor(origin / 0.5).astype(np.int32)
    if np.unique(xyz_grid, axis=0).shape[0] != 1:
        raise AssertionError("controlled xyz setup should collapse to one xyz voxel")
    if len(coord_s) != 2 or grid_s.shape[1] != 6 or coord_s.shape[1] != 6:
        raise AssertionError("native 6D voxelization did not preserve velocity separation")
    if origin_s.shape != (2, 3):
        raise AssertionError(f"origin diagnostics should remain xyz, got {origin_s.shape}")
    if inverse.tolist() != [0, 1]:
        raise AssertionError(f"unexpected native inverse map {inverse}")


def _input_builder_check():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(64, 14)).astype(np.float32)
    X[:, 0:3] *= 32.0
    X[:, 3:6] *= 120.0
    centre = X[:, 0:3].mean(axis=0)
    stats = {
        "feat_mean": np.zeros(17, dtype=np.float32),
        "feat_std": np.ones(17, dtype=np.float32),
    }
    coord_xyz, feat, coord6, grid_size_eff = make_litept_inputs(
        X, centre, stats, feature_set="all", grid_size=4.0
    )
    if coord_xyz.shape[1] != 3:
        raise AssertionError(f"coord_xyz should be 3D diagnostics, got {coord_xyz.shape}")
    if coord6.shape[1] != 6:
        raise AssertionError(f"coord6 should be native 6D, got {coord6.shape}")
    coord_s, feat_s, label_s, grid_s, origin_s, _ = grid_sample_chunk(
        coord_xyz,
        feat,
        rng.integers(0, 2, size=len(X), dtype=np.int64),
        grid_size_eff,
        label_reduce="any_positive",
        phase_coord=coord6,
    )
    item = {
        "coord": torch.from_numpy(coord_s).float(),
        "grid_coord": torch.from_numpy(grid_s).int(),
        "origin_coord": torch.from_numpy(origin_s).float(),
        "feat": torch.from_numpy(feat_s).float(),
        "segment": torch.from_numpy(label_s).long(),
    }
    batch = collate_litept([item, item])
    if batch["coord"].shape[1] != 6 or batch["grid_coord"].shape[1] != 6:
        raise AssertionError("collate did not preserve native 6D coord/grid_coord")
    if "phase_coord" in batch:
        raise AssertionError("native 6D batch should not expose phase_coord as an auxiliary input")


def _dataset_getitem_check():
    rng = np.random.default_rng(321)
    stream = rng.normal(size=(96, 14)).astype(np.float32)
    background = rng.normal(size=(192, 14)).astype(np.float32)
    stream[:, 0:3] *= 3.0
    background[:, 0:3] *= 3.0
    stream[:, 3:6] *= 50.0
    background[:, 3:6] *= 50.0
    stats = {
        "feat_mean": np.zeros(17, dtype=np.float32),
        "feat_std": np.ones(17, dtype=np.float32),
    }
    ds = LitePTChunkDataset(
        stream,
        background,
        stats,
        chunk_size=20.0,
        grid_size=1.0,
        n_chunks=2,
        min_particles=16,
        n_valid_target=4,
        feature_set="all",
        train_mode=False,
    )
    item = ds[0]
    if item["coord"].shape[1] != 6 or item["grid_coord"].shape[1] != 6:
        raise AssertionError("LitePTChunkDataset item is not native 6D")
    if item["origin_coord"].shape[1] != 3:
        raise AssertionError("LitePTChunkDataset lost xyz diagnostics")
    if "phase_coord" in item:
        raise AssertionError("LitePTChunkDataset should not expose phase_coord")


def _source_guard_check():
    model_path = os.path.join(os.path.dirname(__file__), "litept6d", "model.py")
    with open(model_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    banned = ("SubMConv3d", "SparseConvTensor", "sparse_conv_feat", "PointROPE", "reshape(-1, 3)")
    hits = [item for item in banned if item in text]
    if hits:
        raise AssertionError(f"native model contains banned 3D path markers: {hits}")
    if "spconv" in sys.modules:
        raise AssertionError("native model import should not import spconv")


def _cuda_forward_backward_check():
    if not torch.cuda.is_available():
        raise RuntimeError("native 6D smoke test requires CUDA")
    C.MODEL_PRESET = "native_tiny"
    device = torch.device("cuda")
    torch.manual_seed(7)
    n = 384
    in_channels = 8
    grid = torch.randint(0, 16, (n, 6), dtype=torch.int32, device=device)
    coord = grid.float() * 0.1 + 0.01 * torch.randn(n, 6, device=device)
    feat = torch.randn(n, in_channels, device=device)
    target = torch.randint(0, 2, (n,), dtype=torch.long, device=device)
    data = {
        "coord": coord,
        "grid_coord": grid,
        "feat": feat,
        "segment": target,
        "offset": torch.tensor([n], dtype=torch.long, device=device),
    }
    model = LitePTBinarySeg(in_channels=in_channels).to(device)
    if getattr(model.backbone, "coord_dim", None) != 6:
        raise AssertionError("native backbone does not advertise coord_dim=6")
    model.train()
    logits, point = model(data)
    if logits.shape != (n, 2):
        raise AssertionError(f"decoder should return input token count, got {logits.shape}")
    if point.grid_coord.shape[1] != 6 or point.coord.shape[1] != 6:
        raise AssertionError("forward path did not preserve 6D point coordinates")
    if int(point.serialized_ndim) != 6:
        raise AssertionError(f"serialized_ndim should be 6, got {point.serialized_ndim}")
    if int(point.serialized_depth) * 6 + 1 > 63:
        raise AssertionError("serialized depth violates 63-bit budget")
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    if grad_norm <= 0:
        raise AssertionError("backward pass produced no gradients")


def main():
    _source_guard_check()
    _controlled_voxel_check()
    _input_builder_check()
    _dataset_getitem_check()
    _cuda_forward_backward_check()
    print("Native 6D smoke test passed")


if __name__ == "__main__":
    main()
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
