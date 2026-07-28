# Stellar Stream Classification Pipelines

Four particle-level stellar-stream classification pipelines used to compare
tree-based, point-cloud, voxel, and sparse-transformer representations:

- `PointCloud/pointcloud.py` — hybrid local point-cloud classifier.
- `XGBoost/xgboost.py` — gradient-boosted trees using local phase-space summaries.
- `VoxelCNN/voxel_cnn.py` — three-dimensional voxel CNN.
- `6D-LitePT/` — LitePT-derived transformer operating on occupied cells in
  weighted six-dimensional phase space.

These are complete classification pipelines. They differ in input
representation, feature construction, objective, aggregation, and inference
procedure; the repository is not an architecture-only comparison.

## Repository layout

```text
PointCloud/              PointCloud training and inference
XGBoost/                 XGBoost training and inference
VoxelCNN/                Voxel CNN training and inference
6D-LitePT/               6D-LitePT model, training, inference, and diagnostics
jobs/                    Example PBS training and inference scripts
docs/PARAMETERS.md       Guide to the main command-line parameters
docs/GITHUB_GUIDE.md     First-time repository publication guide
DEPENDENCIES.md          Required packages and tested hardware
requirements-*.txt       Package lists for the three software stacks
LICENSE                  MIT licence for this repository
THIRD_PARTY_NOTICES.md   LitePT attribution and licensing notice
licenses/                Copies of applicable third-party licences
```

## Data

PointCloud, XGBoost, and Voxel CNN expect:

```text
DATA_ROOT/
├── training/
│   ├── streams/
│   └── background/
└── validation/
    ├── streams/
    └── background/
```

6D-LitePT additionally expects `testing/streams/` and
`testing/background/`. The programs read HDF5 particle data. Depending on the
operation, fields include coordinates, velocities, masses, labels, and particle
IDs. Simulation data, trained checkpoints, and generated outputs are not
included.

## Dependencies and hardware

The code was run on an institutional HPC system:

- PointCloud, Voxel CNN, and XGBoost used NVIDIA Volta GPUs.
- 6D-LitePT used an NVIDIA Hopper GPU.

Required top-level packages are listed in `DEPENDENCIES.md` and the three
`requirements-*.txt` files. Exact environment-creation commands are not
provided because CUDA-enabled PyTorch, FAISS, CuPy, FlashAttention, and compiled
extensions must be selected for the user's local drivers, GPU architecture,
Python version, and HPC module system. Dependencies installed transitively by
the named packages are not listed individually.

## Usage

Each program exposes command-line help:

```bash
python PointCloud/pointcloud.py train --help
python PointCloud/pointcloud.py infer --help

python XGBoost/xgboost.py train --help
python XGBoost/xgboost.py infer --help

python VoxelCNN/voxel_cnn.py train --help
python VoxelCNN/voxel_cnn.py infer --help

python 6D-LitePT/train.py train --help
python 6D-LitePT/inference.py --help
```

The `jobs/` directory contains one editable PBS training template and one
inference template for each pipeline. Replace the placeholders in the `User
configuration` section, adjust resources for the target system, and submit, for
example:

```bash
qsub jobs/train_pointcloud.pbs
```

The templates demonstrate the commands and parameters used by the code; they
are not universal resource requirements. See `docs/PARAMETERS.md` for a grouped
parameter guide.

## Main outputs

- PointCloud: stream probability, hard label, logits, margin, feature norm,
  patch summaries, and latent representations.
- XGBoost: stream-likeness scores and hard predictions.
- Voxel CNN: blended particle-level stream probabilities and predictions.
- 6D-LitePT: occupied-cell outputs mapped back to particles, including
  overlapping-tile blended predictions.

## Licence, attribution, and citation

This repository is released under the MIT License; see `LICENSE`.

The 6D-LitePT pipeline is derived from the MIT-licensed
[LitePT](https://github.com/prs-eth/LitePT) architecture of Yue et al. It is
adapted for occupied cells in six-dimensional stellar phase space, binary
stream classification, and particle-level reconstruction of cell outputs.
The upstream copyright and licence are retained in
`licenses/LitePT-LICENSE`; see `THIRD_PARTY_NOTICES.md` for details.
