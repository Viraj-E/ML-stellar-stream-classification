# Dependencies and tested hardware

This project does not prescribe a complete Python or HPC environment. GPU
packages must be installed in versions compatible with the user's operating
system, Python version, CUDA drivers, and available hardware.

## PointCloud and Voxel CNN

Tested on an NVIDIA Volta GPU. Top-level packages:

- Python
- NumPy
- h5py
- scikit-learn
- PyTorch
- GPU-enabled FAISS

The corresponding package list is `requirements-pytorch.txt`. FAISS is noted
separately because its GPU installation is platform-specific.

## XGBoost

Tested on an NVIDIA Volta GPU. Top-level packages:

- Python
- NumPy
- h5py
- scikit-learn
- CuPy
- XGBoost with GPU support
- GPU-enabled FAISS

See `requirements-xgboost.txt`.

## 6D-LitePT

Tested on an NVIDIA Hopper GPU. Top-level packages:

- Python
- NumPy
- h5py
- PyTorch
- FlashAttention
- torch-scatter
- timm
- addict
- SciPy
- Matplotlib

See `requirements-litept.txt`. FlashAttention and torch-scatter are compiled
extensions and must match the installed Python, PyTorch, CUDA, and GPU
architecture.

## Installation note

Install the correct GPU-enabled PyTorch, FAISS, CuPy, XGBoost, FlashAttention,
and torch-scatter builds using their official instructions. The requirements
files document package names; they intentionally do not pin unverified versions
or claim to recreate the authors' cluster environment.

Packages installed automatically as dependencies of those listed above are not
enumerated individually.
