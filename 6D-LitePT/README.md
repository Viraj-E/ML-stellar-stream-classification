# 6D-LitePT

This implementation adapts the LitePT transformer to native weighted
six-dimensional phase-space coordinates. Only occupied cells are retained;
the pipeline does not construct a dense six-dimensional tensor or use the
original three-dimensional sparse-convolution path.

Main entry points:

- `train.py` — training, validation, and fixed-split testing.
- `inference.py` — complete-particle and overlapping-tile inference.
- `smoke_test.py` — structural and CUDA forward/backward checks.
- `tools/` — training-log and particle-diagnostic plotting utilities.

Core modules:

- `config.py` — defaults and model presets.
- `data.py` — loading, feature construction, and scaling.
- `dataset.py` — spatial chunks and occupied-cell tokenisation.
- `losses.py` — cross-entropy, FTNMT, and hybrid objectives.
- `network.py` — binary segmentation wrapper.
- `litept6d/` — native 6D transformer backbone and serialisation.

Run `python train.py --help` and `python inference.py --help` for the complete
command-line interfaces.
