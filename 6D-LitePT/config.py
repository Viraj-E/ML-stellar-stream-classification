
"""Configuration for the native 6D-LitePT pipeline.

Raw particles are voxelised directly in six-dimensional phase space and the
occupied-cell coordinates are passed to a sparse-token LitePT backbone without
the original three-dimensional sparse-convolution path.
"""

import os

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- paths -----------------------------------------
TV_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "data", "train_val_test"))
TRAIN_STREAM_DIR = os.path.join(TV_ROOT, "training", "streams")
VAL_STREAM_DIR = os.path.join(TV_ROOT, "validation", "streams")
TRAIN_BACK_DIR = os.path.join(TV_ROOT, "training", "background")
VAL_BACK_DIR = os.path.join(TV_ROOT, "validation", "background")
TEST_STREAM_DIR = os.path.join(TV_ROOT, "testing", "streams")
TEST_BACK_DIR = os.path.join(TV_ROOT, "testing", "background")

STREAM_DIR = TRAIN_STREAM_DIR
BACK_DIR = TRAIN_BACK_DIR
OUT_DIR = os.path.join(_THIS_DIR, "outputs")

# HDF5 particle groups / scaling
TEST_POS_SCALE = 1.0
TEST_PART_TYPE = "PartType4"
BACKGROUND_PART_TYPE = "PartType1"
ID_DATASET_NAMES = ("ParticleIDs", "ParticleID", "ParticleId", "particle_ids", "ids")

# ----------------------------- hardware --------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
USE_MEMMAP = True
USE_AMP = True
PIN_MEMORY = True
AVAILABLE_CPUS = 16
NUM_WORKERS_DEFAULT = 8
PREFETCH_FACTOR = 2
PROGRESS_EVERY = 50
CUDA_PREFETCH = False

# ----------------------------- fixed split caps ------------------------------
# Default class ratios are 1:9 for training and 1:10 for validation/testing.
TRAIN_STREAM_REF_CAP = 300_000
TRAIN_BACK_REF_CAP = 2_700_000
VAL_STREAM_REF_CAP = 300_000
VAL_BACK_REF_CAP = 3_000_000
TEST_STREAM_REF_CAP = 300_000
TEST_BACK_REF_CAP = 3_000_000

MAX_STREAM_TRAIN_POINTS = TRAIN_STREAM_REF_CAP
MAX_BACK_TRAIN_POINTS = TRAIN_BACK_REF_CAP
MAX_STREAM_VAL_POINTS = VAL_STREAM_REF_CAP
MAX_BACK_VAL_POINTS = VAL_BACK_REF_CAP
MAX_STREAM_TEST_POINTS = TEST_STREAM_REF_CAP
MAX_BACK_TEST_POINTS = TEST_BACK_REF_CAP

STREAM_SAMPLING_MODE = "proportional"
BACKGROUND_SAMPLING_MODE = "proportional"
FULL_BACKGROUND_SMALL_FILES = False
BACKGROUND_LARGE_FILE_COUNT = 2
LARGE_BACKGROUND_FILE_NAMES = ""

# ----------------------------- chunking --------------------------------------
CHUNK_SIZE = 128.0
GRID_SIZE = 4.0
TRAIN_CHUNKS_PER_EPOCH = 4000
VAL_CHUNKS = 600
TEST_CHUNKS = 600
MIN_PARTICLES_PER_CHUNK = 1024
N_VALID_CENTRES = 5000
LABEL_REDUCE = "any_positive"
# Training-only cap applied after centre selection and before slicing raw arrays.
# Keeps random raw sampling plus fresh rotations/unit scales, but avoids processing
# the full dense background-heavy chunk on every worker fetch.
MAX_RAW_PARTICLES_PER_CHUNK = 160_000
RAW_SUBSAMPLE_MODE = "balanced"  # "balanced" | "proportional" | "target_prior"
RAW_SUBSAMPLE_TARGET_STREAM_FRAC = 0.10

# The full fixed split is dense enough that caching query indices for every
# valid centre can exceed tens of GB. Cache a bounded centre pool, matching the
# fast reference implementation while keeping RAM comfortably below 27 GB.
CACHE_SPATIAL_QUERIES = True
SPATIAL_QUERY_CACHE_MAX_CENTRES = 1024

# Dense chunks can contain >1M raw particles before voxelisation. The training
# loop uses sampled logits only, so do not carry raw inverse maps in batches.
RETURN_INVERSE = False
MAX_DIRECT_GRID_CELLS = 2_000_000

# Optional worker-local cache of already voxelized samples. This trades RAM for
# faster later epochs; keep disabled for exact per-epoch resampling semantics.
CACHE_MATERIALIZED_SAMPLES = False
MATERIALIZED_CACHE_MAX_ITEMS = 0


# ----------------------------- optimisation ----------------------------------
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
OPTIMIZER = "adam"
LR_SCHEDULE = "warmup_cosine"
WARMUP_EPOCHS = 5
COSINE_T0 = 10
COSINE_T_MULT = 2
LOSS_NAME = "ftnmt"
FTNMT_DEPTH = 0
HYBRID_CE_WEIGHT = 0.50
HYBRID_FTNMT_WEIGHT = 0.50

# ----------------------------- metrics / inference ---------------------------
DEFAULT_THRESHOLD = 0.50
EVAL_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
INFER_STRIDE_FRACTION = 0.5

# ----------------------------- outputs ---------------------------------------
CKPT_NAME = "litept_best_mcc.pt"
CKPT_BEST_LOSS_NAME = "litept_best_loss.pt"
CKPT_EPOCH_DIR = "checkpoints"
META_NAME = "litept_meta.json"
LOG_NAME = "train_log.dat"

# ----------------------------- feature selection -----------------------------
FEATURE_SET = "all"

# LitePT-friendly approximation of the XGBoost phase-space KNN summaries.
# These channels are appended after voxel/grid sampling, so they summarize the
# raw particles that landed in each LitePT voxel without moving LitePT to 6D.
ENABLE_VOXEL_CONTEXT_FEATURES = True
VOXEL_CONTEXT_FEATURES = (
    "voxel_log_count",
    "voxel_sigma_v",
    "voxel_sigma_vr",
    "voxel_sigma_speed",
    "voxel_L_align",
    "voxel_sigma_Lmag",
)

# ----------------------------- phase-space geometry ---------------------------
# Native 6D coordinates use the XGBoost phase-space metric. In dimensionless
# mode [xyz, 0.05 * vxyz] becomes:
#   [coord_norm_xyz, 0.05 * vel_scale / pos_scale * vel_norm_xyz]
PHASE_VEL_WEIGHT = 0.05
NATIVE_COORD_DIM = 6
NATIVE_SERIAL_DEPTH_WARN = 10
TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK = int(os.environ.get("TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK", os.environ.get("MAX_NATIVE_VOXELS_PER_CHUNK", "25000")))
EVAL_MAX_NATIVE_VOXELS_PER_CHUNK = int(os.environ.get("EVAL_MAX_NATIVE_VOXELS_PER_CHUNK", "0"))
MAX_NATIVE_VOXELS_PER_CHUNK = TRAIN_MAX_NATIVE_VOXELS_PER_CHUNK
VOXEL_CAP_MODE = "hash"
ACTIVATION_CHECKPOINTING = bool(int(os.environ.get("LITEPT_ACTIVATION_CHECKPOINTING", "0")))

# Unit-invariant representation. Coordinates and phase-space feature channels
# are divided by robust per-chunk position/velocity scales before voxelisation.
DIMENSIONLESS_INPUTS = True
STANDARDIZE_DIMENSIONLESS_FEATURES = False
POSITION_SCALE_QUANTILE = 0.90
VELOCITY_SCALE_QUANTILE = 0.90
MIN_POSITION_SCALE = 1e-6
MIN_VELOCITY_SCALE = 1e-6
MIN_NORMALIZED_GRID_SIZE = 1e-4

# ----------------------------- augmentations ---------------------------------
AUGMENT_TRAIN = True
AUGMENT_UNIT_SCALE = True
UNIT_SCALE_LOG2_MIN = -1.0
UNIT_SCALE_LOG2_MAX = 1.0

# Random rotations are physically consistent coordinate-frame augmentations when
# applied to both positions and velocities. A relative stream rotation is guarded
# by a train-split geometry check before it is enabled.
AUGMENT_ROTATION = True
ROTATION_PROB = 1.0
SYNTHESIS_AUGMENT = True
SYNTH_RELATIVE_STREAM_ROTATION = True
RELATIVE_ROTATION_PROB = 0.35
GEOMETRY_SAMPLE_CAP = 200_000
STREAM_NONSPHERICAL_AXIS_RATIO_MIN = 1.15
BACKGROUND_SPHERICAL_AXIS_RATIO_MAX = 1.30

# ----------------------------- LitePT presets --------------------------------
MODEL_PRESET = "native_base"

_LITEPT_PRESETS = {
    "native_base": dict(
        order=("z", "z-rev", "z-evenodd", "z-oddeven"), stride=(2, 2, 2),
        enc_depths=(3, 3, 3, 3), enc_channels=(64, 128, 256, 512),
        enc_num_head=(4, 8, 16, 32), enc_patch_size=(512, 512, 512, 512),
        enc_conv=(False, False, False, False), enc_attn=(True, True, True, True),
        enc_rope_freq=(0.0, 0.0, 0.0, 0.0),
        dec_depths=(0, 0, 0), dec_channels=(96, 128, 256),
        dec_num_head=(6, 8, 16), dec_patch_size=(512, 512, 512),
        dec_conv=(False, False, False), dec_attn=(False, False, False),
        dec_rope_freq=(0.0, 0.0, 0.0),
        mlp_ratio=3, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
        drop_path=0.15, shuffle_orders=True, pre_norm=True, FINAL_DIM=96,
    ),
    "native_large": dict(
        order=("z", "z-rev", "z-evenodd", "z-oddeven"), stride=(2, 2, 2),
        enc_depths=(3, 4, 6, 4), enc_channels=(72, 144, 288, 576),
        enc_num_head=(4, 8, 16, 32), enc_patch_size=(512, 512, 512, 512),
        enc_conv=(False, False, False, False), enc_attn=(True, True, True, True),
        enc_rope_freq=(0.0, 0.0, 0.0, 0.0),
        dec_depths=(0, 0, 0), dec_channels=(96, 144, 288),
        dec_num_head=(6, 8, 16), dec_patch_size=(512, 512, 512),
        dec_conv=(False, False, False), dec_attn=(False, False, False),
        dec_rope_freq=(0.0, 0.0, 0.0),
        mlp_ratio=3, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
        drop_path=0.2, shuffle_orders=True, pre_norm=True, FINAL_DIM=96,
    ),
    "native_tiny": dict(
        order=("z", "z-rev"), stride=(2, 2),
        enc_depths=(1, 1, 1), enc_channels=(32, 64, 128),
        enc_num_head=(2, 4, 8), enc_patch_size=(256, 256, 256),
        enc_conv=(False, False, False), enc_attn=(True, True, True),
        enc_rope_freq=(0.0, 0.0, 0.0),
        dec_depths=(0, 0), dec_channels=(48, 64),
        dec_num_head=(3, 4), dec_patch_size=(256, 256),
        dec_conv=(False, False), dec_attn=(False, False),
        dec_rope_freq=(0.0, 0.0),
        mlp_ratio=2, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
        drop_path=0.05, shuffle_orders=True, pre_norm=True, FINAL_DIM=48,
    ),
}


def get_model_config():
    if MODEL_PRESET not in _LITEPT_PRESETS:
        raise ValueError(f"Unknown MODEL_PRESET {MODEL_PRESET!r}; choose {list(_LITEPT_PRESETS)}")
    return dict(_LITEPT_PRESETS[MODEL_PRESET])


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.benchmark = True
# Portions derived from LitePT: https://github.com/prs-eth/LitePT
# Original copyright (c) 2025 Photogrammetry and Remote Sensing Lab.
# LitePT and these modifications are distributed under the MIT License.
# See ../licenses/LitePT-LICENSE and ../THIRD_PARTY_NOTICES.md.
