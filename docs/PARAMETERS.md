# Command-line parameter guide

The authoritative list, accepted choices, and defaults are always available
through each program's `--help` output. This page groups the most important
controls by purpose.

## Common concepts

- `--out_dir`: directory for checkpoints, logs, metrics, and predictions.
- `--tv_root`: root containing the training, validation, and where applicable
  testing directories.
- `--epochs`, `--batch_size`, `--lr`, `--weight_decay`: standard optimisation
  controls.
- `--num_workers`: data-loader worker count. Reduce this if the scheduler
  provides limited shared memory.
- `--threshold`: score threshold used to create a hard stream/background label.
- `--*_ref_cap`: maximum size of a class-specific reference population.
- `--*_sampling_mode`: file/particle sampling policy; supported values are
  reported by `--help`.

## PointCloud

Training:

- `--patch_k`: number of neighbours in each local patch.
- `--patch_space`: construct neighbours in position (`xyz`) or weighted
  position–velocity space (`xyzv`).
- `--patch_vel_weight`: relative velocity weighting in `xyzv` neighbour search.
- `--local_box_size`: spatial extent used for local sampling.
- `--train_*_centres`, `--val_*_centres`: number of stream/background patch
  centres.
- `--label_smoothing`, `--background_loss_weight`: loss controls.
- `--use_patch_augmentation` and `--augment_*`: optional patch augmentation.

Inference:

- positional `test_dataset`: HDF5/NumPy test file or configured filename.
- `--ckpt_path`: trained checkpoint.
- `--infer_chunk_size`: inference batching control.
- `--aux_feature_dtype`: storage type for auxiliary arrays.

## XGBoost

The common parser defines output, test, training-stream, validation-stream,
training-background, and validation-background directories. The `train` and
`infer` subcommands expose additional feature-generation, model, threshold,
checkpoint, and output controls. Use:

```bash
python XGBoost/xgboost.py train --help
python XGBoost/xgboost.py infer --help
```

The example PBS files show complete representative commands. Inference scores
should be described as stream-likeness scores unless calibration has been
independently established.

## Voxel CNN

Training:

- `--brick_size`: physical side length of each spatial brick.
- `--grid_size`: number of voxel cells per brick dimension.
- `--train_bricks`, `--val_bricks`: numbers of sampled bricks.
- `--min_particles_per_brick`: minimum occupancy accepted for a brick.
- `--max_*_points`, `--val_*_points`: class-specific reference caps.

Inference:

- positional `test_dataset`: HDF5/NumPy input.
- `--ckpt_path`: trained checkpoint.
- `--infer_stride_fraction`: overlap stride relative to brick size.
- `--edge_weight_power`: blending weight applied near brick boundaries.

## 6D-LitePT

Training data and representation:

- `--chunk_size`: physical side length of each spatial chunk.
- `--grid_size`: phase-space discretisation scale.
- `--train_chunks_per_epoch`, `--val_chunks`, `--test_chunks`: sampled chunk
  counts.
- `--min_particles_per_chunk`: minimum accepted chunk occupancy.
- `--feature_set`: predefined feature group or comma-separated feature list.
- `--label_reduce`: rule used to obtain occupied-cell labels.

Model and optimisation:

- `--model_preset`: `native_tiny`, `native_base`, or `native_large`.
- `--loss_name`: CE, weighted CE, FTNMT, or hybrid objective.
- `--optimizer`, `--lr_schedule`, `--warmup_epochs`, `--cosine_t0`,
  `--cosine_t_mult`: optimiser and schedule controls.
- `--eval_thresholds`: comma-separated evaluation thresholds.
- `--resume_from`: checkpoint from which to continue training.

Memory and performance:

- `--max_raw_particles_per_chunk`: optional raw-particle cap; zero disables it.
- `--train_max_native_voxels_per_chunk` and
  `--eval_max_native_voxels_per_chunk`: occupied-cell caps.
- `--activation_checkpointing`: trade compute for reduced activation memory.
- `--prefetch_factor`, `--pin_memory`, `--cuda_prefetch`: loading controls.

Inference:

- `--input_h5`, `--output_h5`: input and output HDF5 paths.
- `--run_dir`, `--checkpoint`, `--meta`: trained-run files.
- `--mode`: global, tiled, or both.
- `--thresholds`: comma-separated output thresholds.
- `--tile_size`, `--tile_stride`, `--tile_min_particles`: tiled traversal.
- `--gaussian_edge_weight`, `--gaussian_sigma_norm`: overlap blending.
- `--max_particles`, `--max_tiles`: debugging caps; zero means no cap.

## Boolean integer flags

Several training options use `0` and `1` rather than `store_true`. Check the
individual `--help` output before editing a job script.
