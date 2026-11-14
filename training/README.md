# HLGAM Training Tools

Training and dataset-capture code lives in this top-level folder so it can run in separate processes from the server runtime. Future server-imported inference code should live outside this folder, under the runtime package, so the server can import only inference components without importing training jobs.

PyTorch is available through the optional `ml` extra:

```bash
uv sync --extra ml
```

The `ml` extra also installs Matplotlib so training jobs can write loss and metric plots after training completes.

Static pose capture does not require PyTorch. It uses the existing camera, MediaPipe tracker, normalization, and feature-extraction pipeline:

```bash
uv run python training/capture_static_poses.py
```

Two-handed pose capture uses the same camera and MediaPipe pipeline, but requires two tracked hands in the frame. Each accepted pair pose stores a JPEG under `data/two_handed_poses/images/` and appends one CSV row to `data/two_handed_poses/two_handed_pose_frames.csv` with left/right landmarks, per-hand features, pair geometry, target variants, and pair-level labels from `training/label_schemas/two_hand_labels.json`:

```bash
uv run python training/capture_two_handed_poses.py
uv run python training/capture_two_handed_poses.py --target frame_rectangle
uv run python training/capture_two_handed_poses.py --remove-target frame_rectangle
```

Motion transformer capture uses the same camera and MediaPipe pipeline, but records short frame sequences. Each accepted sample stores JPEG frames under `data/motion_sequences/images/` and appends one parsed CSV row per recorded frame to `data/motion_sequences/motion_sequence_frames.csv`:

```bash
uv run python training/capture_motion_sequences.py
```

Useful motion capture options:

```bash
uv run python training/capture_motion_sequences.py --target drag --handedness right --speed normal
uv run python training/capture_motion_sequences.py --target-per-group 20 --pre-roll-ms 450 --action-ms 1200 --post-roll-ms 450
uv run python training/capture_motion_sequences.py --remove-sample <sample-id>
```

The CSV schema keeps sample/session metadata, target labels, per-frame active labels, phase labels, interval start/end frame indices, raw landmarks, normalized landmarks, and derived features. Motion labels live in `training/label_schemas/motion_labels.json`.

## Static Pose Model Training

The first learned model slice uses the captured static pose CSV as a landmark-first PyTorch dataset. It implements:

- `training/label_schemas/static_pose_labels.json` for the multi-label static pose target schema.
- `training/datasets/static_pose_dataset.py` for CSV validation, tensor parsing, profiling, collation, deterministic splits, and dataloader construction.
- `hlgam/ml/models/static_pose_transformer.py` for the static pose landmark transformer.
- `training/common/losses.py` for BCE or focal multi-label loss plus auxiliary finger-count and direction losses.
- `training/jobs/train_static_pose.py` for config-driven training, metrics logging, Matplotlib plots, early stopping, and checkpointing.

Run a dry pipeline check without training:

```bash
uv run --extra ml python -m training.jobs.train_static_pose --config training/configs/static_pose_transformer.yaml --dry-run
```

Run training after collecting static pose rows:

```bash
uv run --extra ml python -m training.jobs.train_static_pose --config training/configs/static_pose_transformer.yaml
```

Run single-node DDP static-pose training across multiple GPUs:

```bash
uv run --extra ml python -m torch.distributed.run --standalone --nproc-per-node 2 --module training.jobs.train_static_pose --config training/configs/static_pose_transformer.yaml --device auto
```

Run loss-function hyperparameter grid search:

```bash
uv run --extra ml python -m training.jobs.grid_search_static_pose_loss --config training/configs/static_pose_loss_grid.yaml
```

The grid-search job expands `loss_grid` entries into static training trials. Each trial gets its own training run directory, and the grid search writes ranked `grid_search_results.json` and `grid_search_results.csv` files under `runs/static_pose_loss_grid/<timestamp>_<search-name>/`.

Plot an existing grid-search result file:

```bash
uv run --extra ml python -m training.jobs.plot_static_pose_grid_search_results --results runs/static_pose_loss_grid/<timestamp>_<search-name>/grid_search_results.json
```

The plotter writes one all-metrics chart for each varied loss-grid parameter, colours points by parameter value, marks Pareto-front trials, and exports `pareto_front_trials.json` plus `pareto_front_trials.csv` under `plots/grid_search/`.

Use CLI overrides for quick pilot searches:

```bash
uv run --extra ml python -m training.jobs.grid_search_static_pose_loss --config training/configs/static_pose_loss_grid.yaml --epochs 5 --max-trials 4 --no-plots
```

Run outputs are written under `runs/static_pose/<timestamp>_<run-name>/` with resolved config, dataset profile, split manifest, JSONL metrics, checkpoints, and plots.

During training, each epoch prints training loss, validation loss, validation macro precision, validation macro recall, validation exact match, and the configured selection metric. At completion, the trainer evaluates the held-out test split when configured and prints the same key test metrics plus macro/micro F1.

When training completes, the trainer writes two Matplotlib PNG plots under the run metrics folder:

- `training_validation_loss.png` plots training loss and validation loss over epochs.
- `validation_metrics.png` plots validation metrics such as macro/micro F1, precision, recall, exact match, and hamming loss over epochs.

Use `--no-plots` to skip plot generation for automation that only needs JSON metrics and checkpoints.

## Motion Model Training

The single-hand motion model trains on the motion capture CSV as fixed-length sliding landmark windows. It implements:

- `training/label_schemas/motion_labels.json` for action labels, phase labels, and parameter-head metadata.
- `training/datasets/motion_window_dataset.py` for sequence grouping, padded windows, sample-safe splits, phase targets, masks, and deterministic parameter targets.
- `hlgam/ml/models/motion_transformer.py` for the spatial-plus-temporal transformer architecture.
- `training.common.losses.MotionGestureLoss` for action BCE/focal loss, per-action phase loss, direction loss, and masked regression losses.
- `training/jobs/train_motion.py` for config-driven training, metrics logging, plots, early stopping, and checkpointing.

Run a dry pipeline check without training:

```bash
uv run --extra ml python -m training.jobs.train_motion --config training/configs/motion_transformer.yaml --dry-run
```

Run training after collecting motion sequence rows:

```bash
uv run --extra ml python -m training.jobs.train_motion --config training/configs/motion_transformer.yaml
```

Run single-node DDP motion training across multiple GPUs:

```bash
uv run --extra ml python -m torch.distributed.run --standalone --nproc-per-node 2 --module training.jobs.train_motion --config training/configs/motion_transformer.yaml --device auto
```

Run motion loss-function hyperparameter grid search:

```bash
uv run --extra ml python -m training.jobs.grid_search_motion_loss --config training/configs/motion_loss_grid.yaml
```

The motion grid-search job expands grouped `loss_grid` entries into repeated motion transformer training trials. It covers the primary action loss, positive-class weighting, focal-loss parameters, phase/direction auxiliary losses, and masked regression-head weights, then writes `grid_search_results.json` and `grid_search_results.csv` under `runs/motion_loss_grid/<timestamp>_<search-name>/`.

Plot motion loss-grid performance across macro recall, macro precision, macro F1, and exact match:

```bash
uv run --extra ml python -m training.jobs.plot_motion_loss_grid_search_results --results runs/motion_loss_grid/<timestamp>_<search-name>
```

The plotter accepts either the grid run directory, its `trials/` directory, or a `grid_search_results.json` file. It writes a ranked metric overview, one scatter chart per varied loss parameter, and CSV/JSON ranked trial exports under `plots/motion_loss_grid_performance/`.

Use CLI overrides for quick pilot searches:

```bash
uv run --extra ml python -m training.jobs.grid_search_motion_loss --config training/configs/motion_loss_grid.yaml --epochs 5 --max-trials 4 --limit-samples 20 --no-plots
```

Motion run outputs are written under `runs/motion/<timestamp>_<run-name>/` with resolved config, dataset profile, window split manifest, label schema, phase labels, input feature schema, checkpoints, JSONL metrics, and optional Matplotlib plots.

## Shadeform Remote GPU Training

The root-level `shadeform/` package provisions a Shadeform GPU instance, uploads the HLGAM training code, downloads the selected landmark CSV from Google Drive, installs GPU-capable PyTorch on the remote host, runs one of the existing training entrypoints, uploads run artifacts to Google Drive, downloads those artifacts back from Drive locally, and deletes the instance by default.

Create a local `shadeform.env` file with the required Shadeform and Google Drive credentials:

```env
SHADEFORM_API_KEY="..."
SHADEFORM_SSH_KEY_PATH="~/.ssh/id_ed25519"
SHADEFORM_GDRIVE_SERVICE_ACCOUNT_KEY_PATH="~/.config/gcloud/gesture-app-sa.json"
SHADEFORM_GDRIVE_RESULTS_FOLDER_ID="<results-folder-id>"
SHADEFORM_GDRIVE_STATIC_POSE_CSV_FILE_ID="<static-pose-csv-file-id>"
SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID="<motion-csv-file-id>"
```

Share the source CSV files and destination results folder with the service-account email from the JSON key. `SHADEFORM_GDRIVE_CSV_FILE_ID` can be used instead of the two model-family-specific CSV IDs when both trainers should use the same source CSV file.

Optional defaults can also be set in `shadeform.env`:

```env
SHADEFORM_GPU_TYPE="A100"
SHADEFORM_NUM_GPUS="1"
SHADEFORM_RESULTS_DIR="shadeform_results"
SHADEFORM_CLOUD=""
SHADEFORM_REGION=""
SHADEFORM_SHADE_INSTANCE_TYPE=""
```

Set `SHADEFORM_NUM_GPUS` above `1`, or pass `--num-gpus`, to request a multi-GPU instance and launch the selected trainer with `torch.distributed.run`.

Preview a static-pose remote plan without making API calls:

```bash
uv run python -m shadeform train --model-family static_pose --plan
```

Preview a motion remote plan:

```bash
uv run python -m shadeform train --model-family motion --plan --epochs 5 --no-plots
```

Preview a motion loss-grid remote plan:

```bash
uv run python -m shadeform train --model-family motion --job loss_grid --plan --epochs 5 --max-trials 4 --limit-samples 20 --no-plots
```

Run static-pose training on the cheapest currently available matching provider:

```bash
uv run python -m shadeform train --model-family static_pose --gpu-type A100 --yes
```

Run motion training with CLI overrides:

```bash
uv run python -m shadeform train --model-family motion --gpu-type A100 --yes --epochs 50 --batch-size 32
```

Run motion training on four GPUs with single-node DDP:

```bash
uv run python -m shadeform train --model-family motion --gpu-type A100 --num-gpus 4 --yes --epochs 50 --batch-size 32
```

Run motion loss-grid search on Shadeform:

```bash
uv run python -m shadeform train --model-family motion --job loss_grid --gpu-type A100 --yes --epochs 10 --max-trials 8 --limit-samples 100 --no-plots
```

The Drive IDs can also be passed directly instead of using `shadeform.env`:

```bash
uv run python -m shadeform train \
	--model-family static_pose \
	--gpu-type A100 \
	--yes \
	--gdrive-service-account-key ~/.config/gcloud/gesture-app-sa.json \
	--gdrive-csv-file-id <static-pose-csv-file-id> \
	--gdrive-results-folder-id <results-folder-id>
```

The plan uploads `pyproject.toml`, `uv.lock`, `README.md`, `hlgam/`, `training/`, and the Drive helper. It never uploads the local CSV; instead, the input CSV is downloaded on the remote host to the expected project-relative path under `data/static_poses/` or `data/motion_sequences/` before training starts. Captured image folders are intentionally not uploaded because the training datasets read landmark/features from CSV rows. Downloaded local artifacts land under `shadeform_results/<instance-name>/runs/static_pose/`, `shadeform_results/<instance-name>/runs/motion/`, or `shadeform_results/<instance-name>/runs/motion_loss_grid/` depending on the selected job.

After training, the remote uploads each return path to a run-specific Google Drive subfolder named after the Shadeform instance, then the local runner downloads those return paths from Drive into `SHADEFORM_RESULTS_DIR`. The Drive helper is invoked with `uv run --with google-api-python-client`, so Google client libraries are fetched only for the transfer command and are not added to the base project dependencies.

The remote setup intentionally runs a base `uv sync --no-dev` and then installs `matplotlib` plus GPU-capable `torch` with `uv pip install ... --torch-backend=auto`. This avoids the local `ml` extra's CPU-only Torch source while keeping the local CPU workflow unchanged.

When DDP is active, every rank trains on a `DistributedSampler` shard of the train split. Rank 0 owns resolved configs, dataset profiles, validation/test evaluation, checkpoint writes, JSONL metrics, plots, Drive uploads, and printed progress; early-stopping decisions are broadcast to the other ranks so all processes stop together.

## Static Pose Runtime Inference

Trained static pose checkpoints can be enabled in the application runtime through the top-level recognition config. Keep the learned model disabled to run only the deterministic FSM recognizers; enable it when the `ml` extra is installed and the required static-pose environment variables are set.

```yaml
recognition:
	fsm:
		enabled: true
	models:
		static_pose:
			enabled: true
			pose:
				enabled: true
			finger_count:
				enabled: true
			finger_pointing:
				enabled: true
			flat_hand:
				enabled: true
```

The runtime loads the checkpoint lazily on first use. Set these required environment variables in the process environment or `.env`:

```env
HLGAM_STATIC_POSE_MODEL_CHECKPOINT="runs/static_pose/<run>/checkpoints/best.pt"
HLGAM_STATIC_POSE_LABEL_SCHEMA="runs/static_pose/<run>/label_schema.json"
HLGAM_STATIC_POSE_FEATURE_SCHEMA="runs/static_pose/<run>/input_feature_schema.json"
HLGAM_STATIC_POSE_MODEL_DEVICE="cpu"
HLGAM_STATIC_POSE_MODEL_THRESHOLD="0.5"
HLGAM_STATIC_POSE_MODEL_MIN_HOLD_MS="80"
HLGAM_STATIC_POSE_MODEL_COOLDOWN_MS="250"
```

FSM and learned static-pose detection can run at the same time. WebSocket clients may independently toggle `recognition.fsm.enabled`, `recognition.models.static_pose.enabled`, and the static model groups `pose`, `finger_count`, `finger_pointing`, and `flat_hand`; a shorthand `models.static_pose` update is also accepted for future learned model families.

## Motion Runtime Inference

Trained motion checkpoints can be enabled as an independent learned recognizer, including when deterministic FSM recognizers are disabled.

```yaml
recognition:
	fsm:
		enabled: false
	models:
		motion:
			enabled: true
```

The runtime keeps checkpoint artifacts server-controlled and loads the model lazily on first use. Set the required artifact variables and optional runtime thresholds in the process environment or `.env`:

```env
HLGAM_MOTION_MODEL_CHECKPOINT="runs/motion/<run>/checkpoints/best.pt"
HLGAM_MOTION_LABEL_SCHEMA="runs/motion/<run>/label_schema.json"
HLGAM_MOTION_FEATURE_SCHEMA="runs/motion/<run>/input_feature_schema.json"
HLGAM_MOTION_MODEL_DEVICE="cpu"
HLGAM_MOTION_MODEL_THRESHOLD="0.5"
HLGAM_MOTION_MODEL_MIN_SEQUENCE_FRAMES="4"
HLGAM_MOTION_MODEL_MIN_HOLD_MS="0"
HLGAM_MOTION_MODEL_COOLDOWN_MS="250"
```

WebSocket clients may toggle learned motion with `recognition.models.motion.enabled` or the shorthand `models.motion.enabled` without receiving access to checkpoint paths.