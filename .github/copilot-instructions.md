# AI coding agent guide for this repo

Use this as your fast-start map for working productively in this codebase.

## Big picture architecture
- Library: PyTorch-based robotics stack with datasets, policies, environments, processors, and robot I/O.
- Key packages (under `src/lerobot/`):
  - `datasets/`: `LeRobotDataset` format, HF Hub integration, sampling utils.
  - `policies/`: policy families (ACT, Diffusion, TDMPC, VQ-BeT), `factory.py` to construct policies and their processors.
  - `processor/`: processing primitives and pipelines. Core abstraction:
    - `EnvTransition` (typed dict carrying observation/action/reward/etc.)
    - `ProcessorStep` units chained into `PolicyProcessorPipeline` (batched tensors) and `RobotProcessorPipeline` (unbatched robot dicts). See `docs/source/introduction_processors.mdx`.
    - Converters in `lerobot.processor.converters` bridge robot dicts ⇄ transitions ⇄ batch tensors.
  - `envs/`: gymnasium environments and utilities for evaluation/simulation.
  - `robots/`, `motors/`, `cameras/`, `teleoperators/`: hardware integration layers.
  - `scripts/`: CLI entrypoints for training/eval/record/replay, etc.
  - `configs/`: Draccus-backed config dataclasses and parser (`lerobot.configs.parser`).
- Central training flow (`scripts/lerobot_train.py`):
  1) Parse `TrainPipelineConfig` via Draccus
  2) `make_dataset(cfg)` loads dataset and exposes `dataset.meta.stats`
  3) `make_policy(cfg.policy, ds_meta, rename_map)` constructs model
  4) `make_pre_post_processors(...)` builds pre/post pipelines (normalization uses `dataset.meta.stats`)
  5) Data loop uses `accelerate.Accelerator`, logs, checkpoints, optional eval via `envs`

## Project conventions and patterns
- Keep registry/availability lists in `src/lerobot/__init__.py` in sync when adding:
  - environments (`available_tasks_per_env`), datasets (`available_datasets_per_env`), policies (`available_policies`, `available_policies_per_env`)
  - tests reference these (see `tests/test_available.py`).
- Policies must expose a stable `name` class attribute consistent with `available_policies`.
- Processor contracts: steps can modify feature shapes/types; declare via `transform_features()` to keep dataset schemas correct. Feature specs live in `lerobot.configs.types` (`PolicyFeature`, `FeatureType`, `PipelineFeatureType`).
- Pre/Post processors are created from policy config and optionally a pretrained path; normalization stats come from the dataset and `policy.config.normalization_mapping`.
- CLIs are defined in `pyproject.toml` `[project.scripts]` (e.g., `lerobot-train`, `lerobot-eval`, `lerobot-dataset-viz`, `lerobot-record`, `lerobot-replay`, etc.).

## Developer workflows
- Runtime requirements: Python 3.10+, PyTorch 2.2+, ffmpeg presence for video. Optional extras for specific envs/hardware (see `pyproject.toml` optional-dependencies).
- Install for development:
  - Editable install: `pip install -e .` and optionally extras like `pip install -e ".[aloha,pusht]"`.
- Lint/Type:
  - Ruff configured via `pyproject.toml` (`line-length=110`, selected rules E/W/F/I/B/C4/UP/SIM; ignores include E501). Mypy is enabled gradually; many modules are ignored except selected subpackages.
- Tests:
  - Unit tests use pytest (see `tests/`); minimal smoke tests live in `tests/test_available.py`.
  - End-to-end smoke training/eval flows are scripted in the `Makefile` as small runs (CPU/GPU selectable via `DEVICE`), e.g. `test-act-ete-train`, `test-diffusion-ete-eval`, `test-smolvla-ete-*`.
- Training/Eval examples (from `README.md` and `scripts/lerobot_train.py`):
  - Reproduce SOTA example: `lerobot-train --config_path=lerobot/diffusion_pusht`
  - Direct flags (short E2E used in Makefile):
    - Train ACT on ALOHA sim: `lerobot-train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human ...`
    - Evaluate a checkpoint: `lerobot-eval --policy.path=tests/outputs/act/checkpoints/000004/pretrained_model --env.type=aloha ...`
- Dataset tooling:
  - Visualize: `lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0`
  - Dataset format and temporal slicing are described in `README.md`.
- Docker: user/internal Dockerfiles under `docker/` for building images.

## Integration points and external deps
- HF Hub: models and datasets are hosted under `https://huggingface.co/lerobot`; `huggingface-hub` is a core dep. Policies can push to hub via `PreTrainedPolicy.push_model_to_hub` and the training script pushes processors if configured.
- Accelerate: distributed/mixed precision handled via `accelerate.Accelerator` in training; evaluation supports vectorized envs.
- Gymnasium-based envs: optional extras (`aloha`, `pusht`, etc.) provide environment packages like `gym-aloha`, `gym-pusht`.
- Visualization: `rerun-sdk` used for dataset viz.
- Experiment tracking: optional Weights & Biases via `lerobot.rl.wandb_utils.WandBLogger` and `cfg.wandb.*`.

## Common code patterns and examples
- Building processors for a pretrained policy:
  ```python
  from lerobot.policies.factory import make_policy, make_pre_post_processors
  ds = make_dataset(cfg)  # exposes ds.meta.stats
  policy = make_policy(cfg.policy, ds_meta=ds.meta, rename_map=cfg.rename_map)
  pre, post = make_pre_post_processors(
      policy_cfg=cfg.policy,
      pretrained_path=cfg.policy.pretrained_path,
      preprocessor_overrides={
          "device_processor": {"device": "cuda"},
          "normalizer_processor": {
              "stats": ds.meta.stats,
              "features": {**policy.config.input_features, **policy.config.output_features},
              "norm_map": policy.config.normalization_mapping,
          },
          "rename_observations_processor": {"rename_map": cfg.rename_map},
      },
      postprocessor_overrides={
          "unnormalizer_processor": {
              "stats": ds.meta.stats,
              "features": policy.config.output_features,
              "norm_map": policy.config.normalization_mapping,
          },
      },
  )
  ```
- Keep availability maps updated when you add new envs/policies/datasets and mirror changes in `tests/test_available.py`.

## Where to look first
- High-level usage and install: `README.md`, `docs/source/*.mdx`
- CLI training flow: `src/lerobot/scripts/lerobot_train.py`
- Policy creation and processors: `src/lerobot/policies/factory.py`, `src/lerobot/processor/*`
- Availability registries: `src/lerobot/__init__.py`
- Test patterns and fixtures: `tests/`

If any of the above feels ambiguous for your task (e.g., missing commands for a new policy or unclear processor contracts), leave a short note in your PR and request clarification; we’ll update this guide promptly.
