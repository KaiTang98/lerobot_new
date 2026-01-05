#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note: Do NOT enable postponed annotations here.
`parser.wrap()` introspects the first argument's type at runtime.
With `from __future__ import annotations`, annotations become strings and
the parser receives a non-dataclass type, causing a TypeError.
"""

import logging
import warnings
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
	get_step_checkpoint_dir,
	get_step_identifier,
	load_training_state,
	save_checkpoint,
	update_last_checkpoint,
)
from lerobot.utils.utils import (
	format_big_number,
	has_method,
	init_logging,
)

# Optional dataset wrapper to keep only a subset of dims
from lerobot.scripts.dataset_processor.subset_dataset import SubsetStateActionDataset
# from tests.rl.test_actor_learner import cfg


def update_policy(
	train_metrics: MetricsTracker,
	policy: PreTrainedPolicy,
	batch: Any,
	optimizer: Optimizer,
	grad_clip_norm: float,
	accelerator: Accelerator,
	lr_scheduler=None,
	lock=None,
) -> tuple[MetricsTracker, dict]:
	start_time = time.perf_counter()
	policy.train()

	with accelerator.autocast():
		loss, output_dict = policy.forward(batch)

	accelerator.backward(loss)

	if grad_clip_norm > 0:
		grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
	else:
		grad_norm = torch.nn.utils.clip_grad_norm_(
			policy.parameters(), float("inf"), error_if_nonfinite=False
		)

	with lock if lock is not None else nullcontext():
		optimizer.step()

	optimizer.zero_grad()

	if lr_scheduler is not None:
		lr_scheduler.step()

	if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
		accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

	train_metrics.loss = loss.item()
	train_metrics.grad_norm = grad_norm.item()
	train_metrics.lr = optimizer.param_groups[0]["lr"]
	train_metrics.update_s = time.perf_counter() - start_time
	return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
	"""Same as lerobot_train.py but with an optional dataset subset wrapper.

	Enable by setting cfg.dataset.subset_state_names and/or cfg.dataset.subset_action_names
	(lists of feature names). If provided, the dataset will be wrapped with SubsetStateActionDataset
	before policy/processors are constructed, and meta/features/stats will be updated accordingly.
	"""
	cfg.validate()

	if accelerator is None:
		from accelerate.utils import DistributedDataParallelKwargs

		ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

	init_logging(accelerator=accelerator)
	# Suppress torchvision video deprecation warnings
	warnings.filterwarnings(
		"ignore",
		message=(
			"The video decoding and encoding capabilities of torchvision are deprecated"
		),
		category=UserWarning,
	)
	is_main_process = accelerator.is_main_process

	if is_main_process:
		logging.info(pformat(cfg.to_dict()))

	wandb_logger = None
	if cfg.wandb.enable and cfg.wandb.project and is_main_process:
		wandb_logger = WandBLogger(cfg)
	else:
		if is_main_process:
			logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

	if cfg.seed is not None:
		set_seed(cfg.seed, accelerator=accelerator)

	device = accelerator.device
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = True

	# Dataset creation (main process first to avoid race)
	if is_main_process:
		logging.info("Creating dataset")
		dataset = make_dataset(cfg)
	accelerator.wait_for_everyone()
	if not is_main_process:
		dataset = make_dataset(cfg)

	# Optional subset wrapping
	# Optional: load subset names from a JSON config file to avoid CLI verbosity
	subset_config_path = getattr(cfg.dataset, "subset_config_path", None)
	subset_state = getattr(cfg.dataset, "subset_state_names", None)
	subset_action = getattr(cfg.dataset, "subset_action_names", None)
	if subset_config_path:
		import json
		from pathlib import Path
		p = Path(subset_config_path)
		if p.exists():
			try:
				payload = json.loads(p.read_text())
				# Expect keys: {"subset_state_names": [...], "subset_action_names": [...]}
				subset_state = payload.get("subset_state_names", subset_state)
				subset_action = payload.get("subset_action_names", subset_action)
				logging.info(f"Loaded subset config from {p}")
			except Exception as e:
				logging.warning(f"Failed to load subset config from {p}: {e}")
	if subset_state or subset_action:
		# If only one side is provided, default the other to current full names
		full_state_names = dataset.meta.features["observation.state"]["names"]
		full_action_names = dataset.meta.features["action"]["names"]
		state_keep = subset_state if subset_state else full_state_names
		action_keep = subset_action if subset_action else full_action_names
		dataset = SubsetStateActionDataset(dataset, state_keep, action_keep)

	# Evaluation env only for sim configs
	eval_env = None
	if cfg.eval_freq > 0 and cfg.env is not None:
		if is_main_process:
			logging.info("Creating env")
		eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

	if is_main_process:
		logging.info("Creating policy")
	policy = make_policy(
		cfg=cfg.policy,
		ds_meta=dataset.meta,
		rename_map=cfg.rename_map,
	)

	accelerator.wait_for_everyone()

	processor_kwargs = {}
	postprocessor_kwargs = {}
	if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
		processor_kwargs["dataset_stats"] = dataset.meta.stats

	if cfg.policy.pretrained_path is not None:
		processor_kwargs["preprocessor_overrides"] = {
			"device_processor": {"device": device.type},
			"normalizer_processor": {
				"stats": dataset.meta.stats,
				"features": {**policy.config.input_features, **policy.config.output_features},
				"norm_map": policy.config.normalization_mapping,
			},
			"rename_observations_processor": {"rename_map": cfg.rename_map},
		}
		postprocessor_kwargs["postprocessor_overrides"] = {
			"unnormalizer_processor": {
				"stats": dataset.meta.stats,
				"features": policy.config.output_features,
				"norm_map": policy.config.normalization_mapping,
			},
		}

	preprocessor, postprocessor = make_pre_post_processors(
		policy_cfg=cfg.policy,
		pretrained_path=cfg.policy.pretrained_path,
		**processor_kwargs,
		**postprocessor_kwargs,
	)

	if is_main_process:
		logging.info("Creating optimizer and scheduler")
	optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

	step = 0
	if cfg.resume:
		step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

	num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
	num_total_params = sum(p.numel() for p in policy.parameters())

	if is_main_process:
		logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
		if cfg.env is not None:
			logging.info(f"{cfg.env.task=}")
		logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
		logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
		logging.info(f"{dataset.num_episodes=}")
		num_processes = accelerator.num_processes
		effective_bs = cfg.batch_size * num_processes
		logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
		logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
		logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

	if hasattr(cfg.policy, "drop_n_last_frames"):
		shuffle = False
		sampler = EpisodeAwareSampler(
			dataset.meta.episodes["dataset_from_index"],
			dataset.meta.episodes["dataset_to_index"],
			drop_n_last_frames=cfg.policy.drop_n_last_frames,
			shuffle=True,
		)
	else:
		shuffle = True
		sampler = None

	dataloader = torch.utils.data.DataLoader(
		dataset,
		num_workers=cfg.num_workers,
		batch_size=cfg.batch_size,
		shuffle=shuffle and not cfg.dataset.streaming,
		sampler=sampler,
		pin_memory=device.type == "cuda",
		drop_last=False,
		prefetch_factor=2 if cfg.num_workers > 0 else None,
	)

	accelerator.wait_for_everyone()
	policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
		policy, optimizer, dataloader, lr_scheduler
	)
	dl_iter = cycle(dataloader)

	policy.train()

	train_metrics = {
		"loss": AverageMeter("loss", ":.3f"),
		"grad_norm": AverageMeter("grdn", ":.3f"),
		"lr": AverageMeter("lr", ":0.1e"),
		"update_s": AverageMeter("updt_s", ":.3f"),
		"dataloading_s": AverageMeter("data_s", ":.3f"),
	}

	effective_batch_size = cfg.batch_size * accelerator.num_processes
	train_tracker = MetricsTracker(
		effective_batch_size,
		dataset.num_frames,
		dataset.num_episodes,
		train_metrics,
		initial_step=step,
		accelerator=accelerator,
	)

	if is_main_process:
		logging.info("Start offline training on a fixed dataset")

	for _ in range(step, cfg.steps):
		start_time = time.perf_counter()
		batch = next(dl_iter)
		batch = preprocessor(batch)
		train_tracker.dataloading_s = time.perf_counter() - start_time

		train_tracker, output_dict = update_policy(
			train_tracker,
			policy,
			batch,
			optimizer,
			cfg.optimizer.grad_clip_norm,
			accelerator=accelerator,
			lr_scheduler=lr_scheduler,
		)

		step += 1
		train_tracker.step()
		is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
		is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
		is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

		if is_log_step:
			logging.info(train_tracker)
			if wandb_logger:
				wandb_log_dict = train_tracker.to_dict()
				if output_dict:
					wandb_log_dict.update(output_dict)
				wandb_logger.log_dict(wandb_log_dict, step)
			train_tracker.reset_averages()

		if cfg.save_checkpoint and is_saving_step:
			if is_main_process:
				logging.info(f"Checkpoint policy after step {step}")
				checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
				save_checkpoint(
					checkpoint_dir=checkpoint_dir,
					step=step,
					cfg=cfg,
					policy=accelerator.unwrap_model(policy),
					optimizer=optimizer,
					scheduler=lr_scheduler,
					preprocessor=preprocessor,
					postprocessor=postprocessor,
				)
				update_last_checkpoint(checkpoint_dir)
				if wandb_logger:
					wandb_logger.log_policy(checkpoint_dir)

			accelerator.wait_for_everyone()

		if cfg.env and is_eval_step:
			if is_main_process:
				eval_policy_all(
					cfg,
					policy,
					dataset,
					eval_env,
					accelerator,
					step_identifier=get_step_identifier(cfg.steps, step),
				)
			accelerator.wait_for_everyone()

	if eval_env:
		close_envs(eval_env)

	if is_main_process:
		logging.info("End of training")

	accelerator.wait_for_everyone()
	accelerator.end_training()


def main():
	train()


if __name__ == "__main__":
	main()
