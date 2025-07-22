import os
import ray
import uuid
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint
from omegaconf import OmegaConf, open_dict

from verl.protocol import (
    DataProto,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    # drop_dataproto_to_divisor,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    AdvantageEstimator,
    _timer,
    reduce_metrics,
    compute_timing_metrics,
    compute_data_metrics,
    compute_advantage,
    apply_kl_penalty,
    compute_throughout_metrics,
    Role,
    WorkerType,
    ResourcePoolManager,
    RayWorkerGroup,
)
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.reward import compute_reward, compute_reward_async

from custom_verl.custom_dataset import (
    HFRLHFDataset,
    DynamicPromptRLHFDataset,
    DynamicRationalCodeRLHFDataset,
    DynamicTestCaseCodeRLHFDataset,
    CountdownRLHFDataset,
    BoxRLHFDataset,
    Box3DRLHFDataset,
)
from custom_verl.robotic.box_rollout import (
    BoxRollOut,
    StepBoxRollOut,
    ToolStepBoxRollOut,
    ToolStepBox3dRollOut,
)

from custom_verl.callback import CallbackManager


class CustomRayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # TODO (jiabao): change to register
        if config.data.data_class == "hfrlhf":
            self.dataset_cls = HFRLHFDataset
        elif config.data.data_class == "dynamicprompt":
            self.dataset_cls = DynamicPromptRLHFDataset
        elif config.data.data_class == "dynamicrationalcode":
            self.dataset_cls = DynamicRationalCodeRLHFDataset
        elif config.data.data_class == "countdown":
            self.dataset_cls = CountdownRLHFDataset
        elif config.data.data_class == "dynamictestcase":
            self.dataset_cls = DynamicTestCaseCodeRLHFDataset
        elif config.data.data_class == "box":
            self.dataset_cls = BoxRLHFDataset
        elif config.data.data_class == "box3d":
            self.dataset_cls = Box3DRLHFDataset
        else:
            self.dataset_cls = RLHFDataset
            print("Using default dataset class")

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
        )

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = self.dataset_cls(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            prompt_file=self.config.data.get("prompt_file", None),
        )
        assert self.train_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), (
            f"dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get('truncation', 'error')}"
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = self.dataset_cls(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            prompt_file=self.config.data.get("prompt_file", None),
        )
        assert self.val_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), (
            f"dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get('truncation', 'error')}"
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1, (
            "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        self._callback_manager = CallbackManager(
            logger, rootdir=self.config.trainer.default_local_dir
        )

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                    batch.non_tensor_batch["uuid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        if self.config.reward_model.launch_reward_fn_async:
                            future_val_res = compute_reward_async.remote(
                                batch,
                                self.config,
                                self.tokenizer,
                                no_format_score=False,
                            )
                            # val_res = compute_scoroe_self.reward_fn(batch)
                        else:
                            val_res = self.reward_fn(batch)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            val_res = ray.get(future_val_res)

                        batch = batch.union(val_res)
                        val_res.non_tensor_batch["uuid"] = batch.non_tensor_batch[
                            "uuid"
                        ]
                        self._callback_manager.on_generate_sequences(
                            self.global_steps, val_res, "train"
                        )

                        # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.get(
                        #     "use_kl_loss", False
                        # ):
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )
                        if os.environ.get("DEBUG_VERL", "0") == "1":
                            print("=== Start ☁️☁️☁️ ===")
                            for reward_type, token_level_reward, value, ret in zip(
                                batch.non_tensor_batch["reward_types"],
                                batch.batch["token_level_rewards"].sum(-1),
                                batch.batch.get("values", [None] * len(batch.batch)),
                                batch.batch["returns"],
                            ):
                                print(
                                    f"{reward_type}\t{token_level_reward}\t{value}\t{ret}"
                                )
                            print("=== End ☁️☁️☁️ ===")

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return

                self.global_steps += 1

                #! End training
                def end_training():
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if (
                        self.config.trainer.save_freq > 0
                        and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return

                if self.global_steps >= self.total_training_steps:
                    end_training()

                if self.config.trainer.get("max_steps", -1) != -1:
                    if self.global_steps >= self.config.trainer.max_steps:
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and self.global_steps % self.config.trainer.test_freq != 0
                        ) or (
                            self.config.trainer.save_freq > 0
                            and self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            end_training()

                        return

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            test_batch.non_tensor_batch["uuid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                dtype=object,
            )

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            val_res = self.val_reward_fn(test_batch)
            # add inputs and score to it
            val_res.non_tensor_batch["uuid"] = test_batch.non_tensor_batch["uuid"]
            val_res.non_tensor_batch["batch_inputs"] = np.array(
                input_texts, dtype=object
            )
            val_res.non_tensor_batch["scores"] = np.array(
                (val_res.batch["token_level_scores"].sum(-1).cpu().tolist())
            )
            val_res.non_tensor_batch["data_source"] = test_batch.non_tensor_batch.get(
                "data_source",
                np.array(
                    ["unknown"] * val_res.batch["token_level_scores"].shape[0],
                    dtype=object,
                ),
            )

            self._callback_manager.on_generate_sequences(
                self.global_steps, val_res, "val"
            )
            reward_tensor = val_res.batch["token_level_scores"]

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._callback_manager.on_val_end(self.global_steps, val_res, "val")
        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict


# TODO(jiabao): Implement the step-by-step simulation training
class BoxRayPPOTrainer(CustomRayPPOTrainer):
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.get("stepbox", False):
            self.boxenv_rollout = StepBoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
            reduce_traj = True
        elif self.config.trainer.get("toolbox", False):
            self.boxenv_rollout = ToolStepBoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
            reduce_traj = False
        elif self.config.trainer.get("toolbox3d", False):
            self.boxenv_rollout = ToolStepBox3dRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
            reduce_traj = False
        else:
            self.boxenv_rollout = BoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
            reduce_traj = True

        self._callback_manager = CallbackManager(
            logger,
            rootdir=self.config.trainer.default_local_dir,
            reduce_traj=reduce_traj,
        )
        self.actor_rollout_wg.set_sampling_param(
            # {"n": 1, "stop": "<observation>"}
            {
                "n": 1,
            }
        )  # we only sample one response per traj

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uuid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "env_configs",
                            "uids",
                            "uuid",
                        ],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # repeat to align with repeated responses in rollout
                        # Repeat first
                        gen_batch_output = self.boxenv_rollout.generate_sequences(
                            gen_batch
                        )

                    batch = gen_batch_output
                    batch_padded, pad_size = pad_dataproto_to_divisor(
                        # ! DEBUGGGG
                        batch,
                        self.actor_rollout_wg.world_size,
                    )

                    # compute global_valid tokens
                    batch_padded.meta_info["global_token_num"] = torch.sum(
                        batch_padded.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(
                            batch_padded
                        )
                        batch_padded = batch_padded.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch_padded
                            )
                            batch_padded = batch_padded.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch_padded)
                            batch_padded = batch_padded.union(values)

                    with _timer("adv", timing_raw):
                        self._callback_manager.on_generate_sequences(
                            self.global_steps, batch_padded, "train"
                        )

                        # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.get(
                        #     "use_kl_loss", False
                        # ):
                        if self.config.algorithm.use_kl_in_reward:
                            batch_padded, kl_metrics = apply_kl_penalty(
                                batch_padded,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch_padded.batch["token_level_rewards"] = (
                                batch_padded.batch["token_level_scores"]
                            )

                        # compute advantages, executed on the driver process
                        batch = unpad_dataproto(batch_padded, pad_size=pad_size)
                        batch = DataProto.concat([batch])  # make dataitem to dataproto
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                        if os.environ.get("DEBUG_VERL", "0") == "1":
                            print("=== Start ☁️☁️☁️ ===")
                            for reward_type, token_level_reward, value, ret in zip(
                                batch.non_tensor_batch["reward_types"],
                                batch.batch["token_level_rewards"].sum(-1),
                                batch.batch.get("values", [None] * len(batch.batch)),
                                batch.batch["returns"],
                            ):
                                print(
                                    f"{reward_type}\t{token_level_reward}\t{value}\t{ret}"
                                )
                            print("=== End ☁️☁️☁️ ===")

                        batch_padded, pad_size = pad_dataproto_to_divisor(
                            batch, self.actor_rollout_wg.world_size
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch_padded)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch_padded
                            )
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    batch = unpad_dataproto(batch_padded, pad_size=pad_size)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return

                self.global_steps += 1

                #! End training
                def end_training():
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if (
                        self.config.trainer.save_freq > 0
                        and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint(end=True)
                    return

                if self.global_steps >= self.total_training_steps:
                    end_training()

                if self.config.trainer.get("max_steps", -1) != -1:
                    if self.global_steps >= self.config.trainer.max_steps:
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and self.global_steps % self.config.trainer.test_freq != 0
                        ) or (
                            self.config.trainer.save_freq > 0
                            and self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            end_training()

                        return

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        if self.config.trainer.get("stepbox", False):
            self.boxenv_rollout = StepBoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
        elif self.config.trainer.get("toolbox", False):
            self.boxenv_rollout = ToolStepBoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
        elif self.config.trainer.get("toolbox3d", False):
            self.boxenv_rollout = ToolStepBox3dRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )
            reduce_traj = False
        else:
            self.boxenv_rollout = BoxRollOut(
                config=self.config,
                tokenizer=self.tokenizer,
                rollout=self.actor_rollout_wg,
                gen_config={},
            )

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uuid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                dtype=object,
            )
            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "env_configs",
                    "uids",
                ],
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            test_gen_batch.non_tensor_batch["uuid"] = test_batch.non_tensor_batch[
                "uuid"
            ]

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )

            test_output_gen_batch_padded = self.boxenv_rollout.generate_sequences(
                test_gen_batch_padded, validation=True
            )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            self._callback_manager.on_generate_sequences(
                self.global_steps, test_output_gen_batch, "val"
            )
            reward_tensor = test_output_gen_batch.batch["token_level_scores"]

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._callback_manager.on_val_end(
            self.global_steps, test_output_gen_batch, "val"
        )
        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict


# Ray wrap functions
@ray.remote
def compute_log_prob_remote(actor, batch):
    return actor.compute_log_prob(batch)


@ray.remote
def compute_ref_log_prob_remote(ref_policy, batch):
    return ref_policy.compute_ref_log_prob(batch)


@ray.remote
def compute_values_remote(critic, batch):
    return critic.compute_values(batch)


@ray.remote
def reward_fn_remote(fn, batch):
    return fn(batch)


@ray.remote
def compute_rm_remote(rm_wg, batch):
    return rm_wg.compute_rm_score(batch)
