import re
import ray
import copy
import numpy as np
import json
import torch
import random
from typing import List, Dict, Tuple, Any
from functools import partial, reduce

import verl.utils.torch_functional as verl_F
from verl.protocol import (
    DataProto,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.protocol import collate_fn as dataitem_collate_fn
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.model import compute_position_id_with_mask
from custom_verl.robotic.box_reward import (
    compute_score,
    compute_score_one_step,
    compute_score_one_step_ray,
)

from simulation.simenv.box1 import Box1Env
from simulation.simenv.box3d.box3d import Box3DEnv
from inference.import_utils import load_function
from prompts.prompt_utils import load_constants, build_prompt, replace_prompt

# SUCCESS_REWARD = 10.0
# INVALID_REWARD = -5.0
# STEP_PENALTY = -1.0
# SUCCESS_REWARD = 5.0

# SUCCESS_REWARD = 1.0
# INVALID_REWARD = -1.0
# STEP_PENALTY = -0.0
# FORMAT_REWARD = 0.1


class BoxRollOut:
    # This class handles the playing of an LLM against the Box environment
    def __init__(self, config, tokenizer, rollout: ActorRolloutRefWorker, gen_config):
        self.config = config
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.gen_config = gen_config
        # self.MAX_STEPS = int(SUCCESS_REWARD)
        self.MAX_STEPS = 20

        # self.prompt_file = "prompts/box-prompt-v1/dmas-think-r1-stepplan.py"
        self.prompt_file = "prompts/box-prompt-v2/dmas-think-qwen-stepplan.py"
        self.MESSAGE_TEMPLATE = load_constants(self.prompt_file)
        self.build_state_func = load_function(self.prompt_file, "Map2Text")
        self.prepare_prompt_fn = partial(
            tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        self.max_prompt_length = config.data.max_response_length
        self.truncation = "error"

        self.SUCCESS_REWARD = 1.0
        self.INVALID_REWARD = -0.0
        self.STEP_PENALTY = -0.0
        self.FORMAT_REWARD = 0.05
        self.env_cls = Box1Env

    def get_env_prompt(self, env):
        if not isinstance(env, self.env_cls):
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )
        prompt = build_prompt(
            {"mapstate": mapstate}, self.MESSAGE_TEMPLATE, self.prepare_prompt_fn
        )
        return prompt

    def generate_sequences(self, data_batch: DataProto, validation=False):
        env_configs = data_batch.non_tensor_batch["env_configs"]
        uids = data_batch.non_tensor_batch["uids"]
        uuids = data_batch.non_tensor_batch["uuid"]
        if isinstance(env_configs[0], str):
            envs = [self.env_cls.load(json.loads(x)) for x in env_configs]
        else:
            envs = [self.env_cls.load(x) for x in env_configs]
        done = torch.tensor([False] * len(data_batch))
        trajectories = [[] for _ in range(len(envs))]

        current_envs = envs
        current_indicies = [i for i in range(len(envs))]

        for cur_step in range(self.MAX_STEPS):
            if len(current_envs) == 0:  # all done
                break
            prompts = [self.get_env_prompt(env) for env in current_envs]

            # Preprocess input prompt
            input_ids = []
            attention_masks = []
            position_ids = []
            for prompt in prompts:
                input_id, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                position_id = compute_position_id_with_mask(attention_mask)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                position_ids.append(position_id)

            input_ids = torch.cat(input_ids)
            attention_mask = torch.cat(attention_masks)
            position_ids = torch.cat(position_ids)

            batch = DataProto.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            )
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"]
            )

            # Gererate sequences
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, self.rollout.world_size
            )
            output_gen_batch_padded = self.rollout.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(
                output_gen_batch_padded, pad_size=pad_size
            )

            # Simulate and calculate reward
            output_ids = output_gen_batch.batch["responses"]
            batch = batch.union(output_gen_batch)
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]

            outs = [
                compute_score_one_step(
                    out_text, cur_env, format_score=self.FORMAT_REWARD
                )
                for out_text, cur_env in zip(output_texts, current_envs)
            ]

            # Check which envs are done and update pending envs, and assign rewards
            next_envs = []
            next_indices = []
            for i, env in enumerate(current_envs):
                origin_indice = current_indicies[i]

                # Check if the env is done
                if outs[i][0] == 1:
                    done[origin_indice] = True  # Successfully done
                    reward = self.SUCCESS_REWARD + outs[i][1]
                elif outs[i][0] == 0.0:
                    done[origin_indice] = False  # not done, but still can play
                    # if this step is the last chance, and it is not done, we put invalid reward penalize
                    if cur_step == self.MAX_STEPS - 1:  # final step still not done
                        done[origin_indice] = True
                        reward = self.INVALID_REWARD + outs[i][1]
                    else:
                        reward = self.STEP_PENALTY + outs[i][1]
                        next_envs.append(env)
                        next_indices.append(origin_indice)
                elif outs[i][0] == -1:
                    done[origin_indice] = True  # Plan is invalid, penalize
                    reward = self.INVALID_REWARD + outs[i][1]

                # Build reward
                data_item = batch[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                reward_tensor = torch.zeros_like(
                    data_item.batch["responses"], dtype=torch.float32
                )
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                reward_tensor[valid_response_length - 1] = reward
                data_item.batch["token_level_scores"] = reward_tensor
                data_item.non_tensor_batch["batch_responses"] = output_texts[i]
                data_item.non_tensor_batch["batch_inputs"] = prompts[i]
                data_item.non_tensor_batch["reward_types"] = outs[i][-1]
                data_item.non_tensor_batch["uuid"] = uuids[origin_indice]
                data_item.non_tensor_batch["uid"] = uuids[origin_indice]
                data_item.batch["step_ids"] = torch.tensor(cur_step)
                data_item.batch["step_rewards"] = torch.tensor(reward)
                data_item.batch["traj_ids"] = torch.tensor(origin_indice)
                data_item.non_tensor_batch["uids"] = uids[origin_indice]
                trajectories[origin_indice].append(data_item)

            current_envs = next_envs
            current_indicies = next_indices

        # Collect reward trajectory
        for i in range(len(trajectories)):  # over unique rollout
            for x in trajectories[i]:
                x.batch["done"] = done[i]

        # Collect all trajectories
        if validation:  # we only need the last step for each entry
            print("=== Validaation Example Start ===")
            randidx = random.choice(list(range(len(trajectories))))
            for traj_step in trajectories[randidx]:
                print(traj_step.non_tensor_batch["batch_responses"])
                print(
                    "✅✅✅✅✅",
                    traj_step.non_tensor_batch["reward_types"],
                    " ✅✅✅✅✅",
                )
                print("---" * 40)
            print("=== Validation Example End ===")

            trajectories = [x[-1:] for x in trajectories]
            trajectories = reduce(
                lambda x, y: x + y,
                trajectories,
            )

        else:
            print("=== Tool Train Example Start ===")
            randidx = random.choice(list(range(len(trajectories))))
            for traj_step in trajectories[randidx]:
                print(traj_step.non_tensor_batch["batch_responses"])
                print(
                    "✅✅✅✅✅",
                    traj_step.non_tensor_batch["reward_types"],
                    " ✅✅✅✅✅",
                )
                print("---" * 40)
            print("=== Train Example End ===")
            trajectories = reduce(
                lambda x, y: x + y,
                trajectories,
            )

        flattend_trajectories = dataitem_collate_fn(trajectories)
        torch.cuda.empty_cache()
        return flattend_trajectories


class StepBoxRollOut(BoxRollOut):
    def __init__(self, config, tokenizer, rollout, gen_config):
        super().__init__(config, tokenizer, rollout, gen_config)

        # SUCCESS_REWARD, INVALID_REWARD, STEP_PENALTY, FORMAT_REWARD
        self.SUCCESS_REWARD = 5.0
        self.INVALID_REWARD = -1.0
        self.STEP_PENALTY = -0.05
        self.FORMAT_REWARD = 0.1

    def generate_sequences(self, data_batch: DataProto, validation=False):
        env_configs = data_batch.non_tensor_batch["env_configs"]
        uids = data_batch.non_tensor_batch["uids"]
        uuids = data_batch.non_tensor_batch["uuid"]
        if isinstance(env_configs[0], str):
            envs = [self.env_cls.load(json.loads(x)) for x in env_configs]
        else:
            envs = [self.env_cls.load(x) for x in env_configs]
        done = torch.tensor([False] * len(data_batch))
        trajectories = [[] for _ in range(len(envs))]

        current_envs = envs
        current_indicies = [i for i in range(len(envs))]

        for cur_step in range(self.MAX_STEPS):
            if len(current_envs) == 0:  # all done
                break
            prompts = [self.get_env_prompt(env) for env in current_envs]

            # Preprocess input prompt
            input_ids = []
            attention_masks = []
            position_ids = []
            for prompt in prompts:
                input_id, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                position_id = compute_position_id_with_mask(attention_mask)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                position_ids.append(position_id)

            input_ids = torch.cat(input_ids)
            attention_mask = torch.cat(attention_masks)
            position_ids = torch.cat(position_ids)

            batch = DataProto.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            )
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"]
            )

            # Gererate sequences
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, self.rollout.world_size
            )
            output_gen_batch_padded = self.rollout.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(
                output_gen_batch_padded, pad_size=pad_size
            )
            # Simulate and calculate reward
            output_ids = output_gen_batch.batch["responses"]
            batch = batch.union(output_gen_batch)
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]

            outs = [
                compute_score_one_step(
                    out_text, cur_env, format_score=self.FORMAT_REWARD
                )
                for out_text, cur_env in zip(output_texts, current_envs)
            ]

            # Check which envs are done and update pending envs, and assign rewards
            next_envs = []
            next_indices = []
            for i, env in enumerate(current_envs):
                origin_indice = current_indicies[i]

                # Check if the env is done
                if outs[i][0] == 1:
                    done[origin_indice] = True  # Successfully done
                    reward = self.SUCCESS_REWARD + outs[i][1]
                elif outs[i][0] == 0.0:
                    done[origin_indice] = False  # not done, but still can play
                    # if this step is the last chance, and it is not done, we put invalid reward penalize
                    if cur_step == self.MAX_STEPS - 1:  # final step still not done
                        done[origin_indice] = True
                        reward = self.INVALID_REWARD + outs[i][1]
                    else:
                        reward = self.STEP_PENALTY + outs[i][1]
                    next_envs.append(env)
                    next_indices.append(origin_indice)
                elif outs[i][0] == -1:
                    # Plan is invalid, penalize, but we still try
                    # ! THE ONLY DIFFERENCE
                    done[origin_indice] = False
                    # ! THE ONLY DIFFERENCE
                    reward = self.INVALID_REWARD + outs[i][1]

                # Build reward
                data_item = batch[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                reward_tensor = torch.zeros_like(
                    data_item.batch["responses"], dtype=torch.float32
                )
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                reward_tensor[valid_response_length - 1] = reward
                data_item.batch["token_level_scores"] = reward_tensor
                data_item.non_tensor_batch["batch_responses"] = output_texts[i]
                data_item.non_tensor_batch["batch_inputs"] = prompts[i]
                data_item.non_tensor_batch["reward_types"] = outs[i][-1]
                data_item.non_tensor_batch["uuid"] = uuids[origin_indice]
                data_item.non_tensor_batch["uid"] = uuids[origin_indice]
                data_item.batch["step_ids"] = torch.tensor(cur_step)
                data_item.batch["step_rewards"] = torch.tensor(reward)
                data_item.batch["traj_ids"] = torch.tensor(origin_indice)
                data_item.non_tensor_batch["uids"] = uids[origin_indice]
                trajectories[origin_indice].append(data_item)

            current_envs = next_envs
            current_indicies = next_indices

        # Collect reward trajectory
        for i in range(len(trajectories)):  # over unique rollout
            for x in trajectories[i]:
                x.batch["done"] = done[i]

        trajectories = [x[-1:] for x in trajectories]  # DEBUG ONLY
        # Collect all trajectories
        if validation:  # we only need the last step for each entry
            print("=== Validaation Example Start ===")
            randidx = random.choice(list(range(len(trajectories))))
            for traj_step in trajectories[randidx]:
                print(traj_step.non_tensor_batch["batch_responses"])
                print(
                    "✅✅✅✅✅",
                    traj_step.non_tensor_batch["reward_types"],
                    " ✅✅✅✅✅",
                )
                print("---" * 40)
            print("=== Validation Example End ===")

            trajectories = [x[-1:] for x in trajectories]
            trajectories = reduce(
                lambda x, y: x + y,
                trajectories,
            )

        else:
            print("=== Train Example Start ===")
            randidx = random.choice(list(range(len(trajectories))))
            for traj_step in trajectories[randidx]:
                print(traj_step.non_tensor_batch["batch_responses"])
                print(
                    "✅✅✅✅✅",
                    traj_step.non_tensor_batch["reward_types"],
                    " ✅✅✅✅✅",
                )
                print("---" * 40)
            print("=== Train Example End ===")
            trajectories = reduce(
                lambda x, y: x + y,
                trajectories,
            )

        flattend_trajectories = dataitem_collate_fn(trajectories)
        torch.cuda.empty_cache()
        return flattend_trajectories


class ToolStepBoxRollOut(BoxRollOut):
    def __init__(self, config, tokenizer, rollout, gen_config):
        super().__init__(config, tokenizer, rollout, gen_config)

        self.max_obs_length = 2048
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=self.max_prompt_length,
                max_obs_length=self.max_obs_length,
                max_start_length=self.max_prompt_length,
            )
        )
        self.max_start_length = self.max_prompt_length
        # self.OBSERVATION_TEMPLATE = """<observation>\n{mapstate}\n</observation>"""
        # self.prompt_file = "prompts/box-prompt-v2/dmas-think-qwen-stepplan.py"
        self.prompt_file = config.data.prompt_file
        self.MESSAGE_TEMPLATE = load_constants(self.prompt_file)
        self.build_state_func = load_function(self.prompt_file, "Map2Text")
        self.prepare_prompt_fn = partial(
            tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        self.OBSERVATION_TEMPLATE = """\n<observation>\n{mapstate}\n</observation>"""

        self.SUCCESS_REWARD = 1.0
        self.INVALID_REWARD = -0.0
        self.STEP_PENALTY = -0.0
        self.FORMAT_REWARD = 0.1

        self.eff_test = True
        self.env_cls = Box1Env

    def get_observation_prompt(self, env):
        if not isinstance(env, self.env_cls):
            # env = Box1Env.load(env)
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )
        prompt = replace_prompt(
            [{"content": self.OBSERVATION_TEMPLATE}],
            {"mapstate": mapstate},
        )[0]["content"]

        return prompt

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        if num_gpus <= 1, return self.rollout.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
        """
        # num_gpus = self.config.num_gpus
        num_gpus = self.rollout.world_size
        if num_gpus <= 1:
            return self.rollout.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            return self.rollout.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.rollout.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, add_special_tokens=False, return_tensors="pt", padding="longest"
        )["input_ids"]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        new_response_str = []

        for resp in responses_str:
            # if "</observation>" in resp:
            #     resp = resp.split("</observation>")[0] + "</observation>"
            if "<observation>" in resp:
                resp = resp.split("<observation>")[0]
            else:
                resp = resp
            # new_response_str.append(resp)
            new_response_str.append(resp.strip())

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _update_rolling_state(
        self,
        rollings: DataProto,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor,
    ) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(
                info.size(), pad_id, dtype=info.dtype, device=info.device
            )  # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _process_next_obs(
        self,
        next_obs: List[str],
    ) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,  # Prevents adding special tokens
        )["input_ids"]

        if next_obs_ids.shape[1] > self.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.max_obs_length}"
            )
            next_obs_ids = next_obs_ids[:, : self.max_obs_length]

        return next_obs_ids

    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None,
    ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False,
                )
            )
        else:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    pad_to_left=False,
                )
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.max_prompt_length, effective_len)

        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def execute_predictions(
        self,
        cur_step,
        response_str,
        pad_token,
        active_mask,
        envs,
        rewards,
        final_reward_types,
        gt_plans=None,
    ):
        score_refs = [None] * len(active_mask)
        active_indices = []
        if self.config.reward_model.launch_reward_fn_async:
            # print("=== ASYNC ===")
            default_res = ((-1, 0, "TimeoutError", -1), {})
            for i, active in enumerate(active_mask):
                if active:
                    # queue up the remote call
                    score_refs[i] = compute_score_one_step_ray.remote(
                        response_str[i],
                        # envs[i].to_json(),
                        envs[i].get_current_state(),
                        format_score=self.FORMAT_REWARD,
                        envcls=self.env_cls,
                    )
                    active_indices.append(i)
            # Create a list of just the active references
            active_score_refs = [score_refs[i] for i in active_indices]
            remaining_refs = active_score_refs[:]
            ready_results = [None] * len(active_score_refs)

            # Wait for all results or until timeout
            ready, not_ready = ray.wait(
                remaining_refs, num_returns=len(remaining_refs), timeout=90
            )

            # Fetch the ready results
            ready_indices = [active_score_refs.index(ref) for ref in ready]
            ready_outputs = ray.get(ready)

            # Insert ready outputs into results
            for idx, output in zip(ready_indices, ready_outputs):
                ready_results[idx] = output

            # print("TIMING: ", [x[0][-1] for x in ready_outputs])
            # Map back to the full mask
            outs = [None] * len(active_mask)
            for local_idx, global_idx in enumerate(active_indices):
                res = ready_results[local_idx]
                if res is not None:
                    envs[global_idx].reset_from_json(res[1])
                    outs[global_idx] = res[0]
                else:
                    tmp = copy.deepcopy(default_res)
                    outs[global_idx] = tmp[0]

            # outs = ray.get([score_refs[i] for i in active_indices])
            # results = ray.get([score_refs[i] for i in active_indices])
            # outs = [None] * len(active_mask)
            # for idx, res in zip(active_indices, results):
            #     outs[idx] = res
            # for idx in range(len(outs)):
            #     out = outs[idx]
            #     if out is not None:  # active, we may update env
            #         envs[idx].reset_from_json(out[1])
            #         outs[idx] = out[0]
        else:
            outs = []
            for i, active in enumerate(active_mask):
                if active:
                    out = compute_score_one_step(
                        response_str[i],
                        envs[i],
                        format_score=self.FORMAT_REWARD,
                    )
                else:
                    out = None
                outs.append(out)
            # outs = [
            #     compute_score_one_step(x, envs[i], format_score=self.FORMAT_REWARD)
            #     if active_mask[i]
            #     else None
            #     for i, x in enumerate(response_str)
            # ]

        # 3) Now walk through each slot (active or not) and fill out your outputs.
        next_obs = []
        dones = []
        valid_action = []

        j = 0  # pointer into our `outs` list
        for i, active in enumerate(active_mask):
            if not active:
                next_obs.append("")
                dones.append(True)
                valid_action.append(0)
            else:
                out = outs[i]
                # ————— your original post-processing logic —————
                if out[0] == 1:
                    done = True
                    # reward = self.SUCCESS_REWARD + out[1]
                    reward = self.SUCCESS_REWARD
                    if self.eff_test:
                        gt_step = len(gt_plans[i])
                        actual_step = cur_step + 1
                        if actual_step > gt_step:
                            tmp_r = (
                                reward - (actual_step - gt_step) * self.FORMAT_REWARD
                            )
                            tmp_r = max(tmp_r, 2 * self.FORMAT_REWARD)
                            reward = tmp_r
                    obs = ""
                elif out[0] == 0.0:
                    # continue
                    if cur_step == self.MAX_STEPS - 1:
                        done = True
                        reward = self.INVALID_REWARD + out[1]
                        obs = ""
                    else:
                        done = False
                        reward = 0.0
                        obs = self.get_observation_prompt(envs[i])
                else:
                    # invalid
                    done = True
                    reward = self.INVALID_REWARD + out[1]
                    obs = ""

                final_reward_types[i] = out[-1]
                next_obs.append(obs)
                dones.append(done)
                rewards[i] = reward
                valid_action.append(1 if not done else 0)

        return next_obs, dones, valid_action

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        meta_info: Dict,
        uids,
        rewards,
        final_reward_types,
        uuids,
    ) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]

        # Combine input IDs
        final_output["input_ids"] = torch.cat(
            [left_side["input_ids"], right_side["responses"]], dim=1
        )

        # Create attention mask and position ids
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(
                    final_output["responses_with_info_mask"]
                ),
            ],
            dim=1,
        )

        final_output["position_ids"] = self.tensor_fn.create_position_ids(
            final_output["attention_mask"]
        )
        non_tensor_batch = {}
        non_tensor_batch["uids"] = uids
        non_tensor_batch["reward_types"] = np.array(final_reward_types, object)
        non_tensor_batch["batch_inputs"] = self.tokenizer.batch_decode(
            final_output["input_ids"].long(), skip_special_tokens=True
        )
        non_tensor_batch["batch_responses"] = self.tokenizer.batch_decode(
            final_output["responses"].long(), skip_special_tokens=True
        )
        non_tensor_batch["uuid"] = uuids
        non_tensor_batch["uid"] = uuids
        reward_tensor = torch.zeros_like(final_output["responses"], dtype=torch.float32)
        for i in range(len(final_output["responses"])):
            prompt_ids = final_output["prompts"][i]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = final_output["attention_mask"][prompt_length:].sum()
            reward_tensor[i, valid_response_length - 1] = rewards[i]
        final_output["token_level_scores"] = reward_tensor

        final_output = DataProto.from_dict(final_output, non_tensors=non_tensor_batch)
        final_output.meta_info.update(meta_info)

        return final_output

    def generate_sequences(self, data_batch, validation=False):
        env_configs = data_batch.non_tensor_batch["env_configs"]
        if isinstance(env_configs[0], str):
            env_configs = [json.loads(x) for x in env_configs]

        uids = data_batch.non_tensor_batch["uids"]
        uuids = data_batch.non_tensor_batch["uuid"]
        gt_plans = [x["gt_plan"] for x in env_configs]

        envs = [self.env_cls.load(x) for x in env_configs]

        active_mask = torch.ones(
            data_batch.batch["input_ids"].shape[0], dtype=torch.bool
        )
        turns_stats = torch.ones(
            data_batch.batch["input_ids"].shape[0], dtype=torch.int
        )
        valid_action_stats = torch.zeros(
            data_batch.batch["input_ids"].shape[0], dtype=torch.int
        )
        active_num_list = [active_mask.sum().item()]
        rewards = torch.zeros(
            data_batch.batch["input_ids"].shape[0], dtype=torch.float32
        )
        final_reward_types = [
            _ for _ in range(data_batch.batch["input_ids"].shape[0])
        ]  #! track final states

        rollings = data_batch

        initial_input_ids = data_batch.batch["input_ids"].clone().long()
        original_left_side = {
            "input_ids": initial_input_ids[:, -self.max_start_length :]
        }
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        for cur_step in range(self.MAX_STEPS):
            if not active_mask.sum():
                break

            # Cut length?
            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()}
            )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, response_str = self._postprocess_responses(
                gen_output.batch["responses"]
            )
            responses_ids, response_str = self.tensor_fn._example_level_pad(
                responses_ids, response_str, active_mask
            )

            next_obs, dones, valid_action = self.execute_predictions(
                cur_step,
                response_str,
                self.tokenizer.pad_token,
                active_mask,
                envs,
                rewards,
                final_reward_types,
                gt_plans=gt_plans,
            )

            curr_active_mask = torch.tensor(
                [not done for done in dones], dtype=torch.bool
            )
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Update states
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )

        final_out = self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            uids,
            rewards,
            final_reward_types,
            uuids,
        )

        if validation:
            stage = "Validation"
        else:
            stage = "Train"

        print(f"=== {stage} Example Start ===")
        randindexes = random.sample(list(range(len(final_out.batch["responses"]))), k=2)
        # randidx = random.choice(list(range(len(final_out.batch["responses"]))))
        for randidx in randindexes:
            print("🟦" * 50)
            print(final_out.non_tensor_batch["batch_responses"][randidx])
            print()
            print(
                "✅" * 50,
                final_out.non_tensor_batch["reward_types"][randidx],
                final_out.batch["token_level_scores"][randidx].sum(),
                "✅" * 50,
            )
            print("---" * 40)
        print(f"=== {stage} Example End ===")

        for k in [
            "input_ids",
            "attention_mask",
            "info_mask",
            "position_ids",
            "responses",
            "responses_with_info_mask",
            "prompts",
        ]:
            final_out.batch[k] = final_out.batch[k].long()
        return final_out


class ToolStepBox3dRollOut(ToolStepBoxRollOut):
    def __init__(self, config, tokenizer, rollout, gen_config):
        super().__init__(config, tokenizer, rollout, gen_config)
        self.env_cls = Box3DEnv
        self.build_state_func = load_function(self.prompt_file, "describe_obs")
        self.MAX_STEPS = 10  # maximum

    def get_observation_prompt(self, env):
        if not isinstance(env, self.env_cls):
            # env = Box1Env.load(env)
            env = self.env_cls.load(env)

        obs = env.obs
        targets = env.targets
        mapstate = self.build_state_func(obs, env.targets)
        prompt = replace_prompt(
            [{"content": self.OBSERVATION_TEMPLATE}],
            {"mapstate": mapstate},
        )[0]["content"]

        return prompt


import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cut_left: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict["attention_mask"].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(
        self, tensor: torch.Tensor, pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = (
            tensor != self.config.pad_token_id
            if pad_to_left
            else tensor == self.config.pad_token_id
        )
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor], pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(
        self,
        responses: torch.Tensor,
        responses_str: List[str],
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        # Create masked response strings
        padded_responses_str = [""] * batch_size

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1

        return padded_responses, padded_responses_str
