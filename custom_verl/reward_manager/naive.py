# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from functools import partial
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from custom_verl.reward_utils import RewardType

from custom_verl.compute_score import compute_score_dict


class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        configs=None,
        no_format_score=False,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        compute_score_fn = compute_score or _default_compute_score
        if (configs is not None) and (configs["enable"]):
            # NOTE: we add configs that manipulates the compute_score function
            self.compute_score = partial(
                compute_score_fn, configs=configs, no_format_score=no_format_score
            )
            print("We're using the custom compute_score function")
        else:
            self.compute_score = partial(
                compute_score_fn, no_format_score=no_format_score
            )

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_responses = [None] * len(data)
        reward_types = [None] * len(data)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # sequences = valid_response_ids
            valid_response_ids = torch.cat((valid_prompt_ids[-10:], valid_response_ids))
            response_str = self.tokenizer.decode(valid_response_ids)
            batch_responses[i] = response_str

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch["data_source"]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score_res = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_types[i] = score_res[1]
            score = score_res[0]

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(response_str)

        reward_tensor = {
            "token_level_scores": reward_tensor,
            "batch_responses": np.array(batch_responses, dtype=object),
            "reward_types": np.array(reward_types, dtype=object),
        }
        for k in ["level", "uid"]:
            if k in data.non_tensor_batch:
                reward_tensor[k] = data.non_tensor_batch[k]

        return DataProto.from_single_dict(reward_tensor)


class NaiveRewardManagerDict:
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        configs=None,
        no_format_score=False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        compute_score_fn = compute_score or compute_score_dict
        if (configs is not None) and (configs["enable"]):
            # NOTE: we add configs that manipulates the compute_score function
            self.compute_score = partial(
                compute_score_fn, configs=configs, no_format_score=no_format_score
            )
            print("We're using the custom compute_score function")
        else:
            self.compute_score = partial(
                compute_score_fn, no_format_score=no_format_score
            )

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_responses = [None] * len(data)
        reward_types = [None] * len(data)
        traj_lens = [None] * len(data)
        parallelism = [None] * len(data)
        gt_traj_lens = [None] * len(data)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            valid_response_ids = torch.cat((valid_prompt_ids[-10:], valid_response_ids))
            response_str = self.tokenizer.decode(valid_response_ids)
            batch_responses[i] = response_str

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            score_res = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_types[i] = score_res["reward_type"]
            score = score_res["reward"]
            traj_len = score_res.get("traj_len", -1)
            traj_lens[i] = traj_len
            gt_traj_len = score_res.get("gt-traj_len", -1)
            gt_traj_lens[i] = gt_traj_len
            parallelism[i] = score_res.get("parallelism", -1)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("=== Start ☁️☁️☁️ ===")
                print(f"Data source: {data_source}")
                print(response_str)
                print(f"Traj len: {traj_len} Reward: {reward_types[i]}")
                print("=== End ☁️☁️☁️ ===")

        reward_tensor = {
            "token_level_scores": reward_tensor,
            "batch_responses": np.array(batch_responses, dtype=object),
            "reward_types": np.array(reward_types, dtype=object),
            "traj_len": np.array(traj_lens),
            "parallelism": np.array(parallelism),
            "gt-traj_len": np.array(gt_traj_lens),
        }
        for k in ["level", "uid"]:
            if k in data.non_tensor_batch:
                reward_tensor[k] = data.non_tensor_batch[k]

        return DataProto.from_single_dict(reward_tensor)


# 2) Wrap your compute_score function as a Ray remote
#    If you need to pass configs/no_format_score through partial,
#    wrap the *underlying* fn, not the partial itself.
import ray


def _compute_score_fn(
    compute_score,
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


# Adapted from https://github.com/huggingface/open-r1/blob/d436b7b9c0e9205a2d329596273ca0600a794f54/src/open_r1/rewards.py#L70
def format_reward(response):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    if "<think>" not in response:
        return 0.0
    else:
        # Remove potential prefix
        def get_response(text, start):
            return text[text.find(start) :] if start in text else ""

        response = get_response(response, "<think>")
        pattern = r"^<think>.*?</think>.*?```json.*?```$"
        matches = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        return 1.0 if matches else 0.0


_compute_score_remote = ray.remote(_compute_score_fn)


class NaiveRewardManagerDictRay:
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        configs=None,
        no_format_score=False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        # build the base compute_score_fn
        # base_fn = compute_score or compute_score_dict
        # if configs is not None and configs.get("enable", False):
        #     self.compute_score_fn = partial(
        #         compute_score_dict, configs=configs, no_format_score=no_format_score
        #     )
        #     # print("Using custom compute_score")
        # else:
        #     self.compute_score_fn = partial(
        #         compute_score_dict, no_format_score=no_format_score, configs=configs
        #     )
        # print("Using custom compute_score configs:")
        # print(configs)
        self.compute_score_fn = partial(
            compute_score_dict, no_format_score=no_format_score, configs=configs
        )

    def __call__(self, data: DataProto):
        # print("NaiveRewardManagerDictRay __call__")
        if "rm_scores" in data.batch:
            return data.batch["rm_scores"]

        # prepare outputs
        N = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_responses = [None] * N
        reward_types = [None] * N
        traj_lens = [None] * N
        parallelism = [None] * N
        gt_traj_lens = [None] * N

        # collect args for each example
        score_args = []
        for i in range(N):
            item = data[i]
            # --- decode response_str exactly as before ---
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_pl = item.batch["attention_mask"][:prompt_len].sum()
            valid_prompts = prompt_ids[-valid_pl:]

            resp_ids = item.batch["responses"]
            valid_rl = item.batch["attention_mask"][prompt_len:].sum()
            valid_resp = resp_ids[:valid_rl]

            # only keep the last 10 of the prompt + the response
            concat_ids = torch.cat((valid_prompts[-10:], valid_resp))
            response_str = self.tokenizer.decode(concat_ids)

            batch_responses[i] = response_str

            # gather the other args
            ds = item.non_tensor_batch["data_source"]
            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            extra = item.non_tensor_batch.get("extra_info", None)

            # append a tuple that our Ray remote will unpack
            score_args.append((self.compute_score_fn, ds, response_str, gt, extra))

        # 3) fire off all tasks in parallel
        futures = [_compute_score_remote.remote(*args) for args in score_args]

        # 4) Try to get all results with timeout
        try:
            results = ray.get(futures, timeout=600)
        except Exception:
            print("Timeout occurred while waiting for tasks.")
            ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=3)
            completed_results = ray.get(ready)
            result_map = dict(zip(ready, completed_results))
            default_result = {
                "success": False,
                "traj_len": -1,
                "reward": 0,
                "reward_type": "TimeOut",
                "detail": "TImeout",
                "timing": 1000,
            }
            results = []
            for idx, fut in enumerate(futures):
                if fut in result_map:
                    results.append(result_map[fut])
                else:
                    tmpres = default_result.copy()
                    tmpres["reward"] = format_reward(score_args[idx][2])
                    results.append(tmpres)
            # results = [result_map.get(fut, default_result) for fut in futures]

        # now unpack results and fill in your arrays
        for i, score_res in enumerate(results):
            reward_types[i] = score_res["reward_type"]
            score = score_res["reward"]
            traj_lens[i] = score_res.get("traj_len", -1)
            gt_traj_lens[i] = score_res.get("gt-traj_len", -1)
            parallelism[i] = score_res.get("parallelism", -1)
            valid_rl = (data[i].batch["attention_mask"][prompt_len:]).sum()
            reward_tensor[i, valid_rl - 1] = score

        # for i in range(self.num_examine):
        for i in range(2):
            print("=== Start ☁️☁️☁️ ===")
            response_str = batch_responses[i]
            print(response_str)
            print("Results: ", results[i])
            print("=== End ☁️☁️☁️ ===")
            print("🟩" * 50)

        print("Timings: ", [x.get("timing", -1) for x in results])

        # import ipdb
        # ipdb.set_trace()
        # for idx, res in enumerate(results):
        #     # if "parse" in res["detail"].lower():
        #     if res["reward_type"] == RewardType.ParseError:
        #         # if "parseerror" in res["detail"][idx].lower():
        #         # if res['reward_t']
        #         print("=== Start ☁️☁️☁️ ===")
        #         response_str = batch_responses[idx]
        #         print(response_str)
        #         print(results[idx]["detail"])
        #         print("Results: ", results[idx])
        #         print("=== End ☁️☁️☁️ ===")
        #         print("🟩" * 50)

        # build the return dict just like before
        out = {
            "token_level_scores": reward_tensor,
            "batch_responses": np.array(batch_responses, dtype=object),
            "reward_types": np.array(reward_types, dtype=object),
            "traj_len": np.array(traj_lens),
            "parallelism": np.array(parallelism),
            "gt-traj_len": np.array(gt_traj_lens),
        }
        # carry through any other keys
        for k in ["level", "uid"]:
            if k in data.non_tensor_batch:
                out[k] = data.non_tensor_batch[k]

        return DataProto.from_single_dict(out)
