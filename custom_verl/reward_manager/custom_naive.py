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

import torch
import psutil
from functools import partial
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, TimeoutError

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def process_item_wrapper(args):
    i, data_item, tokenizer, compute_score = args

    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # decode the combined sequence
    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    sequences_str = tokenizer.decode(sequences)

    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch["data_source"]
    extra_info = data_item.non_tensor_batch.get("extra_info", None)

    score = compute_score(
        data_source=data_source,
        solution_str=sequences_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return i, valid_response_length, score, sequences_str, data_source


def process_item_default(args):
    """
    Compute the same outputs as process_item_wrapper, but default the score to 0.
    """
    i, data_item, tokenizer, _ = args  # We ignore compute_score here.
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    sequences_str = tokenizer.decode(sequences)

    # For consistency, retrieve other details.
    # Note: ground_truth is not used further here since score is set to 0.
    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch["data_source"]

    score = 0
    return i, valid_response_length, score, sequences_str, data_source


class CustomNaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        configs=None,
        no_format_score=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        if (configs is not None) and (configs["enable"]):
            # NOTE: we add configs that manipulates the compute_score function
            self.compute_score = partial(
                _default_compute_score, configs=configs, no_format_score=no_format_score
            )
            print("We're using the custom compute_score function")
        else:
            self.compute_score = compute_score or partial(
                _default_compute_score, no_format_score=no_format_score
            )
        self.MAX_WORKERS = 16
        self.TIMEOUT = 5 * 60

    def __call__(
        self,
        data,
    ):
        """
        Process tasks in batches to avoid long queuing delays.
        If batch_size is not provided, use max_workers as the batch size.
        """
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        # Prepare all the arguments for tasks.
        args_list = [
            (i, data[i], self.tokenizer, self.compute_score) for i in range(len(data))
        ]
        results = []

        batch_size = self.MAX_WORKERS
        # Process tasks in batches.
        for batch_start in range(0, len(args_list), self.MAX_WORKERS):
            batch_args = args_list[batch_start : batch_start + batch_size]
            with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                future_to_args = {
                    executor.submit(process_item_wrapper, args): args
                    for args in batch_args
                }
                for future, args in future_to_args.items():
                    try:
                        # Wait for each task in the batch individually.
                        result = future.result(timeout=self.TIMEOUT)
                        results.append(result)
                    except TimeoutError:
                        try_cancel = future.cancel()
                        # TODO (jiabao): may check if this is really correct, not sure whether the executor shutdown can fully shutdown all.
                        print("Task timed out:", future)
                        results.append(process_item_default(args))
            executor.shutdown(wait=False)

        # Sort results by index to maintain order.
        results.sort(key=lambda x: x[0])
        for i, valid_response_length, score, sequences_str, data_source in results:
            reward_tensor[i, valid_response_length - 1] = score
            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = (
                    already_print_data_sources.get(data_source, 0) + 1
                )
                print(sequences_str)
        return reward_tensor
