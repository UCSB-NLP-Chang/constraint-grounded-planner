import os
import copy
import json
import pickle
from datasets import concatenate_datasets
from typing import List, Any, Dict, Union, Optional
from omegaconf import ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from inference.import_utils import load_function
from simulation.simenv.box1 import Box1Env

from custom_verl.prompts.prompt_utils import (
    load_constants,
    replace_prompt,
    find_format_map_names,
)
from custom_verl.data_utils import build_dataset


class HFRLHFDataset(RLHFDataset):
    # We read huggingface dataset instead of parquet files
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        **kwargs,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        if self.filter_fn is None:
            self.filter_fn = (
                lambda sample: len(
                    self.tokenizer.apply_chat_template(
                        sample[prompt_key],
                        add_generation_prompt=True,
                    )
                )
                <= self.max_prompt_length
            )
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        parquet_files = (
            self.parquet_files
            if not use_origin_parquet
            else self.original_parquet_files
        )
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(
                src=parquet_file, cache_dir=self.cache_dir
            )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = build_dataset(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = concatenate_datasets(dataframes)
        print(f"original dataset len: {len(self.dataframe)}")
        self.dataframe = self.dataframe.filter(self.filter_fn, num_proc=8)
        print(f"filter dataset len: {len(self.dataframe)}")

    def process_sample(
        self, prompt_with_chat_template: str, row_dict: Dict[str, Any], chat
    ) -> Dict[str, Any]:
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            prompt_with_chat_template, add_special_tokens=False
        )

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item]
        row_dict = copy.deepcopy(row_dict)
        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        return self.process_sample(prompt_with_chat_template, row_dict, chat)


class DynamicPromptRLHFDataset(HFRLHFDataset):
    # This class dynamiclly replaces the prompt with the template in prompt_file.
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        prompt_file=None,
        **kwargs,
    ):
        self.MESSAGE_TEMPLATE = load_constants(
            prompt_file
        )  # We apply differnt prompt dynamically, to avoid waste of hard drive space
        required_names = find_format_map_names(
            "\n".join([x["content"] for x in self.MESSAGE_TEMPLATE])
        )
        self.required_names = [x for x in required_names if x != ""]

        self.filter_fn = (
            lambda sample: len(
                self.tokenizer.apply_chat_template(
                    self.prompt_fn(sample), add_generation_prompt=True
                )
            )
            <= self.max_prompt_length
        )

        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            image_key=image_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=filter_prompts,
            cache_dir=cache_dir,
            chat_template_func=chat_template_func,
            return_raw_chat=return_raw_chat,
            truncation=truncation,
            filter_overlong_prompts=filter_overlong_prompts,
        )

    def prompt_fn(self, sample):
        sample = copy.deepcopy(sample)
        replace_dict = {key: sample.pop(key) for key in self.required_names}
        res = replace_prompt(self.MESSAGE_TEMPLATE, replace_dict)
        return res

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = self.prompt_fn(row_dict)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        return self.process_sample(prompt_with_chat_template, row_dict, chat)


class BoxRLHFDataset(DynamicPromptRLHFDataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        prompt_file=None,
        **kwargs,
    ):
        self.build_state_func = load_function(prompt_file, "Map2Text")
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            image_key=image_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=filter_prompts,
            cache_dir=cache_dir,
            chat_template_func=chat_template_func,
            return_raw_chat=return_raw_chat,
            truncation=truncation,
            filter_overlong_prompts=filter_overlong_prompts,
            prompt_file=prompt_file,
        )

    def prompt_fn(self, sample):
        if isinstance(sample["reward_model"]["ground_truth"], str):
            sample["reward_model"]["ground_truth"] = json.loads(
                sample["reward_model"]["ground_truth"]
            )

        env_json = sample["reward_model"]["ground_truth"]
        env = Box1Env.load(env_json)
        mapstate = self.build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )
        replace_dict = {"mapstate": mapstate}
        res = replace_prompt(self.MESSAGE_TEMPLATE, replace_dict)
        return res

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = self.prompt_fn(row_dict)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        res = self.process_sample(prompt_with_chat_template, row_dict, chat)
        if isinstance(row_dict["reward_model"]["ground_truth"], str):
            env_configs = json.loads(row_dict["reward_model"]["ground_truth"])
        else:
            env_configs = row_dict["reward_model"]["ground_truth"]
        env_configs["gt_plan"] = row_dict["extra_info"]["gt_plan"]
        res["env_configs"] = json.dumps(env_configs)
        res["uids"] = res.get("uid", item)
        return res


class Box3DRLHFDataset(DynamicPromptRLHFDataset):
    def __init__(
        self,
        parquet_files,
        tokenizer,
        processor=None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        prompt_file=None,
        **kwargs,
    ):
        self.build_state_func = load_function(prompt_file, "describe_obs")
        super().__init__(
            parquet_files,
            tokenizer,
            processor,
            prompt_key,
            image_key,
            max_prompt_length,
            filter_prompts,
            cache_dir,
            chat_template_func,
            return_raw_chat,
            truncation,
            filter_overlong_prompts,
            prompt_file,
            **kwargs,
        )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pickle.load(open(parquet_file, "rb"))
            dataframes.extend(dataframe)
        self.dataframe = dataframes
        print(f"original dataset len: {len(self.dataframe)}")
        self.dataframe = [
            # self.dataframe.filter(self.filter_fn, num_proc=8)
            x
            for x in self.dataframe
            if self.filter_fn(x)
        ]
        print(f"filter dataset len: {len(self.dataframe)}")

    def prompt_fn(self, sample):
        if isinstance(sample["reward_model"]["ground_truth"], str):
            sample["reward_model"]["ground_truth"] = json.loads(
                sample["reward_model"]["ground_truth"]
            )
        env_config = sample["reward_model"]["ground_truth"]
        obs = sample["traj-obs"][0]
        mapstate = self.build_state_func(
            obs=obs,
            target_positions=env_config["targets"],
        )

        replace_dict = {"mapstate": mapstate}
        res = replace_prompt(self.MESSAGE_TEMPLATE, replace_dict)
        return res

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = self.prompt_fn(row_dict)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        res = self.process_sample(prompt_with_chat_template, row_dict, chat)
        if isinstance(row_dict["reward_model"]["ground_truth"], str):
            env_configs = json.loads(row_dict["reward_model"]["ground_truth"])
        else:
            env_configs = row_dict["reward_model"]["ground_truth"]
        env_configs["gt_plan"] = row_dict["extra_info"]["gt_plan"]
        res["env_configs"] = json.dumps(env_configs)
        res["uids"] = res.get("uid", item)
        return res


class CountdownRLHFDataset(HFRLHFDataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        **kwargs,
    ):
        self.filter_fn = (
            lambda sample: len(self.tokenizer(sample["prompt"][0]["content"]).input_ids)
            <= self.max_prompt_length
        )

        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            image_key=image_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=filter_prompts,
            cache_dir=cache_dir,
            chat_template_func=chat_template_func,
            return_raw_chat=return_raw_chat,
            truncation=truncation,
            filter_overlong_prompts=filter_overlong_prompts,
        )

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]
        prompt_with_chat_template = row_dict["prompt"][0]["content"]

        return self.process_sample(prompt_with_chat_template, row_dict, [])
