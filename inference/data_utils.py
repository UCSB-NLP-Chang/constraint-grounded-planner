# This file defines GeneridDict, which is similar to TensorDict but supports non-tensor data, and each batch is a list of items.

import os
import json
import pickle
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)
import copy
from tqdm import tqdm
from typing import Dict, Union, List


def build_dataset(dataset_name, split="train") -> Dataset:
    """Load dataset from disk or huggingface datasets"""
    if os.path.exists(dataset_name):
        if os.path.isdir(dataset_name):
            rawdata = load_from_disk(dataset_name)
        else:
            if dataset_name.endswith(".json"):
                rawdata = load_dataset("json", data_files=[dataset_name])
            elif dataset_name.endswith(".jsonl"):
                rawdata = load_dataset("json", data_files=[dataset_name])
            elif dataset_name.endswith(".parquet"):
                rawdata = load_dataset("parquet", data_files=[dataset_name])
            elif dataset_name.endswith(".pkl"):
                rawdata = pickle.load(open(dataset_name, "rb"))
            else:
                raise ValueError("Unknown dataset format")
    else:
        if "gsm8k" in dataset_name:
            rawdata = load_dataset(dataset_name, "main")
        else:
            rawdata = load_dataset(dataset_name)
    if isinstance(rawdata, DatasetDict):
        if split in rawdata:
            rawdata = rawdata[split]
    return rawdata


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class DataDict:
    def __init__(self, data=None):
        """
        A simple implementation of a TensorDict-like container that supports non-tensor data.
        Instead of batching numerically, it stores values in lists.

        Args:
            data (dict, optional): Dictionary containing key-value pairs.
        """
        self.data = data if data is not None else {}

        if self.data:
            for key, value in self.data.items():
                if not isinstance(value, list):
                    self.data[key] = [value]
            lengths = [len(v) for v in self.data.values()]
            if lengths and not all(l == lengths[0] for l in lengths):
                raise ValueError(
                    "All list values in DataDict must have the same length"
                )
            self.length = lengths[0]

    @property
    def batch_size(self):
        """Returns the batch size by checking the length of the first list item."""
        return self.length

    def __getitem__(self, key):
        """If key is an int, return a single indexed item from all keys.
        If key is a string, return the corresponding list."""
        if isinstance(key, int):
            return DataDict.from_dict(
                {k: v[key] for k, v in self.data.items()}
            )  # Extract row-like structure
        if isinstance(key, slice):
            return DataDict({k: v[key] for k, v in self.data.items()})
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        """Deletes a key from the dictionary."""
        if key in self.data:
            del self.data[key]

    def __contains__(self, key):
        """Checks if a key exists."""
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def clone(self):
        """Creates a deep copy of the dictionary."""
        return DataDict(
            {k: v.copy() if isinstance(v, list) else v for k, v in self.data.items()}
        )

    def repeat(self, n):
        return DataDict(
            {k: v * n if isinstance(v, list) else v for k, v in self.data.items()}
        )

    def repeat_interleave(self, n):
        # This function repeats the dict like (1, 2, 3) -> (1, 1, 2, 2, 3, 3)
        return DataDict(
            {
                k: [x for x in v for _ in range(n)] if isinstance(v, list) else v
                for k, v in self.data.items()
            }
        )

    def to_list(self):
        """Converts each stored value into a list format, batching by grouping elements."""
        return [{k: v[i] for k, v in self.data.items()} for i in range(self.batch_size)]

    def select(self, keys):
        """Selects a subset of keys from the dictionary."""
        return DataDict({k: self.data[k] for k in keys if k in self.data})

    def update(self, other_dict):
        """Updates the dictionary with another DataDict or standard dict."""
        self.data.update(
            other_dict if isinstance(other_dict, dict) else other_dict.data
        )

    def pop(self, key):
        """Removes a key and returns its value."""
        return self.data.pop(key, None)

    @classmethod
    def from_dict(cls, data):
        """Creates a DataDict from a standard dictionary."""
        return cls(data)

    @classmethod
    def from_list_of_dicts(cls, list_of_dicts):
        """
        Converts a list of dictionaries into a DataDict.

        Example:
        input: [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"}
        ]
        output: DataDict({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        Args:
            list_of_dicts (list): List of dictionaries where each dictionary represents a row.

        Returns:
            DataDict: The batched representation of the data.
        """
        if not list_of_dicts:
            return cls({})

        keys = list_of_dicts[0].keys()
        batched_data = {key: [d.get(key, None) for d in list_of_dicts] for key in keys}
        return cls(batched_data)

    def union(self, other_datadict: "DataDict") -> "DataDict":
        other_dict = other_datadict.data
        out = self.union_with_dict(other_dict)
        return out

    def union_with_dict(self, other_dict):
        """
        Merges another dictionary into the existing DataDict.
        If a key exists in both, it extends the list values.
        """
        new_data = copy.deepcopy(self.data)
        for k, v in other_dict.items():
            if k in new_data:
                if isinstance(new_data[k], list) and isinstance(v, list):
                    new_data[k].extend(v)  # Merge lists
                else:
                    new_data[k] = [new_data[k], v]  # Convert to list if necessary
            else:
                new_data[k] = v if isinstance(v, list) else [v]

        return DataDict.from_dict(new_data)

    @classmethod
    def concat(self, data_dicts: List["DataDict"]) -> "DataDict":
        for data_dict in data_dicts:
            if not isinstance(data_dict, DataDict):
                raise ValueError("All items must be DataDict instances.")
        if not data_dicts:
            raise ValueError("No DataDicts to concatenate.")
        keys = data_dicts[0].keys()
        for data_dict in data_dicts[1:]:
            if keys != data_dict.keys():
                print(keys)
                print(data_dict.keys())
                raise ValueError("All DataDicts must have the same keys.")
        new_data = {k: [] for k in keys}
        for data_dict in data_dicts:
            for k in keys:
                new_data[k].extend(data_dict[k])
        return DataDict.from_dict(new_data)

    def as_list_dict(self):
        """Ensures all values in the dictionary are lists."""
        return {k: v if isinstance(v, list) else [v] for k, v in self.data.items()}

    def __repr__(self):
        return f"DataDict(batch_size={self.batch_size}, data={list(self.data.keys())})"

    def __len__(self):
        return self.batch_size

    def to_dict(self):
        return self.data

    def groupby(self, keys) -> Dict[str, "DataDict"]:
        """
        Groups the DataDict based on one or multiple keys.

        Args:
            keys (str or list of str): The key(s) used for grouping.

        Returns:
            dict: A dictionary where keys are unique values (or tuples for multiple keys)
                  from the specified grouping keys, and values are DataDict instances
                  containing the grouped data.
        """
        if isinstance(keys, str):
            keys = [keys]  # Convert single key to list for uniform processing

        def clean_group_key(group_key):
            if len(group_key) == 1:
                return group_key[0]
            return group_key

        grouped_data = {}
        key_values_list = [self.data.get(k, []) for k in keys]

        if any(not values for values in key_values_list):  # If any key does not exist
            return {}

        for idx in range(self.batch_size):
            group_key = tuple(
                values[idx] for values in key_values_list
            )  # Create a unique tuple for multiple keys

            if group_key not in grouped_data:
                grouped_data[group_key] = {k: [] for k in self.data.keys()}

            for k, v in self.data.items():
                grouped_data[group_key][k].append(v[idx])

        return {
            clean_group_key(group_key): DataDict(group_values)
            for group_key, group_values in grouped_data.items()
        }


def batchify_sampler(total_num, batch_size, shuffle=False) -> List[List[int]]:
    if shuffle:
        random_index = np.random.permutation(total_num).tolist()
    else:
        random_index = list(range(total_num))

    all_indexes = []
    for i in range(0, total_num, batch_size):
        indexes = [random_index[j] for j in range(i, min(i + batch_size, total_num))]
        all_indexes.append(indexes)
    return all_indexes


def list_repeat_interleave(data, repeats):
    # Repeat interleave a list
    if isinstance(repeats, int):
        return [copy.deepcopy(item) for item in data for _ in range(repeats)]
    elif isinstance(repeats, list):
        if len(data) != len(repeats):
            raise ValueError("Length of data and repeats must be the same.")
        return [item for item, count in zip(data, repeats) for _ in range(count)]
    else:
        raise TypeError("repeats must be an int or a list of ints.")


def count_tokens(data, tokenizer):
    def build_prompt(item):
        parts = []
        if item.get("system"):
            parts.append(item["system"])
        if item.get("history"):
            for turn in item["history"]:
                parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        if item.get("instruction"):
            parts.append(f"Instruction: {item['instruction']}")
        if item.get("input"):
            parts.append(f"Input: {item['input']}")
        parts.append(f"Output: {item.get('output', '')}")
        return "\n".join(parts)

    # Build all prompts first
    prompts = [build_prompt(item) for item in data]

    # Tokenize in batch mode (fast)
    token_lens = []
    for i in tqdm(range(0, len(prompts), 32)):  # process in batches of 32
        batch = prompts[i : i + 32]
        encodings = tokenizer(
            batch, padding=False, truncation=False, return_attention_mask=False
        )
        token_lens.extend([len(ids) for ids in encodings["input_ids"]])

    # Compute stats
    average_length = sum(token_lens) / len(token_lens)
    max_length = max(token_lens)

    return {
        "average_length": average_length,
        "max_length": max_length,
        "total_tokens": sum(token_lens),
        "tokenlens": token_lens,
    }
