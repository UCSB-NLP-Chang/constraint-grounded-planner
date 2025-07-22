import os
import json
import random
from abc import ABC
import numpy as np
from collections import Counter

from simulation.simenv.plan_count import stats
from custom_verl.reward_utils import RewardType
from verl.protocol import DataProto
from verl.utils.tracking import Tracking


class Callback(ABC):
    def __init__(self, logger: Tracking, **kwargs):
        super().__init__()
        self.logger = logger

    def on_generate_sequences(self, global_step, batch: DataProto, split: str):
        pass

    def on_val_end(self, global_step, batch: DataProto, split: str):
        pass


class KeyworkdCountCallback(Callback):
    def __init__(self, logger, keyword_file: str = "keywords.txt", **kwargs):
        super().__init__(logger=logger)
        with open(keyword_file, "r") as f:
            self.keywords = f.readlines()
            self.keywords = [keyword.strip() for keyword in self.keywords]

    def on_generate_sequences(self, global_step: int, batch: DataProto, split: str):
        keyname = "batch_responses"
        if keyname not in batch.non_tensor_batch:
            return

        count_res = {}
        response_strs = batch.non_tensor_batch[keyname]

        if "attention_mask" in batch.batch:  # in this case, there may be some padding
            response_strs = [
                x
                for i, x in enumerate(response_strs)
                if batch.batch["attention_mask"][i].sum() != 0
            ]

        keywords_counts = []
        for response in response_strs:
            response_cnt = 0
            for keyword in self.keywords:
                response_cnt += response.lower().count(keyword.lower())
            keywords_counts.append(response_cnt)

        count_res = {
            f"{split}/keyword/batch_sum": np.sum(keywords_counts).item(),
            f"{split}/keyword/response_mean": np.mean(keywords_counts),
        }
        if random.random() < 0.1:
            print(
                f"Keyword at step {global_step}, split {split}",
                json.dumps(count_res, indent=4),
            )
        self.logger.log(data=count_res, step=global_step)
        return count_res


class PlanLenCallback(Callback):
    def __init__(self, logger, reduce_traj=False, **kwargs):
        super().__init__(logger, **kwargs)
        self.reduce_traj = reduce_traj

    def report_non_reduce(self, global_step, batch, split):
        keyname = "traj_len"
        if keyname not in batch.non_tensor_batch:
            return
        keyname = "reward_types"
        if keyname not in batch.non_tensor_batch:
            return

        traj_len = batch.non_tensor_batch["traj_len"]
        traj_parall = batch.non_tensor_batch["parallelism"]
        gt_traj_len = batch.non_tensor_batch["gt-traj_len"]
        reward_types = batch.non_tensor_batch["reward_types"]

        # filter by reward_score
        traj_len = [
            x for x, y in zip(traj_len, reward_types) if y == RewardType.Correct
        ]
        traj_parall = [
            x for x, y in zip(traj_parall, reward_types) if y == RewardType.Correct
        ]
        gt_traj_len = [
            x for x, y in zip(gt_traj_len, reward_types) if y == RewardType.Correct
        ]
        traj_len_stats = stats(traj_len, name=f"{split}-traj/traj_len")
        traj_parall_stats = stats(traj_parall, name=f"{split}-traj/parallelism")
        traj_advantage_stats = stats(
            [x - y for x, y in zip(traj_len, gt_traj_len)],
            name=f"{split}-traj/traj_len-advantage",
        )
        merged = {
            f"{split}-traj/traj-correct": len(traj_len),
            f"{split}-traj/traj-correct-ratio": len(traj_len)
            / len(batch.non_tensor_batch["traj_len"]),
            **traj_len_stats,
            **traj_parall_stats,
            **traj_advantage_stats,
        }
        print(merged)
        self.logger.log(data=merged, step=global_step)
        return merged

    def report_reduce(self, global_step, batch, split):
        pass

    def on_generate_sequences(self, global_step, batch, split):
        if self.reduce_traj:
            return self.report_reduce(global_step, batch, split)
        else:
            return self.report_non_reduce(global_step, batch, split)


class RewardTypeCallback(Callback):
    def __init__(self, logger, reduce_traj=False, **kwargs):
        super().__init__(logger, **kwargs)
        self.reduce_traj = reduce_traj

    def report_non_reduce(self, global_step, batch, split):
        keyname = "reward_types"
        reward_type_count = batch.non_tensor_batch[keyname]
        if "attention_mask" in batch.batch:
            reward_type_count = [
                x
                for i, x in enumerate(reward_type_count)
                if batch.batch["attention_mask"][i].sum() != 0
            ]

        if isinstance(reward_type_count[0], list) or isinstance(
            reward_type_count[0], np.ndarray
        ):
            reward_type_count = [
                item for sublist in reward_type_count for item in sublist
            ]
        type_counts = Counter(reward_type_count)
        total_count = sum(type_counts.values())

        def convert_reward_type(reward):
            if isinstance(reward, str):
                return reward
            elif isinstance(reward, RewardType):
                return reward.name
            else:
                return str(reward)

        json_counts = {
            f"{split}/reward_type/{convert_reward_type(reward)}": count / total_count
            for reward, count in type_counts.items()
        }
        if random.random() < 0.1:
            print(
                f"RewardType at step {global_step}, split {split}",
                json.dumps(json_counts, indent=4),
            )
        self.logger.log(data=json_counts, step=global_step)

    def report_reduce(self, global_step, batch, split):
        keyname = "reward_types"
        reward_type_count = batch.non_tensor_batch[keyname]
        uids = batch.non_tensor_batch["uids"]
        step_ids = batch.batch["step_ids"].tolist()
        non_pad_indices = [
            i for i, x in enumerate(batch.batch["attention_mask"]) if x.sum() != 0
        ]
        # we want to find the last step for each unique id
        reward_type_count = [
            x for i, x in enumerate(reward_type_count) if i in non_pad_indices
        ]
        if isinstance(reward_type_count[0], list) or isinstance(
            reward_type_count[0], np.ndarray
        ):
            reward_type_count = [
                item for sublist in reward_type_count for item in sublist
            ]
        uids = [x for i, x in enumerate(uids) if i in non_pad_indices]
        step_ids = [x for i, x in enumerate(step_ids) if i in non_pad_indices]

        # group by uids,  they're the same trajectory
        grouped_reward_types = {}
        for uid, step_id, reward_type in zip(uids, step_ids, reward_type_count):
            if uid not in grouped_reward_types:
                grouped_reward_types[uid] = {}
            if step_id not in grouped_reward_types[uid]:
                grouped_reward_types[uid][step_id] = []
            grouped_reward_types[uid][step_id].append(reward_type)

        # we find the last step for each unique id
        last_step_reward_types = {}
        for uid, step_id_dict in grouped_reward_types.items():
            last_step = max(step_id_dict.keys())
            last_step_reward_types[uid] = step_id_dict[last_step]

        # flatten the list
        last_step_reward_types = [
            item for sublist in last_step_reward_types.values() for item in sublist
        ]
        type_counts = Counter(last_step_reward_types)
        total_count = sum(type_counts.values())
        json_counts = {
            f"{split}-traj/reward_type/{reward.name}": count / total_count
            for reward, count in type_counts.items()
        }
        if random.random() < 0.1:
            print(
                f"RewardType at step {global_step}, split {split}",
                json.dumps(json_counts, indent=4),
            )
        self.logger.log(data=json_counts, step=global_step)

    def on_generate_sequences(self, global_step, batch, split):
        keyname = "reward_types"
        if keyname not in batch.non_tensor_batch:
            return

        self.report_non_reduce(global_step, batch, split)
        if self.reduce_traj:
            self.report_reduce(global_step, batch, split)


class ValSaveCallback(Callback):
    def __init__(self, logger, output_dir, **kwargs):
        super().__init__(logger, **kwargs)
        self.output_dir = output_dir

    def on_val_end(self, global_step, batch, split):
        os.makedirs(self.output_dir, exist_ok=True)
        all_reuslts = []
        for i in range(len(batch.non_tensor_batch["batch_responses"])):
            if (
                "attention_mask" in batch.batch
                and batch.batch["attention_mask"][i].sum() == 0
            ):
                continue
            reward_types = batch.non_tensor_batch["reward_types"][i]
            if "uids" in batch.non_tensor_batch:
                uids = batch.non_tensor_batch["uids"][i]
            else:
                uids = batch.non_tensor_batch["uid"][i]
            step_ids = batch.batch.get("step_ids", None)
            all_reuslts.append(
                {
                    "uid": uids,
                    "input": batch.non_tensor_batch["batch_inputs"][i],
                    "response": batch.non_tensor_batch["batch_responses"][i],
                    "score": batch.batch["token_level_scores"][i].sum().item(),
                    "reward_types": [x.name for x in reward_types]
                    if isinstance(reward_types, list)
                    else [reward_types.name],
                    "step_ids": step_ids[i].item() if step_ids is not None else -1,
                }
            )

        with open(
            os.path.join(self.output_dir, f"{split}_step@{global_step}.jsonl"), "w"
        ) as f:
            for line in all_reuslts:
                f.write(json.dumps(line) + "\n")


class CallbackManager(Callback):
    def __init__(self, logger: Tracking, rootdir=None, **kwargs):
        super().__init__(logger=logger, **kwargs)

        # TODO (jiabao): we may add more callbacks here
        self.callbacks = []
        self.callbacks.append(KeyworkdCountCallback(self.logger, **kwargs))
        self.callbacks.append(RewardTypeCallback(self.logger, **kwargs))
        self.callbacks.append(PlanLenCallback(self.logger, **kwargs))

        # self.callbacks.append(PassKMetricCallback(self.logger))
        if rootdir is not None:
            self.callbacks.append(
                ValSaveCallback(self.logger, os.path.join(rootdir, "evallog"), **kwargs)
            )

    def on_generate_sequences(self, step, batch: DataProto, split: str):
        for callback in self.callbacks:
            callback.on_generate_sequences(step, batch, split)

    def on_val_end(self, step, batch: DataProto, split: str):
        for callback in self.callbacks:
            callback.on_val_end(step, batch, split)
