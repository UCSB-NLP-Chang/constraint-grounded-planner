import os
import json
import numpy as np
from collections import defaultdict
from functools import reduce
from datasets import load_dataset, load_from_disk, Dataset
from typing import Dict, List, Any
from inference.eval_utils import pass_at_k

from .box1 import Box1Env, Action, ExecutionRes


def stats(data, name=""):
    if len(data) > 0:
        return {
            f"{name}/mean": np.mean(data).item(),
            f"{name}/min": np.min(data).item(),
            f"{name}/max": np.max(data).item(),
        }
    else:
        return {
            f"{name}/mean": 0,
            f"{name}/min": 0,
            f"{name}/max": 0,
        }


def count_plan_stats(dataset: Dataset):
    plans = [json.loads(x["extra_info"]["gt_plan"]) for x in dataset]
    # if isinstance(plans[0], str):
    # plans = [json.loads(x) for x in plans]
    print(plans[0])
    # if isinstance(plans[0][0], str):
    #     plan_len = [len(x.split("\n")) for x in plans]
    # else:
    plan_len = [len(x) for x in plans]

    if isinstance(plans[0][0], str):
        plan_parallel = [[len(x.split("\n")) for x in y] for y in plans]
    else:
        plan_parallel = [[len(x) for x in y] for y in plans]
    plan_parallel = reduce(lambda x, y: x + y, plan_parallel)

    return {
        "numstep/total": sum(plan_len),
        **stats(plan_len, "numstep"),
        **stats(plan_parallel, "parallellism"),
    }


def visualize_plan(sample: Dict[str, Any], outdir):
    env_configs = json.loads(sample["reward_model"]["ground_truth"])
    boxenv = Box1Env.load(env_configs)
    plans = json.loads(sample["extra_info"]["gt_plan"])
    plan_actions = Action.from_list(plans, boxenv.robots)

    os.makedirs(outdir, exist_ok=True)
    for step, step_action in enumerate(plan_actions):
        for action in step_action:
            action.arm_pos = boxenv.robots[action.robot_id].arm_pos
        exec_res = boxenv.verify(step_action)
        if exec_res.success != ExecutionRes.Success:
            print(f"Invalid actions: {step_action}; Reason: {exec_res}")
            break
        else:
            boxenv.visualize(
                actions=step_action,
                exec_res=exec_res,
                out_file_path=os.path.join(outdir, f"step@{step:02}.png"),
            )
            boxenv.simulate(step_action)

    boxenv.visualize(
        actions=None,
        exec_res=None,
        out_file_path=os.path.join(outdir, "step@final.png"),
    )

    if boxenv.check_final():
        print("Objects are all in target positions")
    else:
        print("Objects failed to be put at target positions")


import re


def extract_json(text):
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    res = match.group(1) if match else ""

    if res != "":
        res = f"""```json\n{res}\n```"""

    pattern = re.compile("```json\n(.*?)```", re.DOTALL)
    match = pattern.findall(res)
    if match:
        try:
            out = json.loads(match[-1])
            return out
        except json.JSONDecodeError:
            return None
    else:
        return None


def get_index(dataset):
    indexes = defaultdict(list)
    for i, sample in enumerate(dataset):
        obj = json.loads(sample["reward_model"]["ground_truth"])["num_objects"]
        indexes[obj].append(i)
    return indexes


def count_result(dataset: Dataset):
    print(dataset)
    print("traj-detail" in dataset)
    if "traj-detail" in dataset.column_names:
        print("Step data")

        def get_plan_len(sample):
            return len(sample["traj-detail"])

        def get_parallelism(sample):
            plans = []
            for step in sample["traj-detail"]:
                plan = extract_json(step["output_text"])
                if plan is None:
                    plans.append(-1)
                else:
                    plans.append(len(plan))
            return max(plans)

        dataset = dataset.add_column("plan_len", [get_plan_len(x) for x in dataset])
        if "parallelism" not in dataset.column_names:
            dataset = dataset.add_column(
                "parallelism", [get_parallelism(x) for x in dataset]
            )

    else:

        def get_plan_len(sample):
            plan = extract_json(sample["output_text"])
            if plan is None:
                return -1
            else:
                return len(plan)

        def get_parallelism(sample):
            plan = extract_json(sample["output_text"])
            if plan is None:
                return -1
            else:
                if len(plan) > 0:
                    return max([len(x) for x in plan])
                else:
                    return -1

        planlen = [get_plan_len(x) for x in dataset]
        dataset = dataset.add_column("plan_len", planlen)
        if "parallelism" not in dataset.column_names:
            parallel = [get_parallelism(x) for x in dataset]
            dataset = dataset.add_column("parallelism", parallel)

    cor = dataset.filter(lambda x: x["success"])
    print(stats(cor["plan_len"], "plan_len"))
    print(stats(cor["parallelism"], "parallelism"))
    if "gt_planlen" in dataset.column_names:
        gtplanlen = cor["gt_planlen"]
        genplanlen = cor["plan_len"]
        print(stats(np.array(genplanlen) - np.array(gtplanlen), "lendiff"))
        # print(stats(gtplanlen, "gt_planlen"))
        # print(stats(cor["gt_planlen"], "gt_planlen"))
    if "gt_para" in dataset.column_names:
        print(stats(cor["gt_para"], "gt_para"))

    print("Overall")
    for k in [1, 2, 4]:
        print(f"pass@{k}", pass_at_k(dataset["uid"], dataset["success"], k))
    print("===" * 40)


def has_k_equal(example, k):
    if "data_source" in example:
        parts = example["data_source"].split("_")
        for part in parts:
            if part.startswith("k@"):
                try:
                    return int(part.split("@")[1]) == k
                except ValueError:
                    return False
    return False


def select_partial(dataset):
    nmtuple = [
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
        (2, 2, 4),
        (2, 2, 5),
        (3, 3, 1),
        (3, 3, 2),
        (3, 3, 3),
        (3, 3, 4),
        (3, 3, 5),
        (4, 4, 1),
        (4, 4, 2),
        (4, 4, 3),
        (4, 4, 4),
        (4, 4, 5),
        (5, 5, 1),
        (5, 5, 2),
        (5, 5, 3),
        (5, 5, 4),
        (5, 5, 5),
        (6, 6, 1),
        (6, 6, 2),
        (6, 6, 3),
        (6, 6, 4),
        (6, 6, 5),
    ]
    targetmode = "full"

    def filter_source(sample):
        # data_source is like: box_n@6_m@6_k@5_mode@full
        data_source = sample["data_source"]
        match = re.search(r"n@(\d+)_m@(\d+)_k@(\d+)_mode@(\w+)", data_source)
        n = int(match.group(1))
        m = int(match.group(2))
        k = int(match.group(3))
        mode = match.group(4)
        if mode == targetmode:
            if (n, m, k) in nmtuple:
                return True
        else:
            return False

    dataset = dataset.filter(filter_source)
    return dataset


def select_group(dataset, k=10):
    df = dataset.to_pandas()
    small_df = df.groupby("data_source").head(k).reset_index(drop=True)
    small_dataset = Dataset.from_pandas(small_df)
    return small_dataset


def group_dataset(dataset, keyfunc):
    def add_key(sample):
        key = keyfunc(sample)
        sample["tmpkey"] = key
        return sample

    dataset = dataset.map(add_key, num_proc=8)
    df = dataset.to_pandas()
    split_df = df.groupby("tmpkey")
    resdict = {k: Dataset.from_pandas(group) for k, group in split_df}
    return resdict
