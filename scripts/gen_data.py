# This script genereates box environment
import os
import math
import json
import hydra
import click
import random
import numpy as np
from functools import partial
from itertools import product
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
from hydra.core.hydra_config import HydraConfig

from simulation.simenv.box1 import (
    Box1Env,
    Action,
    is_valid,
    has_object_collision,
    has_robot_collision,
    ExecutionRes,
)
from simulation.vis_api import TileMap

from inference.data_utils import read_jsonl
from inference.language_models import ModelRegistry, LanguageModel
from inference.config_store import RunConfig
from inference.utils import seed_everything, init_script
from inference.import_utils import load_function
from prompts.prompt_utils import load_constants, build_prompt


def generate_configs(base_config):
    res_configs = []
    list_params = {}
    list_length = None

    for key, value in base_config.items():
        if isinstance(value, list):
            list_params[key] = value
            if list_length is None:
                list_length = len(value)
            elif len(value) != list_length:
                raise ValueError(
                    f"List parameter '{key}' has length {len(value)}, but previous lists had length {list_length}"
                )

    # If no list parameters found, just return the original config
    if not list_params:
        return [base_config]

    # Generate configurations by column
    for i in range(list_length):
        # Create a deep copy of the base config
        config = {}

        # Copy all parameters, replacing lists with their i-th values
        for key, value in base_config.items():
            if key in list_params:
                config[key] = list_params[key][i]
            else:
                config[key] = value

        res_configs.append(config)

    return res_configs


@hydra.main(version_base=None, config_path="../configs", config_name="box_gen_config")
def main(configs: RunConfig):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    env_config = OmegaConf.to_container(configs.env_config)
    seed_everything(configs.seed)

    env_configs = generate_configs(env_config)
    all_envs = {}
    for i, env_config in tqdm(enumerate(env_configs)):
        n = env_config["grid_N"]
        m = env_config["grid_M"]
        obj = env_config["num_objects"]
        outdir = os.path.join(OUTPUTDIR, f"env-n@{n},m@{m},obj@{obj}")
        os.makedirs(outdir, exist_ok=True)
        # for j in range(10):
        MAX_POTENTIAL = (
            math.comb(n * 2 * m * 2, obj * 2)
            * math.factorial(2 * obj)
            / math.factorial(obj)
        )
        accum = 0
        cur_setting = {}
        while accum < min(MAX_POTENTIAL, configs.num):
            # seed_everything(configs.seed + j)
            env = Box1Env(
                "Box1Env",
                env_config["grid_N"],
                env_config["grid_M"],
                env_config["num_objects"],
                env_config["robot_mode"],
                env_config["robot_as"],
                env_config["movement"],
            )
            env.create()
            # env.visualize(out_file_path=os.path.join(outdir, f"env-{j:02}.png"))
            env_hash = env.hash()
            if env_hash not in cur_setting:
                cur_setting[env_hash] = env.to_json()
                accum += 1
        all_envs[(n, m, obj)] = cur_setting

    SPLIT = True
    if SPLIT:
        TEST_POS = 10
        prev_gen_train = read_jsonl(
            "/mnt/data/jiabao/2025-Projects/MultiRobotR1/gen_plan.jsonl"
        )
        CANNOT_TEST = {}
        for env_json in prev_gen_train:
            env = Box1Env.load(env_json["env"])
            env_hash = env.hash()
            CANNOT_TEST[env_hash] = True

        train_envs = []
        test_envs = []

        for sub_env in all_envs.values():
            hashes = list(sub_env.keys())
            envs = list(sub_env.values())
            cur_test_envs = []
            cur_train_envs = []
            for i in range(len(envs)):
                hash_value = hashes[i]
                env = envs[i]
                if hash_value not in CANNOT_TEST:
                    cur_test_envs.append(env)
                    if len(cur_test_envs) >= TEST_POS:
                        break
                else:
                    cur_train_envs.append(env)
                    print("+1")
            train_envs.extend(envs[i + 1 :])
            test_envs.extend(cur_test_envs)

        with open(os.path.join(OUTPUTDIR, "all-envs-train.jsonl"), "w") as f_train:
            for env in train_envs:
                f_train.write(json.dumps(env) + "\n")
        with open(os.path.join(OUTPUTDIR, "all-envs-test.jsonl"), "w") as f_test:
            for env in test_envs:
                f_test.write(json.dumps(env) + "\n")

    else:
        with open(os.path.join(OUTPUTDIR, "all_envs.jsonl"), "w") as f:
            for sub_env in all_envs.values():
                for env in sub_env.values():
                    f.write(json.dumps(env) + "\n")


if __name__ == "__main__":
    main()
