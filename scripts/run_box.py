import os
import ray
import hydra
import click
import random
import numpy as np
import json
from functools import partial, reduce
from itertools import product
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from datetime import datetime, date


from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from simulation.simenv.box1 import (
    Box1Env,
    Action,
    is_valid,
    has_object_collision,
    has_robot_collision,
    ExecutionRes,
)
from simulation import ENV_REGISTRY
from simulation.vis_api import TileMap

from inference.runner import RUNNER_REGISTRY
from inference.data_utils import build_dataset, read_jsonl
from inference.language_models import ModelRegistry, LanguageModel, build_language_model
from inference.config_store import RunConfig
from inference.utils import seed_everything, init_script
from inference.import_utils import load_function
from prompts.prompt_utils import load_constants, build_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def object_to_json_serializable(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: object_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [object_to_json_serializable(item) for item in obj]
    elif hasattr(obj, "to_json") and callable(obj.to_json):
        return object_to_json_serializable(obj.to_json())
    elif hasattr(obj, "__dict__"):
        return {
            key: object_to_json_serializable(value)
            for key, value in obj.__dict__.items()
        }
    else:
        return str(obj)  # fallback to string representation


def get_cache_dir():
    import socket
    import getpass

    hostname = socket.gethostname()  # check machine
    username = getpass.getuser()
    print("Hostname: ", hostname)
    if "bluevela" in hostname:
        if "kqian" in username:
            # return "/proj/long-multi/kqian/granite-3b-ablate/.ray_cache"
            # return "/proj/long-multi/kqian/granite-3b-ablate/.r"
            # return "/tmp/ray"
            return "/tmp"
        elif "becon" in username:
            return "/proj/long-multi/midi/.ray_cache"
    elif "katrina1" in hostname:  # we're in katrina
        return "/mnt/data3/jiabao/.ray_cache/"
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="../configs", config_name="box1_config")
def main(configs: RunConfig):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    env_config = configs.env_config

    seed_everything(configs.seed)

    if configs.data.endswith(".jsonl"):
        all_envs = read_jsonl(configs.data)
    else:
        rawdata = build_dataset(configs.data)

    if configs.max_sample != -1:
        if isinstance(rawdata, Dataset):
            rawdata = rawdata.select(range(configs.max_sample))
        else:
            rawdata = rawdata[: configs.max_sample]

    if configs.start_index != -1:
        assert configs.end_index != -1, (
            "If start_index is set, end_index must be set as well"
        )
        if isinstance(rawdata, Dataset):
            rawdata = rawdata.select(
                range(configs.start_index, min(configs.end_index, len(rawdata)))
            )
        else:
            rawdata = rawdata[
                configs.start_index : min(configs.end_index, len(rawdata))
            ]

    if "uids" not in rawdata and isinstance(rawdata, Dataset):
        rawdata = rawdata.rename_column("uid", "uids")
    else:
        newrawdata = []
        for x in rawdata:
            x.update({"uids": x["uid"]})
            newrawdata.append(x)

    if (configs.model_path is not None) and configs.model_path != "":
        llm = build_language_model(
            configs.llm_name,
            configs.model_path,
            tokenizer=None,
        )
    else:
        llm = ModelRegistry.get(
            configs.llm_name,
        )
    llm: LanguageModel

    # Preprocess data
    MESSAGE_TEMPLATE = load_constants(configs.prompt_file)
    if configs.get("env_cls", "box1") == "box1":
        build_state_func = load_function(configs.prompt_file, "Map2Text")
    else:
        build_state_func = load_function(configs.prompt_file, "describe_obs")

    other_kwargs = {}
    # if "qwen3" in llm.get("model_name", ""):
    # other_kwargs["enable_thinking"] = True

    if configs.get("use_ray", False):
        ray.init(
            _temp_dir=get_cache_dir(),
        )

    prepare_prompt_fn = (
        (
            partial(
                llm.tokenizer.apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
                **other_kwargs,
            )
        )
        if isinstance(llm.tokenizer, PreTrainedTokenizerBase)
        else lambda x: x
    )

    eval_fn = True
    runner_cls = RUNNER_REGISTRY.get(configs.runner)
    env_cls = ENV_REGISTRY.get(configs.get("env_cls", "box1"))

    runner = runner_cls(
        configs,
        OUTPUTDIR,
        rawdata,
        configs.batch_size,
        llm,
        llm.tokenizer,
        eval_fn,
        configs.sample_params,
        MESSAGE_TEMPLATE=MESSAGE_TEMPLATE,
        build_state_func=build_state_func,
        prepare_prompt_fn=prepare_prompt_fn,
        env_cls=env_cls,
        use_ray=configs.get("use_ray", False),
    )

    runner.run(resume=configs.resume, test_key=configs.test_key)
    runner.stop()


if __name__ == "__main__":
    main()
