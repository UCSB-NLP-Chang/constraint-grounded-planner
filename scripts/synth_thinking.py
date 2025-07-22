import os
import json
import copy
import hydra
import click
import random
import numpy as np
from tqdm import tqdm
from functools import partial, reduce
from itertools import product
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase
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

from inference.data_utils import read_jsonl, build_dataset
from inference.language_models import ModelRegistry, LanguageModel
from inference.config_store import RunConfig
from inference.utils import seed_everything, init_script
from inference.import_utils import load_function
from prompts.prompt_utils import load_constants, build_prompt


# TODO (Jiabao): we may put this into a runner
def synth_thinking_step(
    OUTPUTDIR,
    configs,
    llm,
    all_envs,
    build_state_func,
    prepare_prompt_fn,
    MESSAGE_TEMPLATE,
):
    all_results = []

    with open(os.path.join(OUTPUTDIR, "thinking.jsonl"), "w") as f:
        for i in tqdm(range(len(all_envs)), desc="Generating thinking single steps"):
            new_results = []
            raw_plan = all_envs[i]
            env_json = raw_plan["env"]
            env = Box1Env.load(env_json)

            # Generate the prompt
            for step_id, step_action in enumerate(raw_plan["plan"]):
                cur_env_json = env.to_json()
                this_plan = {"env": cur_env_json, "plan": step_action}
                map_state = build_state_func({"env": cur_env_json, "plan": step_action})
                prompt = build_prompt(
                    {"environment": map_state}, MESSAGE_TEMPLATE, prepare_prompt_fn
                )

                gen_thinkings = llm.batch_generate(prompt, **configs.sample_params)
                gen_thinkings = reduce(lambda x, y: x + y, gen_thinkings)

                this_plans = [
                    copy.deepcopy(this_plan) for _ in range(len(gen_thinkings))
                ]
                new_results.extend(
                    [
                        {**plan, "thinking": gen_thinking, "uid": i}
                        for plan, gen_thinking in zip(this_plans, gen_thinkings)
                    ]
                )
                all_results.extend(new_results)
                env.simulate([Action(**x) for x in step_action])

            for x in new_results:
                f.write(json.dumps(x) + "\n")


# TODO(Jiabao): we may put this into a runner
def synth_thinking_full(
    OUTPUTDIR,
    configs,
    llm: LanguageModel,
    all_envs,
    build_state_func,
    prepare_prompt_fn,
    MESSAGE_TEMPLATE,
):
    all_results = []
    with open(os.path.join(OUTPUTDIR, "thinking.jsonl"), "w") as f:
        cur_results = []
        for i in tqdm(range(len(all_envs))):
            plan = all_envs[i]

            # Generate the prompt
            map_state = build_state_func(plan)
            prompt = build_prompt(
                {"environment": map_state}, MESSAGE_TEMPLATE, prepare_prompt_fn
            )

            gen_thinkings = llm.batch_generate(prompt, **configs.sample_params)
            gen_thinkings = reduce(lambda x, y: x + y, gen_thinkings)

            this_plans = [copy.deepcopy(plan) for _ in range(len(gen_thinkings))]
            cur_results = [
                {**plan, "thinking": gen_thinking, "uid": i}
                for plan, gen_thinking in zip(this_plans, gen_thinkings)
            ]
            all_results.extend(cur_results)

            for res in cur_results:
                f.write(json.dumps(res) + "\n")


@hydra.main(
    version_base=None, config_path="../configs", config_name="synth_thinking_config"
)
def main(configs: RunConfig):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    env_config = OmegaConf.to_container(configs.env_config)
    seed_everything(configs.seed)

    llm = ModelRegistry.get(configs.llm_name)
    llm: LanguageModel

    # Load the environment
    all_envs = build_dataset(configs.data)
    all_envs = [
        {
            "env": json.loads(x["reward_model"]["ground_truth"]),
            "plan": json.loads(x["extra_info"]["gt_plan"]),
        }
        for x in all_envs
    ]

    # Preprocess data
    MESSAGE_TEMPLATE = load_constants(configs.prompt_file)
    build_state_func = load_function(configs.prompt_file, "Plan2Text")
    prepare_prompt_fn = (
        (
            partial(
                llm.tokenizer.apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        if isinstance(llm.tokenizer, PreTrainedTokenizerBase)
        else lambda x: x
    )

    if configs.mode == "onestep":
        synth_thinking_step(
            OUTPUTDIR,
            configs,
            llm,
            all_envs,
            build_state_func,
            prepare_prompt_fn,
            MESSAGE_TEMPLATE,
        )
    elif configs.mode == "fullplan":
        synth_thinking_full(
            OUTPUTDIR,
            configs,
            llm,
            all_envs,
            build_state_func,
            prepare_prompt_fn,
            MESSAGE_TEMPLATE,
        )
    else:
        raise ValueError("Unkown mode")


if __name__ == "__main__":
    main()
