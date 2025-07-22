import os
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
from simulation.vis_api import TileMap

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


# TODO(Jiabao): we may put this into a runner
def run_one_round(
    OUTPUTDIR,
    configs,
    llm,
    all_envs,
    build_state_func,
    prepare_prompt_fn,
    MESSAGE_TEMPLATE,
):
    all_results = []
    for i in range(len(all_envs)):
        # for i in range(20, 30):
        env_json = all_envs[i]
        env = Box1Env.load(env_json)
        done = False

        outdir = os.path.join(OUTPUTDIR, f"env-{i:02}")
        os.makedirs(outdir, exist_ok=True)
        # 1 round interaction
        env.visualize(out_file_path=os.path.join(outdir, "_step_initial.png"))
        # Build state
        mapstate = build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )

        prompt = build_prompt(
            {"mapstate": mapstate}, MESSAGE_TEMPLATE, prepare_prompt_fn
        )
        action_strs = llm.batch_generate(prompt, **configs.sample_params)
        action_strs = reduce(lambda x, y: x + y, action_strs)

        for action_str in action_strs:
            try:
                # TODO (jiabao): we may use the simulate function in BoxEnv
                print(action_str["text"])
                all_actions = Action.from_str(action_str["text"], env.robots)
                done = False
                all_step_traj = []
                for step, step_action in enumerate(all_actions):
                    step_traj = {}
                    # update robot pos
                    for action in step_action:
                        action.arm_pos = env.robots[action.robot_id].arm_pos
                    # step_traj["action_str"] = action_str["text"]
                    to_break = False
                    if step_action is not None:
                        step_traj["actions"] = Action.to_str(step_action)
                        exec_res = env.verify(step_action)

                        if exec_res.success != ExecutionRes.Success:
                            print(f"Invalid actions: {step_action}")
                            step_traj["status"] = "failed"
                            to_break = True
                        else:
                            print("Valid Actions:", step_action)
                            step_traj["status"] = "success"
                            to_break = False
                    else:
                        step_traj["status"] = "invalid"
                    env.visualize(
                        actions=step_action,
                        exec_res=exec_res,
                        out_file_path=os.path.join(outdir, f"step@{step:02}.png"),
                    )
                    all_step_traj.append(step_traj)
                    if to_break:
                        break
                    else:
                        env.simulate(step_action)

                if len(all_step_traj) != len(all_actions):
                    all_step_traj.extend(
                        [
                            {"actions": Action.to_str(a), "status": "previous_failed"}
                            for a in all_actions[len(all_step_traj) :]
                        ]
                    )
            except Exception as e:
                print(e)
                done = False
                all_step_traj = [{"actions": "error", "status": "error"}]

            if env.check_final():
                done = True
                print("Objects are all in target positions")
            if not done:
                print("Objects failed to be put at target positions")
            all_results.append(
                {
                    "env": env_json,
                    "final_success": done,
                    "traj": object_to_json_serializable(all_step_traj),
                    **action_str,
                }
            )
        with open(os.path.join(OUTPUTDIR, f"env-{i:02}.json"), "w") as f:
            json.dump(all_results[i], f, indent=2)


# TODO(Jiabao): we may put this into a runner
def run_multi_round(
    OUTPUTDIR,
    configs,
    llm,
    all_envs,
    build_state_func,
    prepare_prompt_fn,
    MESSAGE_TEMPLATE,
):
    all_results = []
    for i in range(len(all_envs)):
        # for i in range(20, 30):
        env_json = all_envs[i]
        env = Box1Env.load(env_json)
        done = False

        outdir = os.path.join(OUTPUTDIR, f"env-{i:02}")
        os.makedirs(outdir, exist_ok=True)
        # 1 round interaction
        env.visualize(out_file_path=os.path.join(outdir, "_step_initial.png"))

        MAX_STEP = 20
        all_step_traj = []
        done = False
        for STEP in range(MAX_STEP):
            # Build state
            mapstate = build_state_func(
                env.map,
                env.objects,
                env.targets,
                {k: v.to_tuple() for k, v in env.robots.items()},
            )

            prompt = build_prompt(
                {"mapstate": mapstate}, MESSAGE_TEMPLATE, prepare_prompt_fn
            )
            import ipdb

            ipdb.set_trace()
            action_strs = llm.batch_generate(prompt, **configs.sample_params)
            action_strs = reduce(lambda x, y: x + y, action_strs)

            to_break = False
            for action_str in action_strs:
                try:
                    print(action_str["text"])
                    step_traj = {}
                    step_action = Action.from_str(action_strs[0]["text"], env.robots)
                    step_traj.update(**action_str)
                    to_break = False
                    if step_action is not None:
                        step_traj["actions"] = Action.to_str(step_action)
                        exec_res = env.verify(step_action)

                        if exec_res.success != ExecutionRes.Success:
                            print(f"Invalid actions: {step_action}")
                            step_traj["status"] = "failed"
                            to_break = True
                        else:
                            print("Valid Actions:", step_action)
                            step_traj["status"] = "success"
                            to_break = False
                    else:
                        step_traj["status"] = "invalid"
                    env.visualize(
                        actions=step_action,
                        exec_res=exec_res,
                        out_file_path=os.path.join(outdir, f"step@{STEP:02}.png"),
                    )
                    all_step_traj.append(step_traj)
                except Exception as e:
                    print(e)
                    done = False
                    to_break = True
                    all_step_traj.append({"actions": "error", "status": "error"})

                if to_break:
                    break
                else:
                    env.simulate(step_action)

                if env.check_final():
                    done = True
                    break
            if done:
                print("Objects are all in target positions")
                break
            else:
                if to_break:
                    print("Failed environment")
                    break
                else:
                    print("Not done yet, continue to next step")
                    print("Objects: ", env.objects)
                    print("Targets: ", env.targets)

        all_results.append(
            {
                "env": env_json,
                "final_success": done,
                "traj": object_to_json_serializable(all_step_traj),
                **action_str,
            }
        )

        with open(os.path.join(OUTPUTDIR, f"env-{i:02}.json"), "w") as f:
            json.dump(all_results[i], f, indent=2)


def run_multi_round_withhistory(
    OUTPUTDIR,
    configs,
    llm,
    all_envs,
    build_state_func,
    prepare_prompt_fn,
    MESSAGE_TEMPLATE,
):
    # TODO (jiabao): we need to implement this
    all_results = []

    for i in range(len(all_envs)):
        # for i in range(20, 30):
        env_json = all_envs[i]
        env = Box1Env.load(env_json)
        done = False

        outdir = os.path.join(OUTPUTDIR, f"env-{i:02}")
        os.makedirs(outdir, exist_ok=True)
        # 1 round interaction
        env.visualize(out_file_path=os.path.join(outdir, "_step_initial.png"))

        MAX_STEP = 20
        all_step_traj = []
        done = False
        for STEP in range(MAX_STEP):
            # Build state
            mapstate = build_state_func(
                env.map,
                env.objects,
                env.targets,
                {k: v.to_tuple() for k, v in env.robots.items()},
            )

            prompt = build_prompt(
                {"mapstate": mapstate}, MESSAGE_TEMPLATE, prepare_prompt_fn
            )
            action_strs = llm.batch_generate(prompt, **configs.sample_params)

            action_strs = reduce(lambda x, y: x + y, action_strs)

            to_break = False
            for action_str in action_strs:
                try:
                    print(action_str["text"])
                    step_traj = {}
                    step_action = Action.from_str(action_strs[0]["text"], env.robots)
                    step_traj.update(**action_str)
                    to_break = False
                    if step_action is not None:
                        step_traj["actions"] = Action.to_str(step_action)
                        exec_res = env.verify(step_action)

                        if exec_res.success != ExecutionRes.Success:
                            print(f"Invalid actions: {step_action}")
                            step_traj["status"] = "failed"
                            to_break = True
                        else:
                            print("Valid Actions:", step_action)
                            step_traj["status"] = "success"
                            to_break = False
                    else:
                        step_traj["status"] = "invalid"
                    env.visualize(
                        actions=step_action,
                        exec_res=exec_res,
                        out_file_path=os.path.join(outdir, f"step@{STEP:02}.png"),
                    )
                    all_step_traj.append(step_traj)
                except Exception as e:
                    print(e)
                    done = False
                    to_break = True
                    all_step_traj.append({"actions": "error", "status": "error"})

                if to_break:
                    break
                else:
                    env.simulate(step_action)

                if env.check_final():
                    done = True
                    break
            if done:
                print("Objects are all in target positions")
                break
            else:
                print("Not done yet, continue to next step")
                print("Objects: ", env.objects)
                print("Targets: ", env.targets)

            if to_break:
                break

        all_results.append(
            {
                "env": env_json,
                "final_success": done,
                "traj": object_to_json_serializable(all_step_traj),
                **action_str,
            }
        )

        with open(os.path.join(OUTPUTDIR, f"env-{i:02}.json"), "w") as f:
            json.dump(all_results[i], f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="box1_config")
def main(configs: RunConfig):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    env_config = configs.env_config

    seed_everything(configs.seed)

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

    # all_envs = build_dataset(configs.data)
    if configs.data.endswith(".jsonl"):
        all_envs = read_jsonl(configs.data)
    else:
        rawdata = build_dataset(configs.data)
        all_envs = []
        for sample in rawdata:
            all_envs.append(json.loads(sample["reward_model"]["ground_truth"]))

    # Preprocess data
    MESSAGE_TEMPLATE = load_constants(configs.prompt_file)
    build_state_func = load_function(configs.prompt_file, "Map2Text")
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

    if configs.mode == "one_round":
        run_one_round(
            OUTPUTDIR,
            configs,
            llm,
            all_envs,
            build_state_func,
            prepare_prompt_fn,
            MESSAGE_TEMPLATE,
        )
    elif configs.mode == "multi_round":
        run_multi_round(
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
