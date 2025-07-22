# This file defines a runner class that runs evaluation in batches, and evaluates, and with a monitor class that supports experiment restoration.

import os
import ray
import re
import time
import json
import torch
import shutil
import logging
import inspect
import structlog
import concurrent
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import Dict, Any, Callable, List
from pathlib import Path
from datasets import Dataset
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from functools import reduce

from inference.eval_utils import pass_at_k
from inference.config_store import RunConfig
from inference.data_utils import batchify_sampler, DataDict, list_repeat_interleave
from inference.api import SampleParams
from inference.registry_utils import Registry
from prompts.prompt_utils import build_prompt, replace_prompt
from simulation.simenv.box1 import Box1Env
from simulation.simenv.box3d.box3d import Box3DEnv

LOGGER = structlog.getLogger(__name__)


class SequentialRunner:
    def __init__(
        self,
        configs: RunConfig,
        rundir: str,
        dataset: Dataset,
        batch_size: int,
        model: AutoModelForCausalLM,  # We assume that the model has a function generate for generating texts
        tokenizer: AutoTokenizer,
        eval_fn: Callable,
        sampling_params: SampleParams,
        **kwargs,
    ):
        self.monitor = Monitor(rundir)
        self.batch_results = []
        self.metrics = {}
        self.configs = configs

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sampling_params = sampling_params
        self.eval_fn = eval_fn

        # For logging
        self.cur_batch_result = None

    def log(self):
        pass

    @torch.no_grad()
    def consume_batch(
        self, batch: DataDict, test_key: str = "reward_model"
    ) -> Dict[str, Any]:
        pass

    def summarize_batch_res(self, batch_res: DataDict):
        batch_error_types = Counter()
        if any([k not in batch_res for k in ["success", "detail"]]):
            return {}
        for error_types in batch_res["detail"]:
            batch_error_types.update({error_types: 1})
        return {
            "batch-score": np.mean(batch_res["success"]).item(),
            "batch-error-types": batch_error_types,
        }

    def run(self, resume=False, test_key="reward_model"):
        all_indexes = batchify_sampler(
            len(self.dataset), self.batch_size, shuffle=False
        )
        if not resume:
            # If not resuming, clear any previous state.
            self.monitor.reset()
            start_batch_index = 0
            LOGGER.info("Evaluation from start")
        else:
            # When resuming, skip over the batches already processed.
            start_batch_index = self.monitor.num_batches
            LOGGER.info("Resuming from batch", start_batch_index)

        for idx, indexes in enumerate(
            tqdm(
                all_indexes[start_batch_index:],
                desc="Running batch",
                initial=start_batch_index,
            )
        ):
            batch = DataDict.from_list_of_dicts([self.dataset[i] for i in indexes])
            with self.monitor.track_batch_time():
                batch_res = self.consume_batch(batch, test_key)
            self.monitor.update(batch_res)
            self.monitor.save()

    def stop(self):
        self.metrics = self.monitor.stop()
        LOGGER.info("Stopping the runner and saving the state")
        LOGGER.info("Final state", metrics=self.metrics, state=self.monitor.state)
        self.monitor.dump_results()


class Monitor:
    def __init__(self, rundir: str, resume: bool = True):
        self.rundir = Path(rundir)
        if not self.rundir.exists():
            self.rundir.mkdir(parents=True)
        self.cache_dir = self.rundir / "cached_res"
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        self.consumed_samples = 0
        self.total_time = 0  # Track total execution time
        self.num_batches = 0  # Track number of batches processed
        self.all_results = []
        self.full_res = None

        if resume:
            self.restore()

    @property
    def state(self):
        return {
            "consumed_samples": self.consumed_samples,
            "average_time": self.average_time,
            "num_batches": self.num_batches,
            "total_time": self.total_time,
        }

    @property
    def average_time(self):
        """Returns the average batch processing time."""
        return self.total_time / self.num_batches if self.num_batches > 0 else 0

    @contextmanager
    def track_batch_time(self):
        """Context function to track batch execution time."""
        start_time = time.time()
        yield  # Execution of the wrapped function happens here
        end_time = time.time()

        batch_time = end_time - start_time
        self.total_time += batch_time
        self.num_batches += 1

    def update(self, batch_res: Dict[str, Any]):
        self.consumed_samples += batch_res["batch_num"]
        self.all_results.append(batch_res["batch_res"])
        if self.cache_dir.exists():
            with open(str(self.cache_dir / f"batch-{self.num_batches}.json"), "w") as f:
                if isinstance(batch_res["batch_res"], DataDict):
                    json.dump(batch_res["batch_res"].to_dict(), f, indent=2)
                else:
                    json.dump(batch_res, f, indent=2)

    def save(self):
        with open(str(self.rundir / "states.json"), "w") as f:
            json.dump(self.state, f, indent=2)

    def restore(self):
        state_path = self.rundir / "states.json"
        if state_path.exists():
            states = json.load(open(str(state_path)))

            for k, v in states.items():
                assert hasattr(self, k), (
                    f"State key {k} not found in the Monitor object"
                )
                try:
                    setattr(self, k, v)
                except Exception as e:
                    LOGGER.error(f"Error restoring state {k}: {e}")

            if self.cache_dir.exists():
                self.all_results = [
                    DataDict.from_dict(json.load(open(str(f))))
                    for f in self.cache_dir.iterdir()
                ]

            LOGGER.info(f"Resuming from previous state, {states['consumed_samples']}")

    def reset(self):
        """Reset internal counters and remove any saved state."""
        self.consumed_samples = 0
        self.total_time = 0
        self.num_batches = 0
        state_path = self.rundir / "states.json"
        if state_path.exists():
            state_path.unlink()

    def stop(self):
        # merge pass@k results
        metric = {}
        # all_res = reduce(lambda x, y: x.union(y), self.all_results)
        all_res = DataDict.concat(self.all_results)
        if "success" in all_res and "uid" in all_res:
            uids = all_res["uid"]
            accs = all_res["success"]
            self.full_res = all_res
            for k in [1, 2, 4, 8]:
                passk = pass_at_k(uids, accs, k)
                metric[k] = passk

        if "detail" in all_res:
            error_types = Counter(all_res["detail"])
            metric.update({"Error Type": error_types})

        return metric

    def dump_results(self):
        all_results = DataDict.concat(self.all_results)
        # assert self.full_res is not None
        # all_results = self.full_res
        dataset = Dataset.from_dict(all_results.to_dict())
        dataset.save_to_disk(str(self.rundir / "finalresult"))
        # shutil.rmtree(str(self.cache_dir))


# gt_data = load_from_disk(
#     "/mnt/data/jiabao/2025-Projects/MultiRobotR1/data_cache/robot/box1-large-test-subset"
# )
# gt_env_mapping = {x["uid"]: x["reward_model"]["ground_truth"] for x in gt_data}


def add_key(dataset):
    def fun(sample):
        if "reward_model" not in sample:
            conf = json.loads(gt_env_mapping[sample["uid"]])
        else:
            conf = json.loads(sample["reward_model"]["ground_truth"])
        sample["key"] = (
            f"n@{conf['grid_n']}_m@{conf['grid_m']}_k@{conf['num_objects']}_mode@{conf['robot_mode']}"
        )
        return sample

    dataset = dataset.map(fun, num_proc=8)
    return dataset


from datasets.utils.logging import disable_progress_bar, enable_progress_bar


@contextmanager
def tmp_disable_progress_bar():
    disable_progress_bar()
    try:
        yield
    finally:
        enable_progress_bar()


def group_by_nm(sample):
    key = sample["key"]
    n = int(key.split("_")[0].split("@")[1])
    m = int(key.split("_")[1].split("@")[1])
    return f"n@{n}_m@{m}"


def group_by_k(sample):
    key = sample["key"]
    k = int(key.split("_")[2].split("@")[1])
    return f"k@{k}"


def group_by_nmk(sample):
    key = sample["key"]
    n = int(key.split("_")[0].split("@")[1])
    m = int(key.split("_")[1].split("@")[1])
    k = int(key.split("_")[2].split("@")[1])
    return f"n@{n}_m@{m}_k@{k}"


class FullPlanRunner(SequentialRunner):
    def __init__(
        self,
        configs,
        rundir,
        dataset,
        batch_size,
        model,
        tokenizer,
        eval_fn,
        sampling_params,
        MESSAGE_TEMPLATE,
        build_state_func,
        prepare_prompt_fn,
        env_cls=Box1Env,
        **kwargs,
    ):
        super().__init__(
            configs,
            rundir,
            dataset,
            batch_size,
            model,
            tokenizer,
            eval_fn,
            sampling_params,
            **kwargs,
        )

        self.MESSAGE_TEMPLATE = MESSAGE_TEMPLATE
        self.build_state_func = build_state_func
        self.prepare_prompt_fn = prepare_prompt_fn
        self.env_cls = env_cls
        self.use_ray = kwargs.get("use_ray", False)

    def organize_outputs(self, outputs):
        outputs = reduce(
            lambda x, y: x + y,
            [
                [
                    {
                        "decode_id": response.pop("decode_id"),
                        "output_text": response.pop("text"),
                        "output_len": response.pop("len") if "len" in response else 0,
                    }
                    for response in output
                ]
                for output in outputs
            ],
        )
        return outputs

    def get_env_prompt(self, env: Box1Env):
        map_states = self.build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )
        prompt = build_prompt(
            {"mapstate": map_states},
            self.MESSAGE_TEMPLATE,
            self.prepare_prompt_fn,
        )
        return prompt

    def consume_batch(self, batch, test_key="reward_model"):
        batch_envs = [x["ground_truth"] for x in batch["reward_model"]]
        batch_envs = [
            json.loads(env_str) if isinstance(env_str, str) else env_str
            for env_str in batch_envs
        ]
        n = self.sampling_params.n
        all_envs = [self.env_cls.load(x) for x in batch_envs]
        # Build prompt, then expand
        all_inputs = [self.get_env_prompt(env) for env in all_envs]
        gen_batch = DataDict.from_dict(
            {
                "input": all_inputs,
                "uid": batch["uids"],
                "reward_model": batch["reward_model"],
            }
        )
        gen_batch = gen_batch.repeat_interleave(n)
        all_envs = list_repeat_interleave(all_envs, n)

        outputs = self.model.batch_generate(all_inputs, **self.sampling_params)
        outputs = self.organize_outputs(outputs)
        gen_batch = gen_batch.union(DataDict.from_list_of_dicts(outputs))

        if not self.use_ray:
            eval_res = [
                env.simulate_all_str(response, return_step_action=False)
                for env, response in zip(all_envs, gen_batch["output_text"])
            ]
        else:
            futures = [
                ray_simulate_all.remote(self.env_cls, env.to_json(), response)
                for env, response in zip(all_envs, gen_batch["output_text"])
            ]
            eval_res = ray.get(futures)

        eval_res = DataDict.from_list_of_dicts(eval_res)
        gen_batch = gen_batch.union(eval_res)
        LOGGER.info("Batch result", batch_res=self.summarize_batch_res(gen_batch))

        return {"batch_num": len(all_inputs), "batch_res": gen_batch}


class FullPlan3DRunner(FullPlanRunner):
    def get_env_prompt(self, env: Box3DEnv):
        obs = env.obs
        targets = env.targets
        map_states = self.build_state_func(obs, targets)
        prompt = build_prompt(
            {"mapstate": map_states},
            self.MESSAGE_TEMPLATE,
            self.prepare_prompt_fn,
        )
        return prompt


class StepPlanRunner(FullPlanRunner):
    def __init__(
        self,
        configs,
        rundir,
        dataset,
        batch_size,
        model,
        tokenizer,
        eval_fn,
        sampling_params,
        MESSAGE_TEMPLATE,
        build_state_func,
        prepare_prompt_fn,
        env_cls=Box1Env,
        **kwargs,
    ):
        super().__init__(
            configs,
            rundir,
            dataset,
            batch_size,
            model,
            tokenizer,
            eval_fn,
            sampling_params,
            MESSAGE_TEMPLATE,
            build_state_func,
            prepare_prompt_fn,
            env_cls=env_cls,
            **kwargs,
        )

    def summarize_batch_res(self, batch_res: List[Dict[str, Any]]):
        successes = []
        batch_error_types = []

        for traj_item in batch_res:
            successes.append(traj_item[-1]["detail"] == "Success")
            batch_error_types.append(traj_item[-1]["detail"])

        return {
            "batch-score": np.mean(successes).item(),
            "batch-error-types": Counter(batch_error_types),
        }

    def consume_batch(self, batch: DataDict, test_key="reward_model"):
        env_configs = [x["ground_truth"] for x in batch["reward_model"]]
        uids = batch["uids"]
        old_n = self.sampling_params.n
        n = self.sampling_params.n
        copyids = list(range(len(uids)))
        uids = list_repeat_interleave(uids, n)
        copyids = list_repeat_interleave(copyids, n)
        env_configs = list_repeat_interleave(env_configs, n)
        self.sampling_params.n = 1  # overwrite

        if isinstance(env_configs[0], str):
            envs = [self.env_cls.load(json.loads(x)) for x in env_configs]
        else:
            envs = [self.env_cls.load(x) for x in env_configs]
        current_envs = envs
        current_indices = [i for i in range(len(current_envs))]
        done = [False] * len(envs)
        traj = [[] for _ in range(len(current_envs))]

        MAX_STEPS = 20
        current_prompts = [self.get_env_prompt(env) for env in current_envs]
        for cur_step in tqdm(range(MAX_STEPS)):
            if len(current_envs) == 0:
                break

            if cur_step > 0:
                prompts = [self.get_env_prompt(env) for env in current_envs]
            else:
                prompts = current_prompts
            outputs = self.model.batch_generate(
                prompts, **self.sampling_params
            )  # per-env steps
            outputs = self.organize_outputs(outputs)
            output_texts = [x["output_text"] for x in outputs]

            step_res = [
                env.simulate_one_step_from_str(plan_str)
                for env, plan_str in zip(current_envs, output_texts)
            ]

            # update per environment
            next_envs = []
            next_indices = []

            for i, env in enumerate(current_envs):
                origin_indice = current_indices[i]
                env_step_res = step_res[i]
                if env_step_res["success"]:  # successfully executed
                    if env_step_res["detail"] == "Success":  # finished
                        done[origin_indice] = True
                    else:
                        done[origin_indice] = False
                        next_envs.append(env)  # not finished and also not error
                        next_indices.append(origin_indice)
                else:
                    done[origin_indice] = True

                traj[origin_indice].append(
                    {
                        "inputs": prompts[i],
                        "step_ids": cur_step,
                        "uids": uids[origin_indice],
                        "copyids": copyids[origin_indice],
                        **outputs[i],
                        **env_step_res,
                    }
                )

            current_envs = next_envs
            current_indices = next_indices

        LOGGER.info("Batch result", batch_res=self.summarize_batch_res(traj))
        res_traj = [
            {
                "uid": traj_item[0]["uids"],
                "success": traj_item[-1]["detail"] == "Success",
                "traj-detail": traj_item,
                "traj_len": len(traj_item),
                "reward_model": env_configs[idx],
            }
            for idx, traj_item in enumerate(traj)
        ]
        self.sampling_params.n = old_n  # restore
        return {
            "batch_num": len(traj),
            "batch_res": DataDict.from_list_of_dicts(res_traj),
        }


class ToolPlanRunner(StepPlanRunner):
    def get_obs_prompt(self, env):
        self.OBSERVATION_TEMPLATE = """<observation>\n{mapstate}\n</observation>"""
        if not isinstance(env, Box1Env):
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.map,
            env.objects,
            env.targets,
            {k: v.to_tuple() for k, v in env.robots.items()},
        )
        prompt = replace_prompt(
            [{"content": self.OBSERVATION_TEMPLATE}],
            {"mapstate": mapstate},
        )[0]["content"]
        return prompt

    def update_env_prompt(self, env, prefix_prompt):
        obs = self.get_obs_prompt(env)
        return prefix_prompt + "\n" + obs

    def consume_batch(self, batch, test_key="reward_model"):
        env_configs = [x["ground_truth"] for x in batch["reward_model"]]
        uids = batch["uids"]
        old_n = self.sampling_params.n
        n = self.sampling_params.n
        copyids = list(range(len(uids)))
        uids = list_repeat_interleave(uids, n)
        copyids = list_repeat_interleave(copyids, n)
        env_configs = list_repeat_interleave(env_configs, n)
        self.sampling_params.n = 1  # overwrite

        if isinstance(env_configs[0], str):
            envs = [self.env_cls.load(json.loads(x)) for x in env_configs]
        else:
            envs = [self.env_cls.load(x) for x in env_configs]
        current_envs = envs
        current_indices = [i for i in range(len(current_envs))]
        done = [False] * len(envs)
        traj = [[] for _ in range(len(current_envs))]

        current_prompts = [self.get_env_prompt(env) for env in current_envs]

        MAX_STEPS = 20
        for cur_step in tqdm(range(MAX_STEPS)):
            if len(current_envs) == 0:
                break

            if cur_step > 0:
                prompts = [
                    self.update_env_prompt(env, prompt)
                    for env, prompt in zip(current_envs, current_prompts)
                ]
            else:
                prompts = current_prompts
            outputs = self.model.batch_generate(
                prompts, **self.sampling_params
            )  # per-env steps

            outputs = self.organize_outputs(outputs)
            output_texts = [x["output_text"] for x in outputs]

            # Find the first ```json ``` block and stop there
            def extract_first_json(text):
                matches = list(re.finditer(r"```json\n(.*?)\n```", text, re.DOTALL))
                if matches:
                    first_match = matches[0]
                    json_block = f"""```json\n{first_match.group(1)}\n```"""
                    prev = text[: first_match.start()]
                    position = (first_match.start(), first_match.end())
                    return prev + json_block
                return "", (-1, -1)

            postprocess_output_texts = [
                extract_first_json(output_text) for output_text in output_texts
            ]
            output_texts = postprocess_output_texts
            step_res = [
                env.simulate_one_step_from_str(plan_str)
                for env, plan_str in zip(current_envs, output_texts)
            ]

            # update per environment
            next_envs = []
            next_indices = []
            next_prompts = []

            for i, env in enumerate(current_envs):
                origin_indice = current_indices[i]
                env_step_res = step_res[i]
                if env_step_res["success"]:  # successfully executed
                    if env_step_res["detail"] == "Success":  # finished
                        done[origin_indice] = True
                    else:
                        done[origin_indice] = False
                        tmpprompt = prompts[i] + output_texts[i].strip()
                        has_next_round = True
                        # if isinstance(self.tokenizer, AutoTokenizer):
                        if "transformers" in str(type(self.tokenizer)):
                            promptlen = self.tokenizer(tmpprompt).input_ids
                            if (
                                len(promptlen)
                                + self.sampling_params.get("max_tokens", 8192)
                                > 32 * 1024
                            ):
                                has_next_round = False

                        if has_next_round:
                            next_envs.append(env)  # not finished and also not error
                            next_indices.append(origin_indice)
                            next_prompts.append(tmpprompt)
                            done[origin_indice] = False
                        else:
                            done[origin_indice] = True
                else:
                    done[origin_indice] = True

                traj[origin_indice].append(
                    {
                        "inputs": prompts[i],
                        "step_ids": cur_step,
                        "uids": uids[origin_indice],
                        "copyids": copyids[origin_indice],
                        **outputs[i],
                        **env_step_res,
                    }
                )

            current_envs = next_envs
            current_indices = next_indices
            current_prompts = next_prompts

        LOGGER.info("Batch result", batch_res=self.summarize_batch_res(traj))
        res_traj = [
            {
                "uid": traj_item[0]["uids"],
                "success": traj_item[-1]["detail"] == "Success",
                "traj-detail": traj_item,
                "traj_len": len(traj_item),
                "reward_model": env_configs[idx],
            }
            for idx, traj_item in enumerate(traj)
        ]
        self.sampling_params.n = old_n  # restore
        return {
            "batch_num": len(traj),
            "batch_res": DataDict.from_list_of_dicts(res_traj),
        }


class ToolPlan3DRunner(ToolPlanRunner):
    def get_env_prompt(self, env):
        if not isinstance(env, Box3DEnv):
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.obs,
            env.targets,
        )
        prompt = build_prompt(
            {"mapstate": mapstate},
            self.MESSAGE_TEMPLATE,
            self.prepare_prompt_fn,
        )
        return prompt

    def get_obs_prompt(self, env: Box3DEnv):
        self.OBSERVATION_TEMPLATE = """<observation>\n{mapstate}\n</observation>"""
        if not isinstance(env, Box3DEnv):
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.obs,
            env.targets,
        )
        prompt = replace_prompt(
            [{"content": self.OBSERVATION_TEMPLATE}],
            {"mapstate": mapstate},
        )[0]["content"]
        return prompt


class StepPlan3DRunner(StepPlanRunner):
    def get_env_prompt(self, env):
        if not isinstance(env, Box3DEnv):
            env = self.env_cls.load(env)

        mapstate = self.build_state_func(
            env.obs,
            env.targets,
        )
        prompt = build_prompt(
            {"mapstate": mapstate},
            self.MESSAGE_TEMPLATE,
            self.prepare_prompt_fn,
        )
        return prompt

    # def get_obs_prompt(self, env: Box3DEnv):
    #     self.OBSERVATION_TEMPLATE = """<observation>\n{mapstate}\n</observation>"""
    #     if not isinstance(env, Box3DEnv):
    #         env = self.env_cls.load(env)

    # mapstate = self.build_state_func(
    #     env.obs,
    #     env.targets,
    # )
    # prompt = replace_prompt(
    #     [{"content": self.OBSERVATION_TEMPLATE}],
    #     {"mapstate": mapstate},
    # )[0]["content"]
    # return prompt


RUNNER_REGISTRY = Registry("RunnerCLS")
RUNNER_REGISTRY.register("sequential", SequentialRunner)
RUNNER_REGISTRY.register("fullplan", FullPlanRunner)
RUNNER_REGISTRY.register("full3dplan", FullPlan3DRunner)
RUNNER_REGISTRY.register("stepplan", StepPlanRunner)
RUNNER_REGISTRY.register("toolplan", ToolPlanRunner)
RUNNER_REGISTRY.register("tool3dplan", ToolPlan3DRunner)
RUNNER_REGISTRY.register("step3dplan", StepPlan3DRunner)


@ray.remote(num_cpus=1)
def ray_simulate_all(env_cls, env_json, solustion_str):
    env = env_cls.load(env_json)
    return env.simulate_all_str(solustion_str, return_step_action=False)
