import re
import ray
import json
import multiprocessing

from simulation.simenv.box1 import Box1Env
from simulation.simenv.box3d.box3d import Box3DEnv

from custom_verl.reward_utils import RewardType

reason_reward_mapping = {
    "Success": RewardType.Correct,
    "StepSuccess": RewardType.StepSuccess,
    "ParseError": RewardType.ParseError,
    "StepParseError": RewardType.StepParseError,
    "InvalidAction": RewardType.InvalidAction,
    "CollisionObject": RewardType.CollisionObject,
    "CollisionRobot": RewardType.CollisionRobot,
    "TimeoutError": RewardType.TimeoutError,
}


def extract_json(text):
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    res = match.group(1) if match else ""

    if res != "":
        res = f"""```json\n{res}\n```"""
    return res


def extract_last_json(text):
    text = text.split("</think>")[-1]
    matches = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
    if matches:
        res = f"""```json\n{matches[-1]}\n```"""
        return res
    return ""


# Adapted from https://github.com/huggingface/open-r1/blob/d436b7b9c0e9205a2d329596273ca0600a794f54/src/open_r1/rewards.py#L70
def format_reward(response):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    if "<think>" not in response:
        # print("Missing <think> tag in response")
        # print(response)
        return 0.0
    else:
        # Remove potential prefix
        def get_response(text, start):
            return text[text.find(start) :] if start in text else ""

        response = get_response(response, "<think>")
        if response.endswith("<|im_end|>"):
            response = response[: response.rfind("<|im_end|>")]
        # print(response)
        pattern = r"^<think>.*?</think>.*?```json.*?```$"
        matches = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        return 1.0 if matches else 0.0


def compute_score(
    solution_str,
    env_config,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    format_r = format_reward(solution_str)
    plan_str = extract_fn(solution_str)
    if isinstance(env_config, str):
        env_config = json.loads(env_config)

    env = Box1Env.load(env_config)
    res = env.simulate_all_str(plan_str)

    expected_steps = len(json.loads(env_config["gt_plan"]))
    if res["success"]:
        # final_reward = format_r * format_score + 1.0
        # reward_type = RewardType.Correct
        tmp_r = format_r * format_score + 1.0  # full reward
        actual_steps = res["traj_len"]
        # we add penalty on the steps
        final_reward = tmp_r
        reward_type = RewardType.Correct
    else:
        final_reward = format_score * format_r + 0.0
        reward_type = reason_reward_mapping.get(res["detail"], RewardType.Wrong)

    if return_dict:
        return {
            "reward": final_reward,
            "reward_type": reward_type,
            "traj_len": res.get("traj_len", -1),
            "parallelism": res.get("parallelism", -1),
        }
    else:
        return (final_reward, reward_type)


def compute_score_with_step_penalty(
    solution_str,
    env_config,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    format_r = format_reward(solution_str)
    plan_str = extract_fn(solution_str)
    if isinstance(env_config, str):
        env_config = json.loads(env_config)

    env = Box1Env.load(env_config)
    res = env.simulate_all_str(plan_str, return_step_action=False)

    expected_steps = len(json.loads(env_config["gt_plan"]))
    if res["success"]:
        tmp_r = format_r * format_score + 1.0  # full reward
        # we add penalty on the steps
        actual_steps = res["traj_len"]
        # we add penalty on the steps
        if actual_steps > expected_steps:
            tmp_r -= (actual_steps - expected_steps) * 0.1
            tmp_r = max(tmp_r, 2 * format_score)  # at least bigger than only format
        final_reward = tmp_r
        reward_type = RewardType.Correct
    else:
        final_reward = format_score * format_r + 0.0
        reward_type = reason_reward_mapping.get(res["detail"], RewardType.Wrong)
    if return_dict:
        return {
            "reward": final_reward,
            "reward_type": reward_type,
            "traj_len": res.get("traj_len", -1),
            "gt-traj_len": expected_steps,
            "parallelism": res.get("parallelism", -1),
        }
    else:
        return (final_reward, reward_type)


def compute_score_nothinking(
    solution_str,
    env_config,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    plan_str = extract_fn(solution_str)
    if plan_str == "":
        format_r = 0.0
    else:
        format_r = 1.0

    if isinstance(env_config, str):
        env_config = json.loads(env_config)

    env = Box1Env.load(env_config)
    res = env.simulate_all_str(plan_str, return_step_action=True)

    expected_steps = len(json.loads(env_config["gt_plan"]))
    if res["success"]:
        tmp_r = format_r * format_score + 1.0
        # we add penalty on the steps
        actual_steps = res["traj_len"]
        # we add penalty on the steps
        if actual_steps > expected_steps:
            tmp_r -= (actual_steps - expected_steps) * 0.1
            tmp_r = max(tmp_r, 2 * format_score)  # at least bigger than only format
        final_reward = tmp_r
        reward_type = RewardType.Correct
    else:
        final_reward = format_score * format_r + 0.0
        reward_type = reason_reward_mapping.get(res["detail"], RewardType.Wrong)

    if return_dict:
        return {
            "reward": final_reward,
            "reward_type": reward_type,
            "traj_len": res.get("traj_len", -1),
            "gt-traj_len": expected_steps,
            "parallelism": res.get("parallelism", -1),
        }
    else:
        return (final_reward, reward_type)


def compute_score_one_step(
    solution_str,
    env: Box1Env,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    format_r = format_reward(solution_str)
    plan_str = extract_fn(solution_str)
    start = time.time()
    res = env.simulate_one_step_from_str(plan_str)

    # TODO: update the reward type finding
    if res["success"]:
        if res["detail"] == "Success":
            success = 1
            final_reward = format_r * format_score
            reward_type = RewardType.Correct
            # return (1, format_r * format_score, RewardType.Correct)
        else:
            success = 0
            final_reward = format_r * format_score
            reward_type = get_reward_type(res["detail"])
            # reward_type = reason_reward_mapping.get(res["detail"], RewardType.Wrong)
            # return (0, format_r * format_score, reason_reward_mapping[res["detail"]])
    else:
        # if res["detail"] not in reason_reward_mapping:
        success = -1
        final_reward = format_r * format_score
        reward_type = get_reward_type(res["detail"])
        # reward_type = RewardType.Wrong
        # return (-1, format_r * format_score, RewardType.Wrong)
        # elif res["detail"] in reason_reward_mapping:
        # success = -1
        # final_reward = format_r * format_score
        # reward_type = reason_reward_mapping[res["detail"]]
        # return (-1, format_r * format_score, reason_reward_mapping[res["detail"]])
    if return_dict:
        end = time.time()
        return {
            "success": success,
            "reward": final_reward,
            "reward_type": reward_type,
            "timing": end - start,
        }
    else:
        end = time.time()
        # print("Time taken:", start, end, " ||| ", end - start)
        # return (success, final_reward, reward_type, end - start)
        return (success, final_reward, reward_type)


# @ray.remote(num_cpus=multiprocessing.cpu_count() // 2)
@ray.remote(num_cpus=1)
def compute_score_one_step_ray(
    solution_str,
    env: Box1Env,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
    envcls=Box1Env,
):
    if isinstance(env, str):
        env = json.loads(env)
    realenv = envcls.load(env)
    return (
        compute_score_one_step(
            solution_str, realenv, format_score, extract_fn, return_dict
        ),
        realenv.get_current_state(),
    )  # need to return env state, otherwise the original env will not be changed


def compute_score_3d(
    solution_str,
    env_config,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    format_r = format_reward(solution_str)
    plan_str = extract_fn(solution_str)
    if isinstance(env_config, str):
        env_config = json.loads(env_config)

    env = Box3DEnv.load(env_config)
    res = env.simulate_full_step(plan_str)

    if res["success"]:
        final_reward = format_r * format_score + 1.0
        reward_type = RewardType.Correct
    else:
        final_reward = format_score * format_r + 0.0
        reward_type = reason_reward_mapping.get(res["detail"], RewardType.Wrong)

    if return_dict:
        return {
            "reward": final_reward,
            "reward_type": reward_type,
            "traj_len": res.get("traj_len", -1),
            "parallelism": res.get("parallelism", -1),
        }
    else:
        return (final_reward, reward_type)


import time


def compute_score_with_step_penalty_3d(
    solution_str,
    env_config,
    format_score=0.1,
    extract_fn=extract_last_json,
    return_dict=False,
):
    format_r = format_reward(solution_str)
    plan_str = extract_fn(solution_str)
    if isinstance(env_config, str):
        env_config = json.loads(env_config)

    start = time.time()
    env = Box3DEnv.load(env_config)
    res = env.simulate_all_str(plan_str)

    print("Format reward:", format_r)
    expected_steps = len(json.loads(env_config["gt_plan"]))
    if res["success"]:
        tmp_r = format_r * format_score + 1.0  # full reward
        # we add penalty on the steps
        actual_steps = res["traj_len"]
        # we add penalty on the steps
        if actual_steps > expected_steps:
            tmp_r -= (actual_steps - expected_steps) * 0.1
            tmp_r = max(tmp_r, 2 * format_score)  # at least bigger than only format
        final_reward = tmp_r
        reward_type = RewardType.Correct
    else:
        final_reward = format_score * format_r + 0.0
        reward_type = get_reward_type(res["detail"])
    if return_dict:
        end = time.time()
        return {
            "reward": final_reward,
            "reward_type": reward_type,
            "traj_len": res.get("traj_len", -1),
            "gt-traj_len": expected_steps,
            "parallelism": res.get("parallelism", -1),
            "detail": res["detail"],
            "timing": end - start,
        }
    else:
        return (final_reward, reward_type)


def get_reward_type(detail):
    for key, value in reason_reward_mapping.items():
        if detail.lower().startswith(key.lower()):
            return value
    else:
        return RewardType.Wrong
