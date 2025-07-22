import os
import copy
import math
import itertools
import time
import json
import heapq
import random
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)  # ProcessPoolExecutor removed

import ray  # <-- NEW: Import Ray
import yaml

from gen_data import generate_configs
from box1_llm import object_to_json_serializable
from inference.utils import seed_everything
from simulation.simenv.box1 import Box1Env, Action, ExecutionRes


@contextmanager
def _timer(label="Elapsed time"):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        # print(f"{label}: {end - start:.4f} seconds")


def count_unique_envs(grid_n: int, grid_m: int, num_objects: int):
    sub_offsets = [0.25, 0.75]
    valid_positions = [
        (i + dx, j + dy)
        for i in range(grid_n)
        for j in range(grid_m)
        for dx in sub_offsets
        for dy in sub_offsets
    ]
    total_pos = len(valid_positions)

    # We need 2 * num_objects unique positions for (start, target), disjoint
    if 2 * num_objects > total_pos:
        return 0

    object_combos = math.comb(total_pos, num_objects)
    target_combos = math.comb(total_pos - num_objects, num_objects)
    total_unique = object_combos * target_combos
    return total_unique


def process_state_batch_input(args):
    current_hash, current_state, g_score, randomness_factor = args

    if current_state.is_goal():
        return "goal", current_hash, None, None

    successors = []
    action_sets = generate_valid_actions(current_state)

    for actions, next_state in action_sets:
        next_hash = next_state.hash()

        action_cost = (
            1.0 + (random.random() - 0.5) * randomness_factor
            if randomness_factor > 0
            else 1.0
        )
        tentative_g = g_score + action_cost
        heuristic = next_state.heuristic()
        f_score = tentative_g + heuristic * (1.0 + random.random() * randomness_factor)

        successors.append(
            (f_score, heuristic, next_hash, tentative_g, actions, next_state)
        )

    return "continue", current_hash, successors, None


def plan_actions(
    env: "Box1Env",
    max_iterations: int = 10000,
    randomness_factor: float = 0.15,
    max_plan_number: int = 3,
) -> List[List["Action"]]:
    # print(f"Starting search algorithm with randomness factor: {randomness_factor}...")

    open_set = []
    closed_set = set()
    found_plans = []
    plan_hashes = set()

    initial_state = State(env)
    initial_hash = initial_state.hash()
    g_scores = {initial_hash: 0.0}
    came_from = {}

    # Add random tie-breaker to f_score
    random_factor = random.random() * randomness_factor
    heapq.heappush(
        open_set,
        (
            initial_state.heuristic() * (1.0 + random_factor),
            random.random(),
            initial_hash,
        ),
    )
    states = {initial_hash: initial_state}
    last_time = time.time()
    iterations = 0

    while (
        open_set and iterations < max_iterations and len(found_plans) < max_plan_number
    ):
        iterations += 1

        if iterations % 2 == 0:
            last_time = time.time()

        _, _, current_hash = heapq.heappop(open_set)
        if current_hash in closed_set:
            continue

        current_state = states[current_hash]

        if current_state.is_goal():
            # print(f"Solution found after {iterations} iterations!")
            path = reconstruct_path(came_from, current_hash, states)
            path_hash = str(path)

            if path_hash not in plan_hashes:
                # print(
                #     f"Unique solution #{len(found_plans) + 1} found after {iterations} iterations!"
                # )
                found_plans.append(path)
                plan_hashes.add(path_hash)

            closed_set.add(current_hash)
            continue

        closed_set.add(current_hash)

        with _timer("Generate valid actions"):
            possible_action_sets = generate_valid_actions(current_state)

        if not possible_action_sets:
            # print(f"No valid actions found for state at iteration {iterations}")
            continue

        for action_state_pair in possible_action_sets:
            actions = action_state_pair[0]
            next_state = action_state_pair[1]
            next_hash = next_state.hash()

            if next_hash in closed_set:
                continue

            action_cost = 1.0
            if random.random() < randomness_factor:
                action_cost = 1.0 + (random.random() - 0.5) * randomness_factor
            tentative_g_score = g_scores[current_hash] + action_cost

            if next_hash not in g_scores or tentative_g_score < g_scores[next_hash]:
                came_from[next_hash] = (current_hash, actions)
                g_scores[next_hash] = tentative_g_score

                heuristic = next_state.heuristic()
                random_factor = random.random() * randomness_factor
                f_score = tentative_g_score + heuristic * (1.0 + random_factor)

                if next_hash not in states:
                    states[next_hash] = next_state
                    heapq.heappush(
                        open_set, (f_score, next_state.heuristic(), next_hash)
                    )

    if found_plans:
        # print("Total found plans: ", len(found_plans))
        pass
    else:
        print(f"Failed to find solution after {iterations} iterations")
    return found_plans


def reconstruct_path(came_from, final_hash, states):
    actions_sequence = []
    current_hash = final_hash

    while current_hash in came_from:
        parent_hash, actions = came_from[current_hash]
        actions_sequence.append(actions)
        current_hash = parent_hash

    actions_sequence.reverse()
    return actions_sequence


def generate_valid_action_single_robot(robot_id, robot_pos, env):
    robot_actions = []
    touch_obj = []
    occured_actions = {}

    carrying_object = False
    carried_obj_id = None
    for obj_id, obj_pos in env.objects.items():
        if list(robot_pos.arm_pos) == list(obj_pos):
            carrying_object = True
            carried_obj_id = obj_id
            break

    if carrying_object:
        target_pos = env.targets[carried_obj_id]
        if list(obj_pos) != list(target_pos):
            if can_robot_reach(robot_pos.base_pos, target_pos):
                action = create_action(robot_id, robot_pos, obj_pos, target_pos, True)
                if verify_action(action, env):
                    if action.hash() not in occured_actions:
                        robot_actions.append(action)
                        touch_obj.append(True)
                        occured_actions[action.hash()] = True

            cur_distance = abs(obj_pos[0] - target_pos[0]) + abs(
                obj_pos[1] - target_pos[1]
            )
            for step_pos in get_step_positions(robot_pos, obj_pos, target_pos, env):
                new_distance = abs(step_pos[0] - target_pos[0]) + abs(
                    step_pos[1] - target_pos[1]
                )
                if (
                    can_robot_reach(robot_pos.base_pos, step_pos)
                    and new_distance < cur_distance
                ):
                    action = create_action(robot_id, robot_pos, obj_pos, step_pos, True)
                    if verify_action(action, env):
                        if action.hash() not in occured_actions:
                            robot_actions.append(action)
                            touch_obj.append(True)
                            occured_actions[action.hash()] = True

    for obj_id, obj_pos in env.objects.items():
        if list(obj_pos) == list(env.targets[obj_id]) or obj_id == carried_obj_id:
            continue

        if can_robot_reach(robot_pos.base_pos, obj_pos):
            action = create_action(
                robot_id, robot_pos, robot_pos.arm_pos, obj_pos, False
            )
            if verify_action(action, env):
                if action.hash() not in occured_actions:
                    robot_actions.append(action)
                    touch_obj.append(True)
                    occured_actions[action.hash()] = True

    robot_actions = [(x, y) for x, y in zip(robot_actions, touch_obj)]
    return robot_id, robot_actions


# --- MODIFIED: Use Ray to run this function in parallel ---
@ray.remote
def check_and_apply_actions(args):
    combined_actions, env, state = args
    if verify_parallel_actions(combined_actions, env):
        return (combined_actions, state.apply_actions(combined_actions))
    return (False,)


def makes_progress(action_tuple, state):
    action, touches_object = action_tuple
    env = state.env

    if not touches_object:
        robot_pos = env.robots[action.robot_id].arm_pos
        target_pos = action.pos_e

        current_min_dist = float("inf")
        new_min_dist = float("inf")

        for obj_id, obj_pos in env.objects.items():
            if list(obj_pos) == list(env.targets[obj_id]):
                continue

            current_dist = abs(robot_pos[0] - obj_pos[0]) + abs(
                robot_pos[1] - obj_pos[1]
            )
            new_dist = abs(target_pos[0] - obj_pos[0]) + abs(target_pos[1] - obj_pos[1])

            current_min_dist = min(current_min_dist, current_dist)
            new_min_dist = min(new_min_dist, new_dist)

        return new_min_dist < current_min_dist

    if action.carry:
        carried_obj_id = None
        for obj_id, obj_pos in env.objects.items():
            if list(obj_pos) == list(action.pos_s):
                carried_obj_id = obj_id
                break

        if carried_obj_id is not None:
            target_pos = env.targets[carried_obj_id]
            current_dist = abs(action.pos_s[0] - target_pos[0]) + abs(
                action.pos_s[1] - target_pos[1]
            )
            new_dist = abs(action.pos_e[0] - target_pos[0]) + abs(
                action.pos_e[1] - target_pos[1]
            )
            return new_dist < current_dist

    return False


def generate_valid_actions(state: "State") -> List[List["Action"]]:
    env = state.env
    single_robot_actions = {}
    valid_action_sets = []

    with _timer("Single robot actions"):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    generate_valid_action_single_robot, robot_id, robot_pos, env
                )
                for robot_id, robot_pos in env.robots.items()
            ]
            for future in as_completed(futures):
                robot_id, actions = future.result()
                if actions:
                    single_robot_actions[robot_id] = actions

    for robot_id, actions in single_robot_actions.items():
        for action in actions:
            if action[1]:
                valid_action_sets.append([action[0], state.apply_actions([action[0]])])

    with _timer("Combination of robot actions"):
        if len(single_robot_actions) >= 2:
            robot_ids = list(single_robot_actions.keys())
            with _timer("Generate task"):
                tasks = []
                for group_size in range(2, min(4, len(state.env.objects) + 1)):
                    for robot_group in itertools.combinations(robot_ids, group_size):
                        combos = list(
                            itertools.product(
                                *[
                                    single_robot_actions[robot_id]
                                    for robot_id in robot_group
                                ]
                            )
                        )
                        combos = [[a[0] for a in combo] for combo in combos]
                        combos = [
                            (combo, env, state)
                            for combo in combos
                            if verify_parallel_actions(combo, env)
                        ]
                        tasks.extend(combos)
            # --- MODIFIED: Replace ProcessPoolExecutor with Ray remote tasks ---
            with _timer("Check and apply actions"):
                ray_tasks = [check_and_apply_actions.remote(task) for task in tasks]
                results = ray.get(ray_tasks)
                for result in results:
                    if result and result[0]:
                        valid_action_sets.append(result)

    valid_action_sets = list(sorted(valid_action_sets, key=lambda x: x[1].heuristic()))
    valid_action_sets = valid_action_sets[:100]
    return valid_action_sets


def verify_parallel_actions(actions, env):
    test_env = copy.deepcopy(env)
    result = test_env.verify(actions)
    return result.success == ExecutionRes.Success


def verify_action(action, env):
    test_env = copy.deepcopy(env)
    result = test_env.verify([action])
    return result.success == ExecutionRes.Success


def is_position_occupied(position, env):
    for obj_id, obj_pos in env.objects.items():
        if tuple(obj_pos) == position:
            return True
    return False


def create_action(robot_id, robot_pos, start_pos, end_pos, carry):
    from_pos = tuple(start_pos) if isinstance(start_pos, list) else start_pos
    to_pos = tuple(end_pos) if isinstance(end_pos, list) else end_pos
    base_pos = (
        list(robot_pos.base_pos)
        if isinstance(robot_pos.base_pos, list)
        else robot_pos.base_pos
    )
    arm_pos = (
        list(robot_pos.arm_pos)
        if isinstance(robot_pos.arm_pos, list)
        else robot_pos.arm_pos
    )
    return Action(
        robot_id=robot_id,
        arm_pos=list(arm_pos),
        base_pos=list(base_pos),
        pos_s=list(from_pos),
        pos_e=list(to_pos),
        carry=carry,
    )


def can_robot_reach(base_pos, target_pos):
    x_diff = abs(base_pos[0] - target_pos[0])
    y_diff = abs(base_pos[1] - target_pos[1])
    return x_diff < 1.0 and y_diff < 1.0


def get_robot_can_reach_positions(base_pos, env):
    offsets = [-0.25, -0.75, 0.25, 0.75]
    res = [(base_pos[0] + dx, base_pos[1] + dy) for dx in offsets for dy in offsets]
    res = [x for x in res if 0 < x[0] < env.grid_n and 0 < x[1] < env.grid_m]
    res = [x for x in res if x[0] != base_pos[0] or x[1] != base_pos[1]]
    return res


def get_step_positions(robot_pos, start_pos, end_pos, env):
    positions = get_robot_can_reach_positions(robot_pos.base_pos, env)
    positions = [list(x) for x in positions]
    positions = [x for x in positions if not (x[0] % 0.5 == 0 or x[1] % 0.5 == 0)]
    positions = sorted(
        positions,
        key=lambda x: abs(x[0] - end_pos[0]) + abs(x[1] - end_pos[1]),
    )
    return positions


class State:
    def __init__(self, env):
        self.env = copy.deepcopy(env)

    def __str__(self):
        res = "State:(\n"
        for obj_id, obj_pos in self.env.objects.items():
            res += f"\tObject {obj_id}: {obj_pos}\tTarget: {self.env.targets[obj_id]}\n"
        res += "\n"
        for robot_id, robot_pos in self.env.robots.items():
            res += f"\tRobot {robot_id}: {robot_pos.base_pos} -> {robot_pos.arm_pos}\n"
        res += ")\n"
        return res

    def apply_actions(self, actions):
        new_state = State(self.env)
        new_state.env.simulate(actions)
        return new_state

    def is_goal(self):
        return self.env.check_final()

    def hash(self):
        h = 0
        for obj_id in sorted(self.env.objects.keys()):
            obj_pos = self.env.objects[obj_id]
            h = h * 31 + hash(obj_id)
            h = h * 31 + hash((int(obj_pos[0] * 100), int(obj_pos[1] * 100)))

        for r_id in sorted(self.env.robots.keys()):
            pos = self.env.robots[r_id]
            h = h * 31 + hash(r_id)
            h = h * 31 + hash((int(pos.arm_pos[0] * 100), int(pos.arm_pos[1] * 100)))

        return h

    def heuristic(self):
        total_distance = 0.0

        for obj_id, obj_pos in self.env.objects.items():
            target_pos = self.env.targets[obj_id]
            obj_distance = abs(obj_pos[0] - target_pos[0]) + abs(
                obj_pos[1] - target_pos[1]
            )
            total_distance += obj_distance * 10

            covered = False
            cur_cover_robot_distance = float("inf")
            for robot_id, robot_pos in self.env.robots.items():
                if list(obj_pos) == list(robot_pos.arm_pos):
                    covered = True
                    this_robot_distance = abs(
                        robot_pos.base_pos[0] - target_pos[0]
                    ) + abs(robot_pos.base_pos[1] - target_pos[1])
                    if this_robot_distance < cur_cover_robot_distance:
                        cur_cover_robot_distance = this_robot_distance

            better_robot_id = None
            better_robot_distance = float("inf")
            for robot_id, robot_pos in self.env.robots.items():
                if can_robot_reach(robot_pos.base_pos, obj_pos) and not (
                    list(obj_pos) == list(robot_pos.arm_pos)
                ):
                    cost = abs(robot_pos.base_pos[0] - target_pos[0]) + abs(
                        robot_pos.base_pos[1] - target_pos[1]
                    )
                    cost += abs(robot_pos.arm_pos[0] - obj_pos[0]) + abs(
                        robot_pos.arm_pos[1] - obj_pos[1]
                    )
                    if cost < better_robot_distance:
                        better_robot_distance = cost
                        better_robot_id = robot_id

            if better_robot_id is not None:
                cur_cover_robot_distance = max(
                    cur_cover_robot_distance, better_robot_distance
                )

            if not covered:
                total_distance += obj_distance * 10
            else:
                total_distance += cur_cover_robot_distance * 5

        return total_distance


def solve_box_environment(
    outfile=None, env_path=None, randomness_factor=0.15, env_config=None
):
    if env_path:
        env = Box1Env.load_json(env_path)
    else:
        if env_config is not None:
            env = Box1Env(**env_config)
        else:
            env = Box1Env(grid_n=2, grid_m=4, num_objects=3)
        env.create()

    action_plans = plan_actions(
        env, max_iterations=200, randomness_factor=randomness_factor, max_plan_number=1
    )

    if not action_plans:
        # print("No solution found!")
        return False

    for action_plan in action_plans:
        action_plan = remove_redundant_move(action_plan)
        action_plan = squeeze_move(env, action_plan)

        test_env = copy.deepcopy(env)
        for i, step_actions in enumerate(action_plan):
            if not isinstance(step_actions, list):
                step_actions = [step_actions]
            exec_res = env.verify(step_actions)
            test_env.simulate(step_actions)
        success = test_env.check_final()
        if success:
            out_obj = {
                "env": env.to_json(),
                "plan": object_to_json_serializable(action_plan),
            }
            if outfile is not None:
                outfile.write(json.dumps(out_obj) + "\n")
    return success


def remove_redundant_move(actions):
    new_actions = []
    used = set()
    for step_action in reversed(actions):
        new_step_action = []
        if not isinstance(step_action, list):
            step_action = [step_action]
        for action in step_action:
            if action.carry:
                used.add(action.robot_id)
            if action.robot_id in used:
                new_step_action.append(action)
        if new_step_action:
            new_actions.append(new_step_action)
    return list(reversed(new_actions))


def squeeze_move(env, actions):
    env = copy.deepcopy(env)
    new_actions = []
    i = 0
    cur_step_actions = []
    while i < len(actions):
        if not isinstance(actions[i], list):
            step_action = [actions[i]]
        else:
            step_action = actions[i]
        tmp_step_actions = cur_step_actions + step_action
        if env.verify(tmp_step_actions).success == ExecutionRes.Success:
            cur_step_actions = tmp_step_actions
        else:
            new_actions.append(cur_step_actions)
            env.simulate(cur_step_actions)
            cur_step_actions = step_action
        i += 1
    if cur_step_actions:
        new_actions.append(cur_step_actions)
    return new_actions


env_config_str = """
name: "box1"
grid_n: [2, 2, 2, 2, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
grid_m: [2, 4, 2, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4]
num_objects: [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3]
robot_mode: "full"
robot_as: "full"
movement: "full"
"""

train_env_config_str = """
name: "box1"
grid_n: [ 3, 4]
grid_m: [ 3, 4]
num_objects: [ 2, 2]
robot_mode: "full"
robot_as: "full"
movement: "full"
"""

large_train_env_config_str = """
name: "box1"
robot_mode: "full"
robot_as: "full"
movement: "full"
"""


def get_cache_dir():
    import socket
    import getpass

    hostname = socket.gethostname()  # check machine
    username = getpass.getuser()
    print("Hostname: ", hostname)
    if "bluevela" in hostname:
        if "kqian" in username:
            # return "/proj/long-multi/kqian/granite-3b-ablate/.ray_cache"
            return "/proj/long-multi/kqian/granite-3b-ablate/.r"
        elif "becon" in username:
            return "/proj/long-multi/midi/.ray_cache"
    elif "katrina1" in hostname:  # we're in katrina
        return "/mnt/data3/jiabao/.ray_cache/"
    else:
        raise NotImplementedError


train_env_config_str = """
name: "box1"
# grid_n: [2, 3, 4, 2, 3, 4]
# grid_m: [2, 3, 4, 2, 3, 4]
# num_objects: [1, 1, 1, 2, 2, 2]
grid_n: [2, 3, 4]
grid_m: [2, 3, 4]
num_objects: [2, 2, 2]
robot_mode: "full"
robot_as: "full"
movement: "full"
"""


large_train_env_config_str = """
name: "box1"
# grid_n: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# grid_m: [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
# num_objects: [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]

# grid_n: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# grid_m: [3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
# num_objects: [6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]

grid_n: [2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]
grid_m: [6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
num_objects: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

robot_mode: "full"
robot_as: "full"
movement: "full"
"""


def main(arg_configs):
    # Initialize Ray if not already started
    if not ray.is_initialized():
        ray.init(_temp_dir=get_cache_dir())  # Adjust based on your machine

    # config_tuple = [(6, 4, 3), (6, 5, 3), (6, 6, 3)]
    # config_tuple = [
    #     (4, 4, 4),
    #     (4, 5, 4),
    #     (4, 6, 4),
    #     (5, 2, 4),
    #     (5, 3, 4),
    #     (5, 4, 4),
    #     (5, 5, 4),
    #     (5, 6, 4),
    #     (6, 2, 4),
    #     (6, 3, 4),
    #     (6, 4, 4),
    #     (6, 5, 4),
    #     (6, 6, 4),
    # ]
    # config_tuple = [
    #     (3, 6, 5),
    #     (4, 2, 5),
    #     (4, 3, 5),
    #     (4, 4, 5),
    #     (4, 5, 5),
    #     (4, 6, 5),
    #     (5, 2, 5),
    #     (5, 3, 5),
    #     (5, 4, 5),
    #     (5, 5, 5),
    #     (5, 6, 5),
    #     (6, 2, 5),
    #     (6, 3, 5),
    #     (6, 4, 5),
    #     (6, 5, 5),
    #     (6, 6, 5),
    # ]

    # config_tuple = [
    #     (2, 6, 6),
    #     (3, 2, 6),
    #     (3, 3, 6),
    #     (3, 4, 6),
    #     (3, 5, 6),
    #     (3, 6, 6),
    #     (4, 2, 6),
    #     (4, 3, 6),
    #     (4, 4, 6),
    #     (4, 5, 6),
    #     (4, 6, 6),
    #     (5, 2, 6),
    #     (5, 3, 6),
    #     (5, 4, 6),
    #     (5, 5, 6),
    #     (5, 6, 6),
    #     (6, 2, 6),
    #     (6, 3, 6),
    #     (6, 4, 6),
    #     (6, 5, 6),
    #     (6, 6, 6),
    # ]
    config_tuple = [
        (2, 2, 1),
        (2, 3, 1),
        (2, 4, 1),
        (2, 5, 1),
        (2, 6, 1),
        (3, 2, 1),
        (3, 3, 1),
        (3, 4, 1),
        (3, 5, 1),
        (3, 6, 1),
        (4, 2, 1),
        (4, 3, 1),
        (4, 4, 1),
        (4, 5, 1),
        (4, 6, 1),
        (5, 2, 1),
        (5, 3, 1),
        (5, 4, 1),
        (5, 5, 1),
        (5, 6, 1),
        (6, 2, 1),
        (6, 3, 1),
        (6, 4, 1),
        (6, 5, 1),
        (6, 6, 1),
        (2, 2, 2),
        (2, 3, 2),
        (2, 4, 2),
        (2, 5, 2),
        (2, 6, 2),
        (3, 2, 2),
        (3, 3, 2),
        (3, 4, 2),
        (3, 5, 2),
        (3, 6, 2),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 2),
        (4, 5, 2),
        (4, 6, 2),
        (5, 2, 2),
        (5, 3, 2),
        (5, 4, 2),
        (5, 5, 2),
        (5, 6, 2),
        (6, 2, 2),
        (6, 3, 2),
        (6, 4, 2),
        (6, 5, 2),
        (6, 6, 2),
        (2, 2, 3),
        (2, 3, 3),
        (2, 4, 3),
        (2, 5, 3),
        (2, 6, 3),
        (3, 2, 3),
        (3, 3, 3),
        (3, 4, 3),
        (3, 5, 3),
        (3, 6, 3),
        (4, 2, 3),
        (4, 3, 3),
        (4, 4, 3),
        (4, 5, 3),
        (4, 6, 3),
        (5, 2, 3),
        (5, 3, 3),
        (5, 4, 3),
        (5, 5, 3),
        (5, 6, 3),
        (6, 2, 3),
        (6, 3, 3),
        (6, 4, 3),
        (6, 5, 3),
        (6, 6, 3),
        (2, 2, 4),
        (2, 3, 4),
        (2, 4, 4),
        (2, 5, 4),
        (2, 6, 4),
        (3, 2, 4),
        (3, 3, 4),
        (3, 4, 4),
        (3, 5, 4),
        (3, 6, 4),
        (4, 2, 4),
        (4, 3, 4),
        (4, 4, 4),
        (4, 5, 4),
        (4, 6, 4),
        (5, 2, 4),
        (5, 3, 4),
        (5, 4, 4),
        (5, 5, 4),
        (5, 6, 4),
        (6, 2, 4),
        (6, 3, 4),
        (6, 4, 4),
        (6, 5, 4),
        (6, 6, 4),
        (2, 2, 5),
        (2, 3, 5),
        (2, 4, 5),
        (2, 5, 5),
        (2, 6, 5),
        (3, 2, 5),
        (3, 3, 5),
        (3, 4, 5),
        (3, 5, 5),
        (3, 6, 5),
        (4, 2, 5),
        (4, 3, 5),
        (4, 4, 5),
        (4, 5, 5),
        (4, 6, 5),
        (5, 2, 5),
        (5, 3, 5),
        (5, 4, 5),
        (5, 5, 5),
        (5, 6, 5),
        (6, 2, 5),
        (6, 3, 5),
        (6, 4, 5),
        (6, 5, 5),
        (6, 6, 5),
        (2, 2, 6),
        (2, 3, 6),
        (2, 4, 6),
        (2, 5, 6),
        (2, 6, 6),
        (3, 2, 6),
        (3, 3, 6),
        (3, 4, 6),
        (3, 5, 6),
        (3, 6, 6),
        (4, 2, 6),
        (4, 3, 6),
        (4, 4, 6),
        (4, 5, 6),
        (4, 6, 6),
        (5, 2, 6),
        (5, 3, 6),
        (5, 4, 6),
        (5, 5, 6),
        (5, 6, 6),
        (6, 2, 6),
        (6, 3, 6),
        (6, 4, 6),
        (6, 5, 6),
        (6, 6, 6),
    ]
    config_tuple = [x for x in config_tuple if x[0] == x[1]]
    tmp_config = yaml.safe_load(large_train_env_config_str)
    tmp_config["grid_n"] = [x[0] for x in config_tuple]
    tmp_config["grid_m"] = [x[1] for x in config_tuple]
    tmp_config["num_objects"] = [x[2] for x in config_tuple]
    split_idx = 3

    env_configs = generate_configs(tmp_config)
    split_idx = arg_configs.numobj
    out_file_path = f"new-train2-large-full-split5-obj@{split_idx}.jsonl"

    # env_configs = env_configs[52:]

    with open(out_file_path, "w") as f:
        for env_config in env_configs:
            if env_config["num_objects"] < split_idx:
                continue
            if env_config["num_objects"] > split_idx:
                break

            seen_envs: Set[str] = set()
            num_generated = 0
            attempts = 0

            total_unique_envs_per_config = min(
                220,
                count_unique_envs(
                    grid_n=env_config["grid_n"],
                    grid_m=env_config["grid_m"],
                    num_objects=env_config["num_objects"],
                ),
            )
            pbar_desc = f"N={env_config['grid_n']}, M={env_config['grid_m']}, Obj={env_config['num_objects']}"
            not_found = 100
            with tqdm(total=total_unique_envs_per_config, desc=pbar_desc) as pbar:
                while num_generated < total_unique_envs_per_config:
                    while num_generated < total_unique_envs_per_config:
                        attempts += 1
                        seed_everything(42 + attempts)

                        # Create environment
                        env = Box1Env(**env_config)
                        env.create()
                        env_json = json.dumps(env.to_json(), sort_keys=True)

                        if env_json in seen_envs:
                            continue  # Skip if already generated
                        seen_envs.add(env_json)

                        # Solve and save
                        ok = solve_box_environment(
                            outfile=f, env_config=env_config, randomness_factor=0.0
                        )
                        if ok:
                            num_generated += 1
                            pbar.update(1)
                            not_found = 100
                        else:
                            not_found -= 1
                            print(f"Failed to solve environment: {env_config}")
                        if not_found < 0:
                            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--numobj", type=int)
    configs = parser.parse_args()
    main(configs)
