# This script performs A* search to solve Box environment.
# Generated through collaboration with Claude-3.7-sonnet

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
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

from gen_data import generate_configs
from box1_llm import object_to_json_serializable
import yaml
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

    # Early check
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
    """
    Implements an A* search algorithm with randomness to find a sequence of actions that moves
    all objects to their target positions in the Box1Env.

    Args:
        env: The Box1Env instance
        max_iterations: Maximum number of search iterations before giving up
        randomness_factor: Factor to determine how much randomness to introduce (0.0-1.0)

    Returns:
        List of action sequences, where each sequence is a list of parallel robot actions
    """
    print(f"Starting search algorithm with randomness factor: {randomness_factor}...")

    # Priority queue for A* search
    open_set = []
    # Track visited states to avoid cycles
    closed_set = set()

    found_plans = []
    plan_hashes = set()

    # Initial state
    initial_state = State(env)
    initial_hash = initial_state.hash()

    # Track g_scores (cost from start to node)
    g_scores = {initial_hash: 0.0}

    # Track parent states and actions that led to them
    came_from = {}

    # Initialize search with the starting state
    # Add random tie-breaker to f_score to introduce exploration variability
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
            # print(
            #     f"Iteration {iterations}, queue size: {len(open_set)}, time: {time.time() - last_time}"
            # )
            last_time = time.time()

        # Get the state with lowest f_score
        _, _, current_hash = heapq.heappop(open_set)

        # Skip if we've already processed this state
        if current_hash in closed_set:
            continue

        current_state = states[current_hash]

        # Check if we've reached the goal
        if current_state.is_goal():
            print(f"Solution found after {iterations} iterations!")
            path = reconstruct_path(came_from, current_hash, states)
            path_hash = str(path)  # Or use a better canonical representation

            if path_hash not in plan_hashes:
                print(
                    f"Unique solution #{len(found_plans) + 1} found after {iterations} iterations!"
                )
                found_plans.append(path)
                plan_hashes.add(path_hash)

            closed_set.add(current_hash)
            continue

        # Mark as visited
        closed_set.add(current_hash)

        # Generate possible actions from the current state
        with _timer("Generate valid actions"):
            possible_action_sets = generate_valid_actions(current_state)

        if not possible_action_sets:
            print(f"No valid actions found for state at iteration {iterations}")
            continue

        # Explore each successor state
        for action_state_pair in possible_action_sets:
            # Create next state by applying actions
            actions = action_state_pair[0]
            next_state = action_state_pair[1]
            next_hash = next_state.hash()

            # Skip if we've already visited this state
            if next_hash in closed_set:
                continue

            # Calculate tentative g_score (cost of one step, with slight randomness)
            action_cost = 1.0
            if random.random() < randomness_factor:
                # Occasionally vary the cost slightly to explore different paths
                action_cost = 1.0 + (random.random() - 0.5) * randomness_factor

            tentative_g_score = g_scores[current_hash] + action_cost

            if next_hash not in g_scores or tentative_g_score < g_scores[next_hash]:
                # This path is better than any previous one
                came_from[next_hash] = (current_hash, actions)
                g_scores[next_hash] = tentative_g_score

                # Calculate f_score with some randomness
                heuristic = next_state.heuristic()
                random_factor = random.random() * randomness_factor
                f_score = tentative_g_score + heuristic * (1.0 + random_factor)

                # Add to open set with random tie-breaker
                if next_hash not in states:
                    states[next_hash] = next_state
                    heapq.heappush(
                        open_set, (f_score, next_state.heuristic(), next_hash)
                    )

    if found_plans:
        print("Total found plans: ", len(found_plans))
    else:
        # If we exit the loop without finding a solution
        print(f"Failed to find solution after {iterations} iterations")
    return found_plans


def reconstruct_path(came_from, final_hash, states):
    """
    Reconstruct the path from start to goal from the came_from map.

    Args:
        came_from: Dictionary mapping state hash to (parent_hash, actions)
        final_hash: Hash of the goal state
        states: Dictionary mapping hashes to state objects

    Returns:
        List of action sequences
    """
    actions_sequence = []
    current_hash = final_hash

    while current_hash in came_from:
        parent_hash, actions = came_from[current_hash]
        actions_sequence.append(actions)
        current_hash = parent_hash

    # Reverse to get actions from start to goal
    actions_sequence.reverse()
    return actions_sequence


def generate_valid_action_single_robot(robot_id, robot_pos, env):
    robot_actions = []
    touch_obj = []
    occured_actions = {}

    # Check if robot arm is already at an object position
    carrying_object = False
    carried_obj_id = None
    for obj_id, obj_pos in env.objects.items():
        if list(robot_pos.arm_pos) == list(obj_pos):
            carrying_object = True
            carried_obj_id = obj_id
            break

    if carrying_object:
        # Robot arm is at an object position - try to move the object toward its target
        target_pos = env.targets[carried_obj_id]
        # Skip if object is already at target
        if list(obj_pos) != list(target_pos):
            # Try direct move to target
            if can_robot_reach(robot_pos.base_pos, target_pos):
                action = create_action(robot_id, robot_pos, obj_pos, target_pos, True)
                if verify_action(action, env):
                    if action.hash() not in occured_actions:
                        robot_actions.append(action)
                        touch_obj.append(True)
                        occured_actions[action.hash()] = True

            # Try intermediate move if direct move not possible
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

    # Robot arm is not at an object position - try to move toward objects that need moving
    for obj_id, obj_pos in env.objects.items():
        # Skip objects already at target
        if list(obj_pos) == list(env.targets[obj_id]) or obj_id == carried_obj_id:
            continue

        # Check if robot can reach the object
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


def check_and_apply_actions(args):
    combined_actions, env, state = args
    if verify_parallel_actions(combined_actions, env):
        return (combined_actions, state.apply_actions(combined_actions))
    return (False,)


def makes_progress(action_tuple, state):
    """
    Determines if an action makes progress toward solving the task.
    Returns True if the action reduces distance between an object and its target.

    Args:
        action_tuple: Tuple of (Action, bool) where bool indicates if action touches object
        state: Current state object

    Returns:
        bool: True if action makes progress, False otherwise
    """
    action, touches_object = action_tuple
    env = state.env

    # If the action doesn't interact with an object, check if it moves toward one
    if not touches_object:
        # Check if robot is moving toward an object that's not at its target
        robot_pos = env.robots[action.robot_id].arm_pos
        target_pos = action.pos_e

        # Calculate if this movement brings robot closer to any object needing movement
        current_min_dist = float("inf")
        new_min_dist = float("inf")

        for obj_id, obj_pos in env.objects.items():
            # Skip objects already at their target
            if list(obj_pos) == list(env.targets[obj_id]):
                continue

            # Calculate current and new distances
            current_dist = abs(robot_pos[0] - obj_pos[0]) + abs(
                robot_pos[1] - obj_pos[1]
            )
            new_dist = abs(target_pos[0] - obj_pos[0]) + abs(target_pos[1] - obj_pos[1])

            current_min_dist = min(current_min_dist, current_dist)
            new_min_dist = min(new_min_dist, new_dist)

        # Progress is made if robot gets closer to any object needing help
        return new_min_dist < current_min_dist

    # If action carries an object, check if it reduces distance to target
    if action.carry:
        # Find which object is being carried (at pos_s)
        carried_obj_id = None
        for obj_id, obj_pos in env.objects.items():
            if list(obj_pos) == list(action.pos_s):
                carried_obj_id = obj_id
                break

        if carried_obj_id is not None:
            target_pos = env.targets[carried_obj_id]

            # Calculate if this movement brings object closer to its target
            current_dist = abs(action.pos_s[0] - target_pos[0]) + abs(
                action.pos_s[1] - target_pos[1]
            )
            new_dist = abs(action.pos_e[0] - target_pos[0]) + abs(
                action.pos_e[1] - target_pos[1]
            )

            # Progress is made if object gets closer to target
            return new_dist < current_dist

    # Default case - assume no progress
    return False


def generate_valid_actions(state: "State") -> List[List["Action"]]:
    """
    Generate valid actions from the current state, including multi-robot actions.

    Args:
        state: Current state

    Returns:
        List of action sets (each set is a list of Actions)
    """
    env = state.env
    single_robot_actions = {}  # Map robot_id -> list of possible actions
    valid_action_sets = []

    # First: generate possible actions for each robot individually
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
        # for robot_id, robot_pos in env.robots.items():
        #     robot_id, actions = generate_valid_action_single_robot(
        #         robot_id, robot_pos, env
        #     )
        #     if actions:
        #         single_robot_actions[robot_id] = actions

    # First try single robot actions that make progress
    # for robot_id, actions in single_robot_actions.items():
    #     progress_actions = [
    #         action for action in actions if makes_progress(action, state)
    # ]
    # if progress_actions:
    #     # If we have actions that make clear progress, prioritize those
    #     return [
    #         (action[0], state.apply_actions([action[0]]))
    #         for action in progress_actions[:10]
    # valid_action_sets.append(progress_actions)

    # # First, add single robot actions as they're more basic
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
                    # for group_size in range(2, 5):
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

                        # Final verification
                        combos = [
                            (combo, env, state)
                            for combo in combos
                            if verify_parallel_actions(combo, env)
                        ]
                        tasks.extend(combos)

            with _timer("Check and apply actions"):
                with ProcessPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(check_and_apply_actions, task) for task in tasks
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        if result and result[0]:
                            valid_action_sets.append(result)

    valid_action_sets = list(sorted(valid_action_sets, key=lambda x: x[1].heuristic()))
    valid_action_sets = valid_action_sets[:100]
    return valid_action_sets


def verify_parallel_actions(actions, env):
    """
    Verify that multiple actions can be executed in parallel.
    This is more complex than verifying single actions.

    Args:
        actions: List of Action objects
        env: Environment

    Returns:
        True if all actions can be executed in parallel, False otherwise
    """
    test_env = copy.deepcopy(env)
    result = test_env.verify(actions)
    return result.success == ExecutionRes.Success


def verify_action(action, env):
    """Verify that an action is valid using the environment's verify function"""
    from enum import Enum

    # Create a copy of the environment to verify
    test_env = copy.deepcopy(env)
    result = test_env.verify([action])

    return result.success == ExecutionRes.Success


def is_position_occupied(position, env):
    """Check if a position is occupied by an object"""
    for obj_id, obj_pos in env.objects.items():
        if tuple(obj_pos) == position:
            return True
    return False


def create_action(robot_id, robot_pos, start_pos, end_pos, carry):
    """Create an Action object"""
    # Convert to tuples if needed
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
    """Check if a robot can reach the target position from its base"""
    x_diff = abs(base_pos[0] - target_pos[0])
    y_diff = abs(base_pos[1] - target_pos[1])
    return x_diff < 1.0 and y_diff < 1.0


def get_robot_can_reach_positions(base_pos, env):
    # Right now we only consider 0.25, 0.75
    offsets = [-0.25, -0.75, 0.25, 0.75]
    res = [(base_pos[0] + dx, base_pos[1] + dy) for dx in offsets for dy in offsets]
    res = [x for x in res if 0 < x[0] < env.grid_n and 0 < x[1] < env.grid_m]
    res = [x for x in res if x[0] != base_pos[0] or x[1] != base_pos[1]]
    return res


def get_step_positions(robot_pos, start_pos, end_pos, env):
    """
    Get valid intermediate positions when moving from start to end.
    Returns positions in order of preference.
    """
    positions = []
    positions = get_robot_can_reach_positions(robot_pos.base_pos, env)
    positions = [list(x) for x in positions]
    positions = [x for x in positions if not (x[0] % 0.5 == 0 or x[1] % 0.5 == 0)]

    # sort by the manhattan distance to the target
    positions = sorted(
        positions,
        key=lambda x: abs(x[0] - end_pos[0]) + abs(x[1] - end_pos[1]),
    )
    return positions


class State:
    """
    Represents a state in the search space.
    Contains the environment configuration at a point in time.
    """

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
        """Apply a set of actions to create a new state"""
        new_state = State(self.env)
        new_state.env.simulate(actions)
        return new_state

    def is_goal(self):
        """Check if this state represents the goal (all objects at targets)"""
        return self.env.check_final()

    def hash(self):
        # Use a more direct hashing approach that avoids sorting and tuple conversion
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
        """
        Estimate the cost to reach the goal from this state.
        Uses the sum of Manhattan distances from objects to their targets.
        """
        total_distance = 0.0

        for obj_id, obj_pos in self.env.objects.items():
            target_pos = self.env.targets[obj_id]
            # Manhattan distance
            obj_distance = abs(obj_pos[0] - target_pos[0]) + abs(
                obj_pos[1] - target_pos[1]
            )
            total_distance += obj_distance * 10  # Weight distance more heavily

            # If there are some objects not covered by one robot
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

            # If there is a potential better robot to cover this object but not covering, we add penalty
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


# Example usage function
def solve_box_environment(
    outfile=None, env_path=None, randomness_factor=0.15, env_config=None
):
    """
    Solve a Box1Env environment with randomized A* search.

    Args:
        env_path: Path to load environment from, or None to create a new one
        randomness_factor: Factor to determine how much randomness to introduce (0.0-1.0)

    Returns:
        List of action sequences that solve the environment
    """

    # Create or load environment
    if env_path:
        env = Box1Env.load_json(env_path)
    else:
        # env = Box1Env(grid_n=2, grid_m=5, num_objects=4)
        if env_config is not None:
            env = Box1Env(**env_config)
        else:
            env = Box1Env(grid_n=2, grid_m=4, num_objects=3)
        env.create()
        # env.visualize(out_file_path="tmp-initial.png")

    # print("Environment created:")
    # print(f"Grid: {env.grid_n}x{env.grid_m}")
    # print(f"Objects: {len(env.objects)}")
    # print(f"Robots: {len(env.robots)}")
    # print(f"Using randomness factor: {randomness_factor}")

    # Run planning algorithm
    action_plans = plan_actions(
        env, max_iterations=5000, randomness_factor=randomness_factor, max_plan_number=1
    )

    if not action_plans:
        print("No solution found!")
        return None

    for action_plan in action_plans:
        action_plan = remove_redundant_move(action_plan)
        action_plan = squeeze_move(env, action_plan)

        # Verify plan
        test_env = copy.deepcopy(env)
        for i, step_actions in enumerate(action_plan):
            if not isinstance(step_actions, list):
                step_actions = [step_actions]
            # print(f"Step {i+1}: {len(step_actions)} robot action(s)")
            # for action in step_actions:
            #     print(
            #         f"  Robot {action.robot_id}: {'Carrying' if action.carry else 'Moving'} from {action.pos_s} to {action.pos_e}"
            #     )
            exec_res = env.verify(step_actions)
            # test_env.visualize(
            #     step_actions, exec_res, out_file_path=f"tmp_step_{i}.png"
            # )
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
    # This function reduces the unnecessary move for some robots
    # Basically, it counts whether a robot touches an object after one move, if not then drop it
    # env = copy.deepcopy(env)
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
    # This function try to squeeze some later moves of certain objects to previous
    # Basically, try if the next step actions can be done in previous step
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
# grid_n: [2, 3, 4, 2, 3, 4]
# grid_m: [2, 3, 4, 2, 3, 4]
# num_objects: [1, 1, 1, 2, 2, 2]
# grid_n: [2, 3, 4]
# grid_m: [2, 3, 4]
# num_objects: [2, 2, 2]

grid_n: [ 3, 4]
grid_m: [ 3, 4]
num_objects: [ 2, 2]
robot_mode: "full"
robot_as: "full"
movement: "full"
"""

large_train_env_config_str = """
name: "box1"
# grid_n: []
# grid_n: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# grid_m: [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
# num_objects: [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
robot_mode: "minimal"
robot_as: "full"
movement: "full"
"""


def main():
    config_tuple = [
        # (2, 4, 3),
        (2, 2, 4),
        # (3, 2, 6),
        # (3, 3, 6),
        # (3, 4, 6),
        # (3, 5, 6),
        # (3, 6, 6),
        # (4, 2, 6),
        # (4, 3, 6),
        # (4, 4, 6),
        # (4, 5, 6),
        # (4, 6, 6),
        # (5, 2, 6),
        # (5, 3, 6),
        # (5, 4, 6),
        # (5, 5, 6),
        # (5, 6, 6),
        # (6, 2, 6),
        # (6, 3, 6),
        # (6, 4, 6),
        # (6, 5, 6),
        # (6, 6, 6),
    ]
    # tmp_config =
    # env_configs = generate_configs(yaml.safe_load(train_env_config_str))
    # env_configs["grid_n"] = [x[0] for x in config_tuple]
    # env_configs["grid_m"] = [x[1] for x in config_tuple]
    # env_configs["num_objects"] = [x[2] for x in config_tuple]
    tmp_config = yaml.safe_load(large_train_env_config_str)
    tmp_config["grid_n"] = [x[0] for x in config_tuple]
    tmp_config["grid_m"] = [x[1] for x in config_tuple]
    tmp_config["num_objects"] = [x[2] for x in config_tuple]
    env_configs = generate_configs(tmp_config)
    out_file_path = "train-small-debug.jsonl"

    with open(out_file_path, "w") as f:
        for env_config in env_configs:
            seen_envs: Set[str] = set()
            num_generated = 0
            attempts = 0

            total_unique_envs_per_config = min(
                420,
                count_unique_envs(
                    grid_n=env_config["grid_n"],
                    grid_m=env_config["grid_m"],
                    num_objects=env_config["num_objects"],
                ),
            )
            pbar_desc = f"N={env_config['grid_n']}, M={env_config['grid_m']}, Obj={env_config['num_objects']}"
            with tqdm(total=total_unique_envs_per_config, desc=pbar_desc) as pbar:
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
                    success = solve_box_environment(
                        outfile=f, env_config=env_config, randomness_factor=0.0
                    )
                    if success:
                        num_generated += 1
                        pbar.update(1)


if __name__ == "__main__":
    main()
