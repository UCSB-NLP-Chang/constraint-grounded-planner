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
import ray
import yaml
import xxhash
from itertools import chain, combinations

from simulation.simenv.box3d.box3d import Box3DEnv

# --- Constants ---
VISUALIZE = False
GRID_SIZE = 0.55
ROUND_DIGITS = 2
DEFAULT_RETRACTION_HEIGHT = 0.21
RETRACTION_XY_OFFSET = 0.3

ALL_POSSIBLE_BOX_POS = []
for n in range(20):
    for m in range(20):
        ALL_POSSIBLE_BOX_POS.append(
            [(n + 0.5) * GRID_SIZE * 2, (m + 0.5) * GRID_SIZE * 2]
        )


def count_unique_envs(grid_n: int, grid_m: int, num_objects: int):
    # Calculate the valid positions within the grid
    valid_positions = []
    for pos in ALL_POSSIBLE_BOX_POS:
        x, y = pos
        # Check if the position is within the grid boundaries
        if 0 < x < grid_n * GRID_SIZE * 2 and 0 < y < grid_m * GRID_SIZE * 2:
            valid_positions.append(pos)

    total_pos = len(valid_positions)

    if 2 * num_objects > total_pos:
        return 0

    # Calculate combinations for object and target positions
    object_combos = math.comb(total_pos, num_objects)
    target_combos = math.comb(total_pos - num_objects, num_objects)
    total_unique = object_combos * target_combos
    return total_unique


@contextmanager
def _timer(label="Elapsed time"):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start:.4f} seconds")


class Box3DStates:
    _hash_val: Optional[int] = None

    def __init__(self, box_env: "Box3DEnv", parent_state=None, current_state=None):
        self.box_env = box_env
        self.parent_state = parent_state
        self.cur_states = current_state
        self.target_positions = {
            k: np.array(list(x) + [DEFAULT_RETRACTION_HEIGHT])
            for k, x in box_env.env.get_target_pos().items()
        }

    def hash(self) -> int:
        if self._hash_val is not None:
            return self._hash_val
        self._hash_val = xxhash.xxh64(self.cur_states.tobytes()).intdigest()
        return self._hash_val

    def _box_positions(self):
        self.box_env.reset(states=self.cur_states)
        return {
            boxname: self.box_env.get_box_pos(boxname)
            for boxname in self.box_env.env.object_names
        }

    def current_positions(self):
        self.box_env.reset(states=self.cur_states)
        return {
            robot_name: self.box_env.get_arm_pos(robot_name)
            for robot_name in self.box_env.env.robot_names
        }

    def is_goal(self) -> bool:
        return all(
            np.allclose(x[:2], y[:2], atol=0.1)
            for x, y in zip(
                self._box_positions().values(), self.target_positions.values()
            )
        )

    def heuristic(self) -> float:
        self.box_env.reset(states=self.cur_states)
        current_positions = self._box_positions().values()
        target_positions = list(self.target_positions.values())
        arm_positions = list(self.current_positions().values())

        h = 0.0
        if any(any(np.isnan(p)) for p in current_positions) or len(
            current_positions
        ) != len(target_positions):
            return float("inf")

        for curr, targ in zip(current_positions, target_positions):
            h += np.sum((np.array(curr)[:2] - np.array(targ)[:2]) ** 2)

        # add penalize when some box not covered by robot
        for box_pos in current_positions:
            if not any(
                np.allclose(box_pos[:2], arm[:2], atol=0.1) for arm in arm_positions
            ):
                h += 1

        return math.sqrt(h)

    def apply_actions(self, action_strings: List[str]) -> "Box3DStates":
        self.box_env.reset(states=self.cur_states)
        out = self.box_env.simulate_one_step(action_strings)
        if out["success"]:
            return Box3DStates(
                self.box_env,
                parent_state=self.cur_states,
                current_state=self.box_env.get_states(),
            )
        else:
            raise ValueError("Invalid action application")


@ray.remote
def verify_parallel_actions_ray(actionstr, state: Box3DStates):
    return verify_parallel_actions(actionstr, state)


def get_possible_nextpos(base_pos):
    close_dis = 1e9
    res = []
    distances = []
    for pos in ALL_POSSIBLE_BOX_POS:
        x, y = pos
        dist = np.hypot(base_pos[0] - x, base_pos[1] - y)
        distances.append((dist, pos))
    k = 4
    distances.sort()
    nearest = [pos for _, pos in distances[:k]]
    return [np.array(x + [DEFAULT_RETRACTION_HEIGHT]) for x in nearest]


def get_possible_safepos(base_pos):
    offsets_x = [-0.5, 0.5, 0, 0]
    offsets_y = [0, 0, -0.5, 0.5]
    return [(base_pos[0] + x, base_pos[1] + y) for x, y in zip(offsets_x, offsets_y)]


def get_closet_safe_pos(base_pos):
    return min(
        get_possible_safepos(base_pos),
        key=lambda p: np.linalg.norm(np.array(p) - np.array(base_pos[:2])),
    )


def l2distance(a, b):
    return np.sum((a[:2] - b[:2]) ** 2)


def generate_single_robot_action(robot_id, state: Box3DStates) -> List[str]:
    robot_actions = []
    base_pos = state.box_env.get_base_pos(robot_id)
    arm_pos = state.current_positions()[robot_id]

    def is_at_target(obj_pos, target_pos):
        return np.allclose(
            np.array(obj_pos[:2]),
            np.array(target_pos[:2]),
            atol=10 ** (-ROUND_DIGITS + 1),
        )

    def is_arm_above_object(arm_pos, obj_pos):
        return np.allclose(
            np.array(arm_pos[:2]), np.array(obj_pos[:2]), atol=10 ** (-ROUND_DIGITS + 1)
        )

    for obj_id, obj_pos in state._box_positions().items():
        target_pos = state.target_positions[obj_id]
        if state.box_env.env.check_reach_range(robot_id, obj_pos):
            if is_at_target(obj_pos, target_pos):
                continue

            possible_next_step = get_possible_nextpos(base_pos)
            cur_distance = l2distance(obj_pos, target_pos)
            best_pos = None
            has_better_pos = False
            best_pos = obj_pos
            for next_pos in possible_next_step:
                tmp_distance = l2distance(next_pos, target_pos)
                if tmp_distance < cur_distance:
                    has_better_pos = True
                    best_pos = next_pos
                    cur_distance = tmp_distance

            is_arm_at_box = is_arm_above_object(arm_pos, obj_pos)
            is_arm_at_best_pos = is_arm_above_object(arm_pos, best_pos)

            action_str = None
            if not is_arm_at_box:
                if has_better_pos:
                    action_str = (
                        f"{robot_id}: Move ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}) False"
                    )
            else:
                if has_better_pos and not is_arm_at_best_pos:
                    cur_distance = l2distance(obj_pos, target_pos)
                    action_str = []
                    for pos in possible_next_step:
                        if is_arm_above_object(pos, obj_pos):
                            continue
                        tmp_distance = l2distance(pos, target_pos)
                        if tmp_distance < cur_distance:
                            action = (
                                f"{robot_id}: Move ({pos[0]:.2f}, {pos[1]:.2f}) True"
                            )
                            action_str.append(action)
                    safe_pos = random.choice(get_possible_safepos(base_pos))
                    action = (
                        f"{robot_id}: Move ({safe_pos[0]:.2f}, {safe_pos[1]:.2f}) False"
                    )
                    action_str.append(action)
                else:
                    safe_pos = random.choice(get_possible_safepos(base_pos))
                    action_str = (
                        f"{robot_id}: Move ({safe_pos[0]:.2f}, {safe_pos[1]:.2f}) False"
                    )

            if action_str is None:
                continue
            if isinstance(action_str, str):
                action_str = [action_str]

            for action in action_str:
                if action not in robot_actions:
                    robot_actions.append(action)

    return robot_actions


def verify_parallel_actions(actionstr, state: Box3DStates):
    if isinstance(actionstr, list):
        actionstr = "\n".join(actionstr)
    box_env = state.box_env
    box_env.reset(states=state.cur_states)
    out = box_env.simulate_one_step(actionstr)
    if out["success"]:
        new_state = Box3DStates(
            state.box_env,
            parent_state=state.cur_states,
            current_state=box_env.get_states(),
        )
        box_env.reset(states=state.cur_states)
        return True, new_state
    else:
        return False, None


def generate_potential_actions_box3d(state: Box3DStates) -> List[str]:
    box_env = state.box_env
    env = box_env.env
    robot_names = sorted(env.robot_names)

    single_robot_potential_actions = {
        r: generate_single_robot_action(r, state) for r in robot_names
    }

    valid_action_sets = []
    verify_tasks = []
    action_lookup = []

    for robot_id, actions in single_robot_potential_actions.items():
        for action in actions:
            verify_tasks.append(verify_parallel_actions_ray.remote([action], state))
            action_lookup.append([action])

    active_robots = [r for r in robot_names if single_robot_potential_actions[r]]
    if len(active_robots) >= 2:
        for group_size in range(2, min(4, len(state.box_env.env.object_names)) + 1):
            for robot_group in itertools.combinations(active_robots, group_size):
                combos = list(
                    itertools.product(
                        *[single_robot_potential_actions[r] for r in robot_group]
                    )
                )
                combos = [[a for a in combo] for combo in combos]
                for combo in combos:
                    verify_tasks.append(
                        verify_parallel_actions_ray.remote(combo, state)
                    )
                    action_lookup.append(combo)

    verify_results = ray.get(verify_tasks)

    for i, (success, new_state) in enumerate(verify_results):
        if success:
            valid_action_sets.append(["\n".join(action_lookup[i]), new_state])

    valid_action_sets.sort(
        key=lambda x: (x[1].heuristic(), -x[0].count("True"), -len(x[0].split("\n")))
    )
    valid_action_sets = valid_action_sets[:20]
    return valid_action_sets


def reconstruct_path_box3d(came_from, final_hash):
    actions_sequence = []
    current_hash = final_hash
    while current_hash in came_from:
        parent_hash, action = came_from[current_hash]
        if action is not None:
            actions_sequence.append(action)
        if parent_hash is None:
            break
        current_hash = parent_hash
    actions_sequence.reverse()
    return actions_sequence


def search_plan(box_env, max_iterations=200):
    open_set = []
    closed_set = set()

    try:
        current_state = Box3DStates(box_env, current_state=box_env.get_states())
        initial_hash = current_state.hash()
    except Exception as e:
        print(f"Failed to initialize initial state: {e}")
        return None

    g_scores = {initial_hash: 0.0}
    came_from = {initial_hash: (None, None)}
    states = {initial_hash: current_state}

    heapq.heappush(open_set, (current_state.heuristic(), random.random(), initial_hash))
    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        _, _, current_hash = heapq.heappop(open_set)

        if current_hash in closed_set:
            continue

        current_state = states[current_hash]
        if current_state.is_goal():
            return reconstruct_path_box3d(came_from, current_hash)

        closed_set.add(current_hash)

        possible_action_sets = generate_potential_actions_box3d(current_state)
        for action, next_state in possible_action_sets:
            next_hash = next_state.hash()

            if next_hash in closed_set:
                continue

            tentative_g_score = g_scores[current_hash] + 0.5 / len(action.split("\n"))
            if next_hash not in g_scores or tentative_g_score < g_scores[next_hash]:
                came_from[next_hash] = (current_hash, action)
                g_scores[next_hash] = tentative_g_score

                if next_hash not in states:
                    states[next_hash] = next_state
                    f_score = next_state.heuristic()
                    heapq.heappush(open_set, (f_score, tentative_g_score, next_hash))

    return None


def solve_box3d_environment(outfile=None, env_path=None, env_config=None):
    if env_path:
        env = Box3DEnv.load_json(env_path)
    else:
        if env_config is not None:
            env = Box3DEnv(**env_config, do_render=False)
        else:
            env = Box3DEnv(grid_n=3, grid_m=3, num_objects=3, do_render=False)
        # env.create()

    plan = search_plan(env, max_iterations=200)

    if not plan:
        return False

    # import ipdb

    # ipdb.set_trace()
    env.reset()
    all_steps_succeeded = True
    if VISUALIZE:
        env.do_render = True
        env.env.do_render = True

    out = env.simulate_full_step(plan)

    if out["success"]:
        out_obj = {"env": env.to_json(), "plan": plan}
        if outfile is not None:
            outfile.write(json.dumps(out_obj) + "\n")
            outfile.flush()

    if ray.is_initialized():
        # destroy, clean up cache dir, reinit
        session_dir = ray._private.worker._global_node._session_dir
        ray.shutdown()
        import gc
        import shutil

        gc.collect()
        shutil.rmtree(session_dir, ignore_errors=True)
        ray.init(_temp_dir=get_cache_dir())  # reinit

    return out["success"]


def find_robotname(action):
    return action.split(":")[0].strip()


def powerset_sorted_desc(lst):
    ps = list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))
    ps_sorted = sorted(ps, key=lambda x: len(x), reverse=True)
    return ps_sorted


def squeeze_move(base_env, actions):
    import ipdb

    ipdb.set_trace()
    env = copy.deepcopy(base_env)
    new_actions = []
    i = 1
    cur_step_actions = actions[0]
    while i < len(actions):
        step_action = actions[i]
        splitted_action = step_action.split("\n")
        cur_states = env.get_states()
        potential_merge = [
            x for x in splitted_action if find_robotname(x) in cur_step_actions
        ]
        ok = False
        for potential_action in powerset_sorted_desc(potential_merge):
            if cur_step_actions != "":
                tmp_step_actions = cur_step_actions + "\n" + "\n".join(potential_action)
            else:
                tmp_step_actions = "\n".join(potential_action)

            out = env.simulate_one_step(tmp_step_actions)

            if out["success"]:
                ok = True  # can merge
                cur_step_actions = tmp_step_actions
                break
            else:
                env.reset(states=cur_states)

        if not ok:
            env.reset(states=cur_states)
            env.simulate_one_step(cur_step_actions)
            new_actions.append(cur_step_actions)
            cur_step_actions = ""

        i += 1
    if cur_step_actions:
        new_actions.append(cur_step_actions)
    return new_actions


def get_cache_dir():
    import socket
    import getpass

    hostname = socket.gethostname()
    username = getpass.getuser()
    print("Hostname: ", hostname)
    if "bluevela" in hostname:
        # if "kqian" in username:
        #     return "/proj/long-multi/kqian/granite-3b-ablate/.r"
        # elif "becon" in username:
        #     return "/proj/long-multi/midi/.ray_cache"
        # return "/tmp/ray"
        return "/tmp"
    elif "katrina1" in hostname:
        return "/mnt/data3/jiabao/.ray_cache/"
    elif "wireless" in hostname:  # run in local
        return "/tmp/ray_cache"
    elif "local" in hostname:
        return "/tmp/ray_cache"
    else:
        raise NotImplementedError


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# def main(arg_configs):
#     if not ray.is_initialized():
#         ray.init(_temp_dir=get_cache_dir())

#     # seed_everything(44)
#     # init_seed = 42
#     # init_seed = 1042
#     init_seed = 3042
#     config_tuple = [
#         (2, 2, 1),
#         (2, 3, 1),
#         (2, 4, 1),
#         (3, 2, 1),
#         (3, 3, 1),
#         (3, 4, 1),
#         (4, 2, 1),
#         (4, 3, 1),
#         (4, 4, 1),
#         (2, 2, 2),
#         (2, 3, 2),
#         (2, 4, 2),
#         (3, 2, 2),
#         (3, 3, 2),
#         (3, 4, 2),
#         (4, 2, 2),
#         (4, 3, 2),
#         (4, 4, 2),
#         (2, 2, 3),
#         (2, 3, 3),
#         (2, 4, 3),
#         (3, 2, 3),
#         (3, 3, 3),
#         (3, 4, 3),
#         (4, 2, 3),
#         (4, 3, 3),
#         (4, 4, 3),
#         (2, 2, 4),
#         (2, 3, 4),
#         (2, 4, 4),
#         (3, 2, 4),
#         (3, 3, 4),
#         (3, 4, 4),
#         (4, 2, 4),
#         (4, 3, 4),
#         (4, 4, 4),
#     ]

#     # config_tuple = [(2, 3, 2)]
#     NUM_PER_CONFIG = 60

#     split_idx = arg_configs.numobj
#     basedir = "./traces-more"
#     os.makedirs(basedir, exist_ok=True)
#     out_file_path = f"{basedir}/train3d-large-full-obj@{split_idx}.jsonl"

#     with open(out_file_path, "w") as f:
#         for grid_n, grid_m, num_objects in config_tuple:
#             if num_objects < split_idx:
#                 continue
#             if num_objects > split_idx:
#                 break

#             seen_envs: Set[str] = set()
#             num_generated = 0
#             attempts = 0

#             total_unique_envs_per_config = min(
#                 NUM_PER_CONFIG,
#                 count_unique_envs(
#                     grid_n=grid_n,
#                     grid_m=grid_m,
#                     num_objects=num_objects,
#                 ),
#             )
#             pbar_desc = f"N={grid_n}, M={grid_m}, Obj={num_objects}"
#             not_found = 20
#             with tqdm(total=total_unique_envs_per_config, desc=pbar_desc) as pbar:
#                 while num_generated < total_unique_envs_per_config:
#                     attempts += 1
#                     seed_everything(init_seed + attempts)

#                     env_config = {
#                         "grid_n": grid_n,
#                         "grid_m": grid_m,
#                         "num_objects": num_objects,
#                         "robot_mode": "full",
#                         # "do_render": False,
#                     }

#                     env = Box3DEnv(**env_config)
#                     env_json = json.dumps(env.to_json(), sort_keys=True)

#                     if env_json in seen_envs:
#                         continue
#                     seen_envs.add(env_json)

#                     ok = solve_box3d_environment(outfile=f, env_config=env_config)
#                     if ok:
#                         num_generated += 1
#                         pbar.update(1)
#                         not_found = 100
#                         print("✅" * 50)
#                         print("Plus 1", env_config)
#                         print("✅" * 50)
#                     else:
#                         not_found -= 1
#                         print(f"Failed to solve environment: {env_config}")
#                     if not_found < 0:
#                         break


def main(arg_configs):
    if not ray.is_initialized():
        ray.init(_temp_dir=get_cache_dir())

    init_seed = 542
    config_tuple = [
        (2, 2, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 2, 1),
        (3, 3, 1),
        (3, 4, 1),
        (4, 2, 1),
        (4, 3, 1),
        (4, 4, 1),
        (2, 2, 2),
        (2, 3, 2),
        (2, 4, 2),
        (3, 2, 2),
        (3, 3, 2),
        (3, 4, 2),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 2),
        (2, 2, 3),
        (2, 3, 3),
        (2, 4, 3),
        (3, 2, 3),
        (3, 3, 3),
        (3, 4, 3),
        (4, 2, 3),
        (4, 3, 3),
        (4, 4, 3),
        (2, 2, 4),
        (2, 3, 4),
        (2, 4, 4),
        (3, 2, 4),
        (3, 3, 4),
        (3, 4, 4),
        (4, 2, 4),
        (4, 3, 4),
        (4, 4, 4),
    ]

    NUM_PER_CONFIG = 60
    split_idx = arg_configs.numobj
    job_idx = arg_configs.job_idx
    num_jobs = arg_configs.num_jobs

    # import ipdb

    # ipdb.set_trace()
    # Filter by obj count
    config_tuple = [cfg for cfg in config_tuple if cfg[2] == split_idx]

    # # Split configs across jobs
    config_tuple.sort()
    chunk_size = math.ceil(len(config_tuple) / num_jobs)
    start_idx = job_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(config_tuple))
    config_subset = config_tuple[start_idx:end_idx]

    basedir = "./traces-more-5"
    os.makedirs(basedir, exist_ok=True)
    out_file_path = (
        f"{basedir}/train3d-job{job_idx}-of-{num_jobs}-obj@{split_idx}.jsonl"
    )

    # config_subset = config_tuple
    maximum_plan = 0
    with open(out_file_path, "w") as f:
        for grid_n, grid_m, num_objects in config_subset:
            seen_envs: Set[str] = set()
            num_generated = 0
            attempts = 0

            total_unique_envs_per_config = min(
                NUM_PER_CONFIG,
                count_unique_envs(
                    grid_n=grid_n,
                    grid_m=grid_m,
                    num_objects=num_objects,
                ),
            )
            pbar_desc = f"N={grid_n}, M={grid_m}, Obj={num_objects}"
            not_found = 20

            maximum_plan += total_unique_envs_per_config
            # continue
            with tqdm(total=total_unique_envs_per_config, desc=pbar_desc) as pbar:
                while num_generated < total_unique_envs_per_config:
                    attempts += 1
                    seed_everything(init_seed + attempts)

                    env_config = {
                        "grid_n": grid_n,
                        "grid_m": grid_m,
                        "num_objects": num_objects,
                        "robot_mode": "full",
                    }

                    env = Box3DEnv(**env_config)
                    env_json = json.dumps(env.to_json(), sort_keys=True)

                    if env_json in seen_envs:
                        continue
                    seen_envs.add(env_json)

                    ok = solve_box3d_environment(outfile=f, env_config=env_config)
                    if ok:
                        num_generated += 1
                        pbar.update(1)
                        print("✅" * 50)
                        print("Plus 1", env_config)
                        print("✅" * 50)
                        not_found = 100
                    else:
                        not_found -= 1
                    if not_found < 0:
                        break

    print("Maximum plan: ", maximum_plan)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--numobj", type=int, default=1, help="Number of objects to use"
    )
    parser.add_argument("--job_idx", type=int, default=0, help="Index of this job")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs")
    configs = parser.parse_args()
    main(configs)

# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument("--numobj", type=int, default=1)
#     configs = parser.parse_args()
#     print("HIHIHIHIHI ", configs.numobj)
#     main(configs)
