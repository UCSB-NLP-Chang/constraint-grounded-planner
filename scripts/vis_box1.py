import os
import click
import random
import numpy as np
from itertools import product

from simulation.simenv.box1 import (
    Box1Env,
    Action,
    is_valid,
    has_object_collision,
    has_robot_collision,
)
from simulation.vis_api import TileMap
from inference.utils import seed_everything


# seed_everything(50)
seed_everything(103)


def generate_action(robots, all_coords):
    robotid = random.choice(list(robots.keys()))
    robot = robots[robotid]
    x = robot.arm_pos
    # x = random.choice(all_coords
    while True:
        y = random.choice(all_coords)
        action = Action(
            robotid,
            x,
            robot.base_pos,
            x,
            y,
        )
        if is_valid(action):
            return [x, y]


def generate_actions(robots, all_coords, num_action):
    robotids = list(robots.keys())
    robotids = random.sample(robotids, k=num_action)

    resactions = []
    for robotid in robotids:
        robot = robots[robotid]
        x = robot.arm_pos
        while True:
            y = random.choice(all_coords)
            action = Action(
                robotid,
                list(x),
                list(robot.base_pos),
                list(x),
                list(y),
            )
            if is_valid(action):
                resactions.append(action)
                break
    return resactions


# def test_box1env(trial_num, outdir="outputs/debug_boxenv2"):
def test_box1env(trial_num, outdir="outputs/debug_boxenv2-shape"):
    os.makedirs(outdir, exist_ok=True)
    trial_result = []
    trial_num = 1
    # seed_everything(100)
    seed_everything(34)
    for trial in range(trial_num):
        # env = Box1Env("Box1Env", 4, 4, 4, "minimal")
        # env = Box1Env("Box1Env", 5, 5, 2, "minimal")
        # env = Box1Env("Box1Env", 3, 3, 1, "full")
        # env = Box1Env("Box1Env", 3, 3, 1, "randrobot")
        env = Box1Env("Box1Env", 4, 4, 1, "randrobot")
        env.create()

        all_coords = list(env.map.keys())
        actions = []

        # for i in range(random.randint(1, 10)):
        #     (x, y) = generate_action(env.robots, all_coords)
        #     actions.append((x, y))

        actions = generate_actions(env.robots, all_coords, min(5, len(env.robots)))
        print(actions)

        # if trial == 0:
        # (x, y) crossing
        # actions.append(((0.0, 0.0), (0.5, 0.5)))
        # actions.append(((0.5, 0.0), (0.0, 0.5)))

        # actions.append(((0.0, 0.0), (0.5, 0.5)))
        # actions.append(((0.5, 0.0), (0.0, 0.0)))
        # action_list = [Action(0, x, x, x, y) for (x, y) in actions]
        action_list = actions
        # if trial == 38:
        #     import ipdb

        #     ipdb.set_trace()
        # verify_res = env.verify(action_list)

        env.visualize(
            actions=action_list,
            exec_res=None,
            out_file_path=os.path.join(outdir, f"action-{trial}.png"),
        )
        # trial_result.append(verify_res)
        break

    # print(f"Trial {trial}", trial_result)
    for trial in range(trial_num):
        print(f"Trial {trial}", trial_result[trial])
        # break


def main():
    # test_box1env(1)
    test_box1env(40)


if __name__ == "__main__":
    main()
