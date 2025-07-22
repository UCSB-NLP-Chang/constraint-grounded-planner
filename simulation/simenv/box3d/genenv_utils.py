# TODO some functions to implement for the 3d env
import re
import os
import copy
import time
import json
import cv2
import random
import numpy as np
from pydantic import dataclasses, validator
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from lxml import etree
import mujoco
from collections import defaultdict
from functools import lru_cache

import shutil
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from .base_env import MujocoSimEnv, EnvState
from .robot import SimRobot
from .env_utils import visualize_voxel_scene
from .rrt_multi_arm import MultiArmRRT
from .prompting.parser import LLMResponseParser
from .prompting.feedback import FeedbackManager
from .policy import PlannedPathPolicy

from simulation.colors import *

colornames = [
    "cube-purple",
    "cube-red",
    "cube-blue",
    "cube-green",
    "cube-orange",
    "cube-yellow",
    "cube-pink",
    "cube-gray",
]


def generate_grid_xml(N, M, size):
    """
    Generate MuJoCo XML for a grid map environment.

    Parameters:
    - N: Number of tiles along the x-axis
    - M: Number of tiles along the y-axis
    - size: Size of each tile (half-length)

    Returns:
    - String containing the MuJoCo XML for the grid map

    <body name="grid_tiles" pos="0 0 0.001">
      <!-- 3x3 Tiles: Each tile is 1x1m (size=0.5) -->
      <geom name="tile_0_0" pos="-1.0 -1.0 0" size="0.5 0.5 0.001" type="box" rgba="0.8 0.8 0.8 1"/>
      <geom name="tile_0_1" pos="-1.0  0.0 0" size="0.5 0.5 0.001" type="box" rgba="0.9 0.9 0.9 1"/>
      <geom name="tile_0_2" pos="-1.0  1.0 0" size="0.5 0.5 0.001" type="box" rgba="0.8 0.8 0.8 1"/>
        ...
      <!-- Vertical grid lines (X direction) -->
      <geom name="vline_1" pos="-0.5 0 0.011" size="0.0025 1.5 0.001" type="box" rgba="0 0 0 1"/>

      <!-- Horizontal grid lines (Y direction) -->
      <geom name="hline_1" pos="0 -0.5 0.011" size="1.5 0.0025 0.001" type="box" rgba="0 0 0 1"/>
    </body>
    """

    # xml_lines = ['<body name="table" pos="0 0 0.000">']
    xml_lines = ['<body name="table" pos="0 0.0 0">']
    xml_lines.append(
        f"    <!-- {N}x{M} Tiles: Each tile is {size * 2}x{size * 2}m (size={size}) -->"
    )
    xml_lines.append(
        f"""
        <geom name="table_collision" pos="{size * 3} {size * 3} -0.02" size="{size * (N + 1)} {size * (M + 1)} 0.02" type="box" group="0" friction="1 0.005 0.0001" rgba="1 0 0 0" />
      """
    )

    # Adjust positions to ensure all coordinates are positive
    # We'll shift everything so the bottom-left corner is at (0,0)
    for i in range(N):
        for j in range(M):
            # Calculate tile position
            x_pos = i * (size * 2) + size
            y_pos = j * (size * 2) + size
            color = "0.8 0.8 0.8 1" if (i + j) % 2 == 0 else "0.9 0.9 0.9 1"

            # Add tile geometry
            xml_lines.append(
                f'    <geom name="tile_{i}_{j}" pos="{x_pos} {y_pos} 0" size="{size} {size} 0.01" type="box" rgba="{color}" friction="1 0.005 0.0001" mass="0.00001" conaffinity="0" contype="0"  group="1" />'
            )

    # Calculate grid dimensions
    grid_width = N * (size * 2)
    grid_height = M * (size * 2)

    # Add vertical grid lines (lines along y-axis)
    for i in range(N + 1):
        x_pos = i * (size * 2)
        xml_lines.append(
            f'    <geom name="vline_{i + 1}" pos="{x_pos} {(grid_height / 2)} 0.01" size="0.0025 {grid_height / 2} 0.01" type="box" rgba="0 0 0 1" contype="0" conaffinity="0" />'
        )

    # Add horizontal grid lines (lines along x-axis)
    for j in range(M + 1):
        y_pos = j * (size * 2)
        xml_lines.append(
            f'    <geom name="hline_{j + 1}" pos="{(grid_width / 2)} {y_pos} 0.01" size="{grid_width / 2} 0.0025 0.01" type="box" rgba="0 0 0 1" contype="0" conaffinity="0" />'
        )

    xml_lines.append("</body>")

    return "\n".join(xml_lines)


BASE_ARM_XML_MAPPING = {"suction": "assets/"}


def generate_box_xmls(num):
    #   <geom type="box" material="cube-blue" size='0.05 0.05 0.05' density="1000" />
    TEMPLATE_XML = """
    <body name="blue_square" pos="0 0.0 0.25">
      <freejoint name="blue_square_joint" />
      <site name="blue_square_top" pos="0 0 0.03" class="site_top" />
      <geom type="box" material="cube-blue" size='0.1 0.1 0.02' density="1000" />
      <body name="blue_square_weld" pos="0 0 0.06" /> 
    </body>
"""

    #   <geom type="box" material="cube-blue" size='0.1 0.1 0.03' density="1000" />
    TEMPLATE_XML = """
    <body name="blue_square" pos="0 0.0 0.25">
      <freejoint name="blue_square_joint" />
      <site name="blue_square_top" pos="0 0 0.03" class="site_top" />
      <geom type="box" material="cube-blue" size='0.1 0.1 0.03' density="1000" />
      <body name="blue_square_weld" pos="0 0 0.1" /> 
    </body>
"""
    result = []
    for i in range(num):
        materialname = colornames[i % len(colornames)]
        xmlobj = replace_prefix_in_mujoco_xml(
            TEMPLATE_XML,
            "blue_square",
            "box_" + str(i),
        )
        xmlobj.find("geom").attrib["material"] = materialname
        result.append(xmlobj)
    return result


def replace_prefix_in_mujoco_xml(xml_source, old_prefix, new_prefix, output_path=None):
    if os.path.exists(xml_source):  # From a file
        tree = ET.parse(xml_source)
    else:  # from a xml string
        tree = ET.fromstring(xml_source)

    if hasattr(tree, "getroot"):
        root = tree.getroot()
    else:
        root = tree

    def replace_attributes(element):
        for attr in element.attrib:
            val = element.attrib[attr]
            if isinstance(val, str) and old_prefix in val:
                element.attrib[attr] = val.replace(old_prefix, new_prefix)
        for child in element:
            replace_attributes(child)

    replace_attributes(root)
    if output_path is not None:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return tree


def generate_env(
    n,
    m,
    num_obj,
    robot_model,
    tmpdirname,
    grid_size=0.4,
    return_xml_path=True,
    obj_coords=None,  # if None, then randomly generate
    obj_targets=None,  # if None, then randomly generate
):
    # print("created temporary directory", tmpdirname)
    tmpdirname = Path(tmpdirname)

    # Create assets directory
    assets_dir = tmpdirname / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Read base XML file
    with open("assets/task_box_empty.xml", "r") as f:
        base_xml = f.read()

    # Create XML tree from base file
    tree = ET.fromstring(base_xml)
    root = tree.find("worldbody")

    # Add grid tiles
    grid_tile_xml = generate_grid_xml(n, m, grid_size)
    grid_tile_body = ET.fromstring(grid_tile_xml)
    root.append(grid_tile_body)

    padding_size = grid_size * 2
    robot_names = []
    robot_id = 0

    # Add robots
    for i in range(0, n - 1):
        for j in range(0, m - 1):
            robot_name = f"robot_{robot_id}"
            robot_id += 1
            robot_names.append(robot_name)

            # Read robot XML
            with open("assets/ur5e/ur5e_full_suction.xml", "r") as f:
                robot_xml = f.read()

            # Replace robot name in XML
            robot_xml = robot_xml.replace("ur5e_1_", f"{robot_name}_")
            robot_body = ET.fromstring(robot_xml).find("body")

            position = f"{(i + 1) * padding_size} {(j + 1) * padding_size} 0.001"
            site_elem = ET.Element(
                "body",
                attrib={
                    "name": robot_name,
                    "pos": position,
                },
            )
            site_elem.append(robot_body)
            root.append(site_elem)

            # Read and add actuator XML
            with open(
                "assets/ur5e/ur5e_1_assets_actuator.xml",
                "r",
            ) as f:
                actuator_xml = f.read()

            actuator_xml = actuator_xml.replace("ur5e_1_", f"{robot_name}_")
            actuator_body = ET.fromstring(actuator_xml).find("actuator")
            tree.append(actuator_body)

    def rand_coords(n, m):
        random_x = np.random.choice(np.arange(n)).item()
        random_y = np.random.choice(np.arange(m)).item()
        pos = (
            (random_x + 0.5) * padding_size,
            (random_y + 0.5) * padding_size,
        )
        return pos

    # Add box objects
    box_objs = generate_box_xmls(num_obj)
    box_poses = []
    box_names = []
    box_targets = []
    seen_coords = set()
    seen_target_coords = set()

    if obj_coords is not None:
        assert len(obj_coords) == num_obj
        assert obj_targets is not None
        assert len(obj_targets) == num_obj

    for i in range(num_obj):
        if obj_coords is None:
            pos = rand_coords(n, m)
            while tuple(pos) in seen_coords:
                pos = rand_coords(n, m)
            target_pos = rand_coords(n, m)
            while target_pos == pos or tuple(target_pos) in seen_target_coords:
                target_pos = rand_coords(n, m)
        else:
            pos = obj_coords[i]
            # if i in obj_targets:
            if isinstance(obj_targets, list):
                target_pos = obj_targets[i]
            else:
                target_pos = obj_targets[f"box_{i}"]

        seen_coords.add(tuple(pos))
        seen_target_coords.add(tuple(target_pos))
        box_objs[i].set(
            "pos",
            f"{pos[0]} {pos[1]} 0.18",
        )
        box_names.append(f"box_{i}")
        box_poses.append(pos)
        box_targets.append((target_pos[0], target_pos[1]))
        root.append(box_objs[i])

        # add a non-collision geom at the target position for visualization
        target_body = ET.SubElement(root, "body")
        target_geom = ET.SubElement(target_body, "geom")
        target_geom.set("name", f"target_{i}-cylinder")
        target_geom.set("pos", f"{target_pos[0]} {target_pos[1]} 0.005")
        target_geom.set("size", "0.17 0.025")
        target_geom.set("type", "cylinder")
        target_geom.set("contype", "0")
        target_geom.set("conaffinity", "0")
        target_geom.set("material", colornames[i % len(colornames)])

        target_geom = ET.SubElement(target_body, "geom")
        target_geom.set("name", f"target_{i}-box")
        target_geom.set("pos", f"{target_pos[0]} {target_pos[1]} 0.005")
        target_geom.set("size", "0.1 0.1 0.03")
        target_geom.set("type", "box")
        target_geom.set("rgba", "1 1 1 0.3")
        target_geom.set("contype", "0")
        target_geom.set("conaffinity", "0")

    # Add keyframe
    keyframe = ET.Element("keyframe")
    key = ET.SubElement(keyframe, "key")
    key.set("name", "home")
    robot_ctrl = "0 -1.57079 1.57079 -1.57079 -1.57079 0 0"
    key.set("ctrl", " ".join([robot_ctrl] * len(robot_names)))
    key.set(
        "qpos",
        " ".join(
            ["3.14158 -1.57079 1.57079 -1.57079 -1.57079 0"] * len(robot_names)
            + [f"{pos[0]} {pos[1]} 0.1 0 0 0 0" for pos in box_poses]
        ),
    )
    tree.append(keyframe)

    # Add debug spheres
    # coords = [
    #     (0, 0),
    #     (0, m * padding_size),
    #     (n * padding_size, 0),
    #     (n * padding_size, m * padding_size),
    # ]
    # for coord in coords:
    #     sphere = ET.SubElement(root, "geom")
    #     x, y = coord
    #     sphere.set("name", f"debug-sphere-{x}-{y}")
    #     sphere.set("pos", f"{x} {y} 1.0")
    #     if coord == (0, 0):
    #         sphere.set("rgba", "1 0 0 1")
    #     elif coord == (n * padding_size, m * padding_size):
    #         sphere.set("rgba", "0 0 1 1")
    #     else:
    #         sphere.set("rgba", "0 1 0 1")
    #     sphere.set("size", "0.15")

    # # Add debug location spheres
    # coords = [
    #     ((x + 0.5) * padding_size, (y + 0.5) * padding_size)
    #     for x in range(n)
    #     for y in range(m)
    # ]
    # for idx, coord in enumerate(coords):
    #     sphere = ET.SubElement(root, "geom")
    #     x, y = coord
    #     sphere.set("name", f"debug-sphere2-{x}-{y}")
    #     sphere.set("pos", f"{x} {y} 1.5")
    #     sphere.set("size", "0.1")
    #     sphere.set("rgba", f"0 0 {0.05 * idx} 1")

    # Add grip equality
    equality = ET.Element("equality")
    for robot_name in robot_names:
        for obj_name in box_names:
            weld_elem = ET.SubElement(equality, "weld")
            weld_elem.set("name", f"{obj_name}_top_{robot_name}_suction")
            weld_elem.set("body1", f"{obj_name}_weld")
            weld_elem.set("body2", f"{robot_name}_suction_tip_1")
            weld_elem.set("relpose", "0 0 0 1 0 0 0")
            weld_elem.set("active", "false")
    tree.append(equality)

    # Write final XML file
    output_path = tmpdirname / "task_box.xml"
    tree_str = ET.tostring(tree, encoding="utf-8", method="xml")

    if return_xml_path:
        with open(output_path, "wb") as f:
            f.write(tree_str)

        return {
            "xml": str(output_path),
            "targets": box_targets,
            "obj_coords": box_poses,
        }
    else:
        return {
            "xml": tree_str,
            "targets": box_targets,
            "obj_coords": box_poses,
        }


class Box3DEnv:
    def __init__(
        self,
        grid_n,
        grid_m,
        num_obj,
        robot_mode,
        **kwargs,
    ):
        self.grid_n = grid_n
        self.grid_m = grid_m
        self.num_obj = num_obj
        self.robot_mode = robot_mode

        self.env = BoxTask(
            filepath="",
            grid_n=self.grid_n,
            grid_m=self.grid_m,
            num_obj=self.num_obj,
            robot_mode=self.robot_mode,
            task_objects=[],
        )
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=self.env.get_sim_robots(),
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        self.response_keywords = ["NAME", "ACTION"]
        self.direct_waypoints = 5  # BY DEFAULT
        self.max_failed_waypoints = 1
        self.parser = LLMResponseParser(
            self.env,
            "action",
            self.env.robot_name_map,
            self.response_key_words,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace,
            split_parsed_plans=False,
        )
        self.feedback_manager = FeedbackManager(
            env=self.env,
            planner=self.planner,
            llm_output_mode="action",
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,
        )
        self.policy_kwargs = dict(
            control_freq=50,
            use_weld=1,
            skip_direct_path=0,
            skip_smooth_path=0,
            check_relative_pose=False,
        )

    def reset(self):
        self.env.reset()

    def simulate_one_step(self, response):
        result_summary = {}

        # This is one step
        obs = self.env.get_obs()
        origin_sim_data = self.env.save_intermediate_state()

        parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response)
        if not parse_succ:
            print("Parse failed")
            # execute_str = "EXECUTE" + response.split("EXECUTE")[-1]
            # curr_feedback = "Parse failed"
            ready_to_execute = False

            result_summary["success"] = False
            result_summary["reason"] = "ParseError"
        else:
            print("Parse ok")
            ready_to_execute = True
            for j, llm_plan in enumerate(llm_plans):
                ready_to_execute, env_feedback = self.feedback_manager.give_feedback(
                    llm_plan
                )
                if not ready_to_execute:  # Potentially conflict
                    curr_feedback = env_feedback
                    result_summary["success"] = False
                    if "Reachability failed" in curr_feedback:
                        result_summary["reason"] = "InvalidAction"
                    elif "IK failed" in curr_feedback:
                        result_summary["reason"] = "InvalidAction"
                    elif "Collision" in curr_feedback:
                        result_summary["reason"] = "CollisionRobot"

        if not ready_to_execute:
            return result_summary

        rewind_env = False
        # This point we can really execute
        for i, plan in enumerate(llm_plans):
            print("tograsp:", plan.tograsp, "inhand:", plan.inhand, plan.action_strs)
            policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.robots,
                path_plan=plan,
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                plan_splitted=self.split_parsed_plans,
                **self.policy_kwargs,
            )

            num_sim_steps = 0
            plan_success, reason = policy.plan(self.env)
            print(f"Plan success: {plan_success}, reason: {reason}")
            if plan_success:
                print(f"Execute the plan for {len(policy.action_buffer)} steps")

                while not policy.plan_exhausted:
                    sim_action = policy.act(obs, self.env.physics)
                    obs, reward, done, info = self.env.step(sim_action, verbose=False)
                    num_sim_steps += 1

            if num_sim_steps > 0:
                vid_name = "debug/execute.mp4"
                self.env.export_render_to_video(vid_name, out_type="mp4", fps=30)
                result_summary["success"] = True
            else:
                print("Plan execute failed")
                result_summary["success"] = False
                result_summary["detail"] = "PlanExecuteFailed: " + reason
                rewind_env = True
                break

            if done:
                break

        if rewind_env:
            # rewind env
            self.env.load_saved_state(origin_sim_data)
            return False  # Failed execution
        else:
            # update sim
            origin_sim_data = self.env.save_intermediate_state()
            success = reward > 0
            if success:  #! reach target
                return True
            else:
                return False  # not done but no error

    def simulate_full_step(self, response):
        step_responses = json.loads(response)  # suppose it is a json string

        for step, step_response in enumerate(step_responses):
            cur_result_summary = self.simulate_one_step(step_response)
            if not cur_result_summary["success"]:
                return cur_result_summary

        return cur_result_summary


class BoxTask(MujocoSimEnv):
    def __init__(
        self,
        filepath,
        grid_n,
        grid_m,
        num_obj,
        robot_mode,
        task_objects,
        #
        agent_configs=...,
        render_cameras=...,
        image_hw=...,
        render_freq=20,
        home_qpos=None,
        sim_forward_steps=100,
        sim_save_freq=100,
        home_keyframe_id=0,
        error_threshold=0.001,
        error_freq=3,
        randomize_init=True,
        np_seed=0,
        render_point_cloud=False,
        skip_reset=False,
        obj_coords=None,
        obj_targets=None,
        **kwargs,
    ):
        self.grid_n = grid_n
        self.grid_m = grid_m
        self.num_obj = num_obj
        self.robot_mode = robot_mode

        env_res = generate_env(
            self.grid_n,
            self.grid_m,
            self.num_obj,
            self.robot_mode,
            tmpdirname=os.path.dirname(__file__),
            grid_size=0.55,
            return_xml_path=False,
            obj_coords=obj_coords,
            obj_targets=obj_targets,
        )
        # print(self.target_pos)
        self.env_xml = env_res["xml"]
        self.target_pos = env_res["targets"]
        self.obj_coords = env_res["obj_coords"]
        self.object_names = self.find_all_objects(self.env_xml)
        self.robot_names = self.find_all_robots(self.env_xml)

        self.robot_name_map = {k: k for k in self.robot_names}
        self.robot_name_map_inv = {k: k for k in self.robot_names}

        agent_configs = {
            robot_name: self.bulid_robot_constants(robot_name)
            for robot_name in self.robot_names
        }
        task_objects = self.object_names

        super(BoxTask, self).__init__(
            self.env_xml,
            task_objects,
            agent_configs,
            render_cameras,
            image_hw,
            render_freq,
            home_qpos,
            sim_forward_steps,
            sim_save_freq,
            home_keyframe_id,
            error_threshold,
            error_freq,
            randomize_init,
            np_seed,
            render_point_cloud,
            skip_reset,
            do_render=kwargs.get("do_render", True),
        )

        self.robots = {
            robot_name: SimRobot(
                physics=self.physics,
                use_ee_rest_quat=False,
                base_joint="",
                **self.bulid_robot_constants(robot_name),
            )
            for robot_name in self.robot_names
        }
        self.align_threshold = 0.1
        # self.object_coords = dict()

    def bulid_robot_constants(self, robot_name):
        UR5E_SUCTION_CONSTANTS = dict(
            name="ur5e_1_",
            all_joint_names=[
                "ur5e_1_shoulder_pan_joint",
                "ur5e_1_shoulder_lift_joint",
                "ur5e_1_elbow_joint",
                "ur5e_1_wrist_1_joint",
                "ur5e_1_wrist_2_joint",
                "ur5e_1_wrist_3_joint",
                # "ur5e_1_base_joint",
            ],
            ik_joint_names=[
                "ur5e_1_shoulder_pan_joint",
                "ur5e_1_shoulder_lift_joint",
                "ur5e_1_elbow_joint",
                "ur5e_1_wrist_1_joint",
                "ur5e_1_wrist_2_joint",
                "ur5e_1_wrist_3_joint",
                # "ur5e_1_base_joint",
            ],
            arm_joint_names=[
                "ur5e_1_shoulder_pan_joint",
                "ur5e_1_shoulder_lift_joint",
                "ur5e_1_elbow_joint",
                "ur5e_1_wrist_1_joint",
                "ur5e_1_wrist_2_joint",
                "ur5e_1_wrist_3_joint",
            ],
            actuator_info={
                "ur5e_1_shoulder_pan_joint": "ur5e_1_shoulder_pan",
                "ur5e_1_shoulder_lift_joint": "ur5e_1_shoulder_lift",
                "ur5e_1_elbow_joint": "ur5e_1_elbow",
                "ur5e_1_wrist_1_joint": "ur5e_1_wrist_1",
                "ur5e_1_wrist_2_joint": "ur5e_1_wrist_2",
                "ur5e_1_wrist_3_joint": "ur5e_1_wrist_3",
                # "ur5e_1_base_joint": "ur5e_1_base",
            },
            all_link_names=[
                "ur5e_1_shoulder_link",
                "ur5e_1_upper_arm_link",
                "ur5e_1_forearm_link",
                "ur5e_1_wrist_1_link",
                "ur5e_1_wrist_2_link",
                "ur5e_1_wrist_3_link",
                "ur5e_1_suction_gripper",
                "ur5e_1_suction_base",
                "ur5e_1_suction_midLink",
                "ur5e_1_suction_headLink",
                "ur5e_1_suction_tipLink",
                "ur5e_1_suction_disk",
            ],
            arm_link_names=[
                "ur5e_1_shoulder_link",
                "ur5e_1_upper_arm_link",
                "ur5e_1_forearm_link",
                "ur5e_1_wrist_1_link",
                "ur5e_1_wrist_2_link",
                "ur5e_1_wrist_3_link",
            ],
            ee_link_names=[
                "ur5e_1_suction_headLink",
                "ur5e_1_suction_tipLink",
                "ur5e_1_suction_disk",
            ],
            # base_joint="ur5e_1_base_joint",
            ee_site_name="ur5e_1_suction_ee",
            grasp_actuator="ur5e_1_adhere_gripper",
            weld_body_name="ur5e_1_suction",
        )
        res = copy.deepcopy(UR5E_SUCTION_CONSTANTS)
        for k, v in res.items():
            # replce the name prefix in all values
            if isinstance(v, str):
                res[k] = v.replace("ur5e_1_", robot_name + "_")
            elif isinstance(v, list):
                res[k] = [x.replace("ur5e_1_", robot_name + "_") for x in v]
            elif isinstance(v, dict):
                res[k] = {
                    x.replace("ur5e_1_", robot_name + "_"): y.replace(
                        "ur5e_1_", robot_name + "_"
                    )
                    for x, y in v.items()
                    if isinstance(x, str)
                }
        res["name"] = robot_name
        return res

    def find_all_robots(self, xml_file_path):
        if os.path.exists(xml_file_path):
            tree = etree.parse(xml_file_path)
            root = tree.getroot()
        else:
            tree = etree.fromstring(xml_file_path)
            root = tree
        robot_elements = root.xpath("//@name[starts-with(., 'robot_')]")
        pattern = re.compile(r"robot_\d+$")
        robot_elements = [x for x in robot_elements if pattern.match(x)]
        # print("Found", robot_elements)
        return robot_elements

    def find_all_objects(self, xml_file_path):
        if os.path.exists(xml_file_path):
            tree = etree.parse(xml_file_path)
            root = tree.getroot()
        else:
            tree = etree.fromstring(xml_file_path)
            root = tree
        object_elements = root.xpath("//@name[starts-with(., 'box_')]")
        # object_elements = [x for x in object_elements if x.endswith("_")]
        pattern = re.compile(r"box_\d+$")
        object_elements = [x for x in object_elements if pattern.match(x)]
        # print("Found", object_elements)
        return object_elements

    def set_targets(self, targets):
        self.target_pos = targets

    @classmethod
    def load(cls, filepath, **kwargs):
        # TODO: implement this
        pass

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        # if "ur5e_" in robot_name:
        if "robot_" in robot_name:
            return dict(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(-0.5, 1))
        else:
            raise NotImplementedError(f"Unsupported robot name {robot_name}")

    def check_reach_range(
        self, robot_name, points: Union[Tuple[float, float, float], np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """Vectorized version of check_reach_range that can handle single points or arrays of points"""
        reach_range = self.get_robot_reach_range(robot_name)

        if isinstance(points, tuple):
            points = np.array(points)

        states = self.get_agent_state(self.agent_configs[robot_name], {})
        points = points - states.base_xpos
        # points = points - self.physics.model.body(robot_name).xpos[:3]
        if points.ndim == 1:
            # Single point
            for i, axis in enumerate(["x", "y", "z"]):
                if points[i] < reach_range[axis][0] or points[i] > reach_range[axis][1]:
                    return False
            return True
        else:
            # Array of points
            results = np.ones(len(points), dtype=bool)
            for i, axis in enumerate(["x", "y", "z"]):
                results &= (points[:, i] >= reach_range[axis][0]) & (
                    points[:, i] <= reach_range[axis][1]
                )
            return results

    def sample_initial_scene(self):
        # Sample a list of positions for each box object,
        pass

    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, (
                f"Robot {name} is not in the observation"
            )
        return obs

    def describe_robot_state(self, obs: EnvState, robot_name: str):
        agent_name = self.get_agent_name(robot_name)
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos

        robot_desp = ""
        if len(robot_state.contacts) == 0:
            obj = "empty"
        else:
            obj = "holding " + ",".join([c for c in robot_state.contacts])
        robot_desp += f"{agent_name}'s gripper is {obj},"

        # Vectorized reachability check for all objects
        object_names = np.array(self.object_names)
        object_states = np.array([obs.objects[name] for name in object_names])
        top_sites = np.array(
            [
                state.sites[f"{name}_top"]
                for name, state in zip(object_names, object_states)
            ]
        )
        top_positions = np.array([site.xpos for site in top_sites])

        # Check reachability for all objects at once
        reachable_mask = np.array(
            [self.check_reach_range(robot_name, pos) for pos in top_positions]
        )
        reachables = object_names[reachable_mask]
        not_reachables = object_names[~reachable_mask]

        if len(reachables) > 0:
            robot_desp += " can reach cubes: "
            robot_desp += ", ".join(reachables) + ", "
        if len(not_reachables) > 0:
            robot_desp += "can't reach cubes: "
            robot_desp += ", ".join(not_reachables) + ", "

        return robot_desp

    @lru_cache(maxsize=1)
    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        ret = []
        cube_ids = [self.physics.model.body(cube).id for cube in self.object_names]

        table_id = self.physics.model.body("table").id
        # bin_ids = [
        #     self.physics.model.body(bin_name).id for bin_name in SORTING_BIN_NAMES
        # ]

        for robot_name in self.robot_names:
            for link_id in self.robots[robot_name].all_link_body_ids:
                # if not "shoulder_lnik" in link_id:
                for cube_id in cube_ids:
                    ret.append((link_id, cube_id))

                # for bin_id in bin_ids:
                #     ret.append((link_id, bin_id))
                ret.append((link_id, table_id))

        for cube_id in cube_ids:
            ret.append((cube_id, table_id))
            # for cube_id2 in cube_ids:
            # if cube_id != cube_id2:
            # ret.append((cube_id, cube_id2))
            # for bin_id in bin_ids:
            #     ret.append((cube_id, bin_id))

        return ret

    def get_cube_panel(self, obs, cube_name: str):
        cube_state = obs.objects[cube_name]
        dist_to_panels = [
            (name, np.linalg.norm(cube_state.xpos[:2] - pos[:2]))
            for name, pos in self.panel_coords.items()
        ]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0]
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in obs.objects[cube_name].contacts:
                closest_panel = pname
                break
        return closest_panel

    def describe_cube_state(self, obs: EnvState, cube_name: str):
        cube_state = obs.objects[cube_name]
        top_site = cube_state.sites[f"{cube_name}_top"]
        x, y, z = top_site.xpos
        return f"{cube_name} is at ({x:.2f} {y:.2f})"

    def describe_obs(self, obs: EnvState):
        """For each cube, just describe whether it's on a bin, or between which two bins, no output numerical coordinates"""
        object_desp = "[Scene description]\n"
        for cube_name in self.robot_names:
            object_desp += self.describe_cube_state(obs, cube_name) + "\n"

        robot_desp = ""
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name) + "\n"
        robot_desp = robot_desp[:-2] + ".\n"
        full_desp = object_desp + robot_desp
        return full_desp

    def get_reward_done(self, obs):
        for obj_name in self.object_names:
            if obj_name not in obs.objects:
                continue
            block_state = obs.objects[obj_name]
            if np.allclose(block_state.xpos[:2], self.target_pos, atol=0.1):
                reward = 1
                done = True
                return reward, done
        return -1, False

    def get_grasp_site(self, obj_name: str = "pink_polygon") -> str:
        return f"{obj_name}_top"

    def get_target_pos(
        self,
    ) -> Optional[np.ndarray]:
        """useful for parsing place targets"""
        if isinstance(self.target_pos, list):
            return {
                box_name: self.target_pos[i]
                for i, box_name in enumerate(self.object_names)
            }
        else:
            return {
                box_name: self.target_pos[box_name]
                for i, box_name in enumerate(self.object_names)
            }

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix!
        for robname in self.robot_names:
            contacts[robname] = [
                c for c in contacts.get(robname, {}) if c in self.object_names
            ]
        return contacts

    def get_task_feedback(self, llm_plan, pose_dict):
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            # if ("PICK" in action_str and "PLACE" not in action_str) or (
            #     "PLACE" in action_str and "PICK" not in action_str
            # ):
            #     feedback += f"{agent_name}'s ACTION must contain both PICK and PLACE"
            if not ("Move" in action_str or "WAIT" in action_str):
                feedback += f"{agent_name}'s ACTION must contain Move"
            # if "PICK" in action_str and "PLACE" in action_str:
            #     obj = action_str.split("PICK")[1].split("PLACE")[0].strip()
            #     target = action_str.split("PLACE")[1].strip()
            #     if obj in self.cube_names and target in self.cube_to_bin.values():
            #         correct_panel = self.cube_to_bin[obj]
            #         if correct_panel not in target:
            #             valid_panels = ", ".join([correct_panel, "panel3", "panel5"])
            #             feedback += f"{agent_name}'s ACTION is not valid, {obj} cube can only be placed on {valid_panels}, but not on {target}"
        if all(["WAIT" in action_str for action_str in llm_plan.action_strs.values()]):
            feedback += "You can't all WAIT. The task is not complete, at least one robot should be acting."
        return feedback

    def get_object_joint_name(self, obj_name):
        return f"{obj_name}_joint"

    def get_waypoint_feedback(
        self,
        waypoint_paths: Dict[str, List],
        display=False,
        save_img=False,
        img_path="test.jpg",
    ):
        """
        Give feedback to the robots about the waypoints they are going to visit.
        """
        bad_waypoints = defaultdict(list)
        for robot_name, path in waypoint_paths.items():
            for waypoint in path:
                if not self.check_reach_range(robot_name, waypoint):
                    bad_waypoints[robot_name].append(waypoint)
        summ = ""
        for name, waypoints in bad_waypoints.items():
            summ += f"{name}: {waypoints} \n"
        if display:
            print(summ)
            self.render_point_cloud = True
            obs = self.get_obs()
            path_ls = list(waypoint_paths.values())
            visualize_voxel_scene(
                obs.scene,
                path_pts=path_ls,
                path_colors=[],
                save_img=save_img,
                img_path=img_path,
            )
        if summ == "":
            summ = "Reachability feedback: sucess."
        else:
            summ = (
                "Reachability feedback: failed. These steps are beyond the robot's reach: \n"
                + summ
            )
        return summ

    @property
    def use_preplace(self):
        return True

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]

    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]

    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs

    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots


if __name__ == "__main__":
    # ! MUJOCO 3.2.0 visualization
    with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdirname = "/Users/jiyi/Documents/2025-Projects/Mac-MultiRobotR1/3rdparty/robo3d/"
        tmpdirname = os.path.abspath(os.path.dirname(__file__))
        old_pwd = os.getcwd()
        import mujoco.viewer

        envres = generate_env(
            # 4, 4, 8, "full", tmpdirname, grid_size=0.55, return_xml_path=True
            4,
            4,
            8,
            "full",
            tmpdirname,
            grid_size=0.55,
            return_xml_path=False,
        )
        env_path = envres["xml"]
        print(envres["targets"])

        os.chdir(tmpdirname)
        print(os.listdir(tmpdirname))
        print(env_path)

        model = mujoco.MjModel.from_xml_path(env_path)
        data = mujoco.MjData(model)
        import ipdb

        ipdb.set_trace()
        kf_id = 0
        mujoco.mj_resetDataKeyframe(model, data, kf_id)
        mujoco.viewer.launch(model, data)
        os.chdir(old_pwd)
