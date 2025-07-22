import re
import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Optional

from .genenv_utils import BoxTask
from .rrt_multi_arm import MultiArmRRT
from .prompting import CustomLLMResponseParser
from .prompting.feedback import FeedbackManager
from .policy import PlannedPathPolicy


class Box3DEnv:
    def __init__(
        self,
        name="box3denv",
        grid_n=3,
        grid_m=3,
        num_objects=1,
        robot_mode="full",
        do_render=False,
        fast=False,
        obj_coords=None,
        obj_targets=None,
        use_json_format=True,
        **kwargs,
    ):
        self.name = name
        self.grid_n = grid_n
        self.grid_m = grid_m
        self.num_obj = num_objects
        self.robot_mode = robot_mode
        self.do_render = do_render

        self.env = BoxTask(
            filepath="",
            grid_n=self.grid_n,
            grid_m=self.grid_m,
            num_obj=self.num_obj,
            robot_mode=self.robot_mode,
            task_objects=[],
            render_freq=1200,
            image_hw=(400, 400),
            sim_forward_steps=300,
            error_freq=30,
            error_threshold=1e-5,
            randomize_init=True,
            render_point_cloud=0,
            render_cameras=[
                "top_cam",
                # "face_ur5e",
                # "teaser",
                "debug_cam",
            ],
            one_obj_each=True,
            do_render=do_render,
            obj_coords=obj_coords,
            obj_targets=obj_targets,
        )
        self.physics = self.env.physics
        self.initial_states = self.env.physics.get_state()
        self.targets = self.env.get_target_pos()
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=self.env.get_sim_robots(),
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        # self.response_keywords = ["NAME", "ACTION"]
        self.response_keywords = ["Move"]
        self.direct_waypoints = 3  # BY DEFAULT
        self.max_failed_waypoints = 1
        self.parser = CustomLLMResponseParser(
            self.env,
            "action",
            self.env.robot_name_map,
            self.response_keywords,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace,
            split_parsed_plans=False,
            json_format=use_json_format,
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
            control_freq=15,
            use_weld=1,
            skip_direct_path=0,
            skip_smooth_path=0,
            check_relative_pose=False,
        )

    def hash(self):
        tmp = self.to_json()
        tmp = json.dumps(tmp)
        return hash(tmp)

    def to_json(self):
        # get robot pos
        # we save the mujoco data
        state_dict = {
            "physics_state": self.initial_states.tolist(),
            "targets": self.targets,
            "grid_n": self.grid_n,
            "grid_m": self.grid_m,
            "name": self.name,
            "num_objects": self.num_obj,
            "robot_mode": self.robot_mode,
            "obj_coords": self.env.obj_coords,
            "obj_targets": self.env.target_pos,
        }
        if isinstance(state_dict["obj_targets"], dict):
            state_dict["obj_targets"] = {
                k: v for k, v in state_dict["obj_targets"].items() if v is not None
            }
        if isinstance(state_dict["targets"], dict):
            state_dict["targets"] = {
                k: v for k, v in state_dict["targets"].items() if v is not None
            }
        return state_dict

    def get_current_state(self):
        # we save the mujoco data
        state_dict = {
            "physics_state": self.env.physics.get_state(),
            "targets": self.targets,
            "grid_n": self.grid_n,
            "grid_m": self.grid_m,
            "name": self.name,
            "num_objects": self.num_obj,
            "robot_mode": self.robot_mode,
            "obj_coords": self.env.obj_coords,
            "obj_targets": self.env.target_pos,
        }
        if isinstance(state_dict["obj_targets"], dict):
            state_dict["obj_targets"] = {
                k: v for k, v in state_dict["obj_targets"].items() if v is not None
            }
        if isinstance(state_dict["targets"], dict):
            state_dict["targets"] = {
                k: v for k, v in state_dict["targets"].items() if v is not None
            }
        return state_dict

    @classmethod
    def load(cls, dict: Dict[str, Any], **kwargs):
        newobj = cls(
            name=dict["name"],
            grid_n=dict["grid_n"],
            grid_m=dict["grid_m"],
            num_objects=dict["num_objects"],
            robot_mode=dict["robot_mode"],
            obj_coords=dict["obj_coords"],
            obj_targets=dict["obj_targets"],
            **kwargs,
        )
        newobj.initial_states = np.array(dict["physics_state"])
        newobj.env.physics.set_state(newobj.initial_states)
        newobj.env.set_targets(dict["targets"])
        newobj.reset(states=dict["physics_state"])
        return newobj

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @property
    def objects(self):
        obs = self.env.get_obs()
        return obs.objects

    @property
    def robots(self):
        obs = self.env.get_obs()
        return {
            robot_name: getattr(obs, robot_name)
            for robot_name in self.env.robot_name_map.keys()
        }

    def describe_robots(self):
        obs = self.env.get_obs().to_json()
        return {
            robot_name: {
                "base": obs[robot_name]["base_xpos"][:2],
                "ee": obs[robot_name]["ee_xpos"][:2],
            }
            for robot_name in self.env.robot_name_map.keys()
        }

    @property
    def obs(self):
        return self.env.get_obs()

    def get_box_pos(self, box_name):
        # return self.physics.model.body(f"{box_name}").pos
        obs = self.env.get_obs()
        return obs.objects[box_name].xpos

    def get_arm_pos(self, robot_name):
        obs = self.env.get_obs()
        return getattr(obs, robot_name).ee_xpos

    def get_base_pos(self, robot_name: str) -> np.ndarray:
        obs = self.env.get_obs()
        return getattr(obs, robot_name).base_xpos

    def get_states(self):
        return self.physics.get_state()

    def reset(self, states=None):
        if states is None:
            states = self.initial_states
        if isinstance(states, list):
            states = np.array(states)
        self.env.reset(states=states)

    def reset_from_json(self, jsonobj):
        self.reset(states=jsonobj["physics_state"])

    def check_final(self):
        box_positions = {
            boxname: np.array(self.get_box_pos(boxname))
            for boxname in self.env.object_names
        }
        target_positions = {
            k: np.array(list(x)) for k, x in self.env.get_target_pos().items()
        }
        assert set(box_positions.keys()) == set(target_positions.keys())
        # print(box_positions)
        # print(target_positions)
        return all(
            np.allclose(box_positions[x][:2], target_positions[x][:2], atol=0.13)
            for x in box_positions.keys()
        )

    def simulate_one_step(self, response, step_idx=None):
        if "json" in response:
            try:
                response = convert_str_to_json(response)
                response = match_robot_name(response)
                response = "\n".join([k + ": " + v for k, v in response.items()])
            except Exception as e:
                return {
                    "success": False,
                    "detail": "ParseError: " + str(e),
                    "traj_len": -1,
                    "frame_count": -1,
                }
        else:
            response = response

        result_summary = {
            "success": False,
            "traj_len": -1,
            "detail": "UnknownError",
            "frame_count": -1,
        }
        # This is one step
        obs = self.env.get_obs()

        origin_sim_data = self.env.save_intermediate_state()

        parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response)
        if not parse_succ:
            # print("Parse failed")
            # execute_str = "EXECUTE" + response.split("EXECUTE")[-1]
            # curr_feedback = "Parse failed"
            ready_to_execute = False

            result_summary["success"] = False
            if "InvalidAction" in parsed_str:
                result_summary["detail"] = parsed_str
            else:
                result_summary["detail"] = "ParseError: " + parsed_str
        else:
            # print("Parse ok")
            ready_to_execute = True
            for j, llm_plan in enumerate(llm_plans):
                ready_to_execute, env_feedback = self.feedback_manager.give_feedback(
                    llm_plan
                )
                if not ready_to_execute:  # Potentially conflict
                    curr_feedback = env_feedback
                    result_summary["success"] = False
                    if "Reachability failed" in curr_feedback:
                        result_summary["detail"] = "InvalidAction: " + env_feedback
                    elif "IK failed" in curr_feedback:
                        result_summary["detail"] = "InvalidAction: " + env_feedback
                    elif "Collision" in curr_feedback:
                        result_summary["detail"] = "CollisionRobot: " + env_feedback

                    # import ipdb
                    # ipdb.set_trace()
                    print("Already failed", result_summary)
                    print(env_feedback)

        if (not ready_to_execute) and (not os.environ.get("DEBUG_SIM", False)):
            print("Failed run", result_summary)
            return result_summary

        rewind_env = False
        robots = self.env.get_sim_robots()
        # This point we can really execute
        # if step_idx == 1:
        #     import ipdb

        #     ipdb.set_trace()
        for i, plan in enumerate(llm_plans):
            # print("tograsp:", plan.tograsp, "inhand:", plan.inhand, plan.action_strs)
            try:
                policy = PlannedPathPolicy(
                    physics=self.env.physics,
                    robots=robots,
                    path_plan=plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=False,
                    **self.policy_kwargs,
                )
            except Exception as e:
                print("Policy init failed", e)
                result_summary["success"] = False
                result_summary["detail"] = "InvalidAction: " + str(e)
                rewind_env = True
                # print(plan.action_strs)
                break

            num_sim_steps = 0
            plan_success, reason = policy.plan(self.env)
            # print(f"Plan success: {plan_success}, reason: {reason}")
            if plan_success or (os.environ.get("DEBUG_SIM", False)):
                # print(f"Execute the plan for {len(policy.action_buffer)} steps")
                while not policy.plan_exhausted:
                    sim_action = policy.act(obs, self.env.physics)
                    obs, reward, done, info = self.env.step(sim_action, verbose=False)
                    num_sim_steps += 1

            if num_sim_steps > 0:
                if self.do_render:
                    if step_idx is not None:
                        vid_name = f"debug/{step_idx:02}.mp4"
                    else:
                        vid_name = "debug/execute.mp4"
                    os.makedirs("debug", exist_ok=True)
                    self.env.export_render_to_video(vid_name, out_type="mp4", fps=30)

                result_summary["success"] = True
                result_summary["frame_count"] = num_sim_steps

                already_done = self.check_final()
                if already_done:
                    result_summary["detail"] = "Success"
                else:
                    result_summary["detail"] = "StepSuccess"
            else:
                print("Plan execute failed")
                result_summary["success"] = False
                result_summary["detail"] = "PlanExecuteFailed: " + reason
                rewind_env = True
                break

            if done:
                break

        if rewind_env:
            self.env.load_saved_state(origin_sim_data)
        else:
            # update sim
            origin_sim_data = self.env.save_intermediate_state()
        return result_summary

    def simulate_full_step(self, response):
        # step_responses = json.loads(response)  # suppose it is a json string
        # import ipdb

        # ipdb.set_trace()
        if "json" in response:
            try:
                step_responses = convert_str_to_json(response)
                step_responses = [match_robot_name(x) for x in step_responses]
                step_responses = [
                    "\n".join([k + ": " + v for k, v in x.items()])
                    for x in step_responses
                ]
            except Exception as e:
                return {
                    "success": False,
                    "detail": "ParseError: " + str(e),
                    "traj_len": -1,
                    "frame_count": -1,
                }
        else:
            step_responses = response

        cur_result_summary = {
            "success": False,
            "traj_len": -1,
            "detail": "UnknownError",
            "frame_count": -1,
        }
        frameall = 0
        for step, step_response in enumerate(step_responses):
            cur_result_summary = self.simulate_one_step(step_response, step_idx=step)
            # print(cur_result_summary)
            frameall += cur_result_summary.get("frame_count", 0)
            cur_result_summary["traj_len"] = step + 1
            if not cur_result_summary["success"]:
                return cur_result_summary

        if cur_result_summary["detail"] == "Success":
            cur_result_summary["success"] = True
            cur_result_summary["traj_len"] = len(step_responses)
            cur_result_summary["frame_count"] = frameall
            cur_result_summary["detail"] = "Success"
        else:
            cur_result_summary["success"] = False
        return cur_result_summary

    def simulate_one_step_from_str(self, solution_str, step_idx=None):
        return self.simulate_one_step(solution_str)

    def simulate_all_str(self, solution_str, return_step_action=True):
        return self.simulate_full_step(solution_str)

    @classmethod
    def describe_obs(cls, obs):
        # obs:
        # {
        #   "objects": {} # a dict of objects
        #   "robot_0": {
        #       "base_xpos": [0, 0, 0],
        #      "ee_xpos": [0, 0, 0],}
        # }
        #   "robot_1": {
        #       "base_xpos": [0, 0, 0],
        #      "ee_xpos": [0, 0, 0],}
        # }
        # }

        if not isinstance(obs, dict):
            obs = obs.to_json()


def convert_str_to_json(string):
    pattern = re.compile("```json\n(.*?)```", re.DOTALL)
    match = pattern.findall(string.split("</think>")[-1])
    # match = pattern.findall(string)
    if match:
        return json.loads(match[-1])
    else:
        return None


def match_robot_name(json_obj):
    return {
        k.replace("Robot ", "robot_"): v for k, v in json_obj.items() if "Robot" in k
    }
