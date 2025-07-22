import re
import json
import copy
import random
import numpy as np
from enum import Enum
from itertools import product
from typing import List, Tuple, Dict, Union, Any

from .simenv import SimulationEnv
from .box_utils import line_intersection


class StatusEnum(Enum):
    Empty = 0
    ObjTaken = 1
    RobTaken = 2

    def __repr__(self):
        return self.name


class ExecutionRes(Enum):
    Success = 0
    InvalidAction = 1
    CollisionRobot = 2
    CollisionObject = 3
    Fail = 4

    def __repr__(self):
        return self.name


class Status:
    def __init__(self, status: StatusEnum, objid: int, detail: str):
        self.status = status
        self.objid = objid
        self.detail = detail

    def __repr__(self):
        return f"Status({self.status}, {self.detail})"


class RobotPos:
    def __init__(self, base_pos: Tuple[float, float], arm_pos: Tuple[float, float]):
        self.base_pos = base_pos
        self.arm_pos = arm_pos

    def __repr__(self):
        return f"RobotPos({self.base_pos}, {self.arm_pos})"

    def to_json(self):
        return {
            "base_pos": self.base_pos,
            "arm_pos": self.arm_pos,
        }

    def to_tuple(self):
        return (self.base_pos, self.arm_pos)

    @classmethod
    def load(cls, data: Dict[str, Tuple[float, float]]):
        return cls(data["base_pos"], data["arm_pos"])


class Action:
    def __init__(
        self,
        robot_id,
        arm_pos: Tuple[float, float],  # the position of arm
        base_pos: Tuple[float, float],  # the position of arm's base
        pos_s: Tuple[float, float],  # the position of starting position
        pos_e: Tuple[float, float],  # the position of ending position
        carry: bool = True,
    ):
        self.robot_id = robot_id
        self.arm_pos = arm_pos
        self.base_pos = base_pos
        self.pos_s = pos_s
        self.pos_e = pos_e
        self.carry = carry

    def __repr__(self):
        return f"Action(\nrobot_id: {self.robot_id}\n\trobot_arm: {self.arm_pos},\n\trobot_base: {self.base_pos},\n\tfrom: {self.pos_s},\n\tto: {self.pos_e},\n\tcarry: {self.carry}\n)"

    @classmethod
    def to_str(cls, actions: List["Action"]) -> str:
        res = {}
        for action in actions:
            k = f"Robot {action.robot_id}"
            v = f"{action.pos_s} -> {action.pos_e}, {action.carry}"
            res[k] = v
        return json.dumps(res)

    @classmethod
    def from_list(cls, plan_obj, all_bot_pos: Dict[int, "Action"]) -> List["Action"]:
        traj_actions = []

        def parse_action(obj):
            return Action(
                robot_id=obj["robot_id"],
                arm_pos=obj["arm_pos"],
                base_pos=obj["base_pos"],
                pos_s=obj["pos_s"],
                pos_e=obj["pos_e"],
                carry=obj["carry"],
            )

        def get_step_action(obj: List):
            if isinstance(obj, List):
                return [parse_action(x) for x in obj]
            elif isinstance(obj, dict):
                return [parse_action(obj)]

        if isinstance(plan_obj, list):
            for step_action in plan_obj:
                this_step_actions = get_step_action(step_action)
                traj_actions.append(this_step_actions)
            return traj_actions
        elif isinstance(plan_obj, dict):
            # Only one-step action
            this_step_actions = get_step_action(plan_obj)
            return this_step_actions
        else:
            raise ValueError("Invalid json object")

    @classmethod
    def from_str(cls, str, all_bot_pos: Dict[int, "Action"]) -> List["Action"]:
        def extract_json(str):
            pattern = re.compile("```json\n(.*?)```", re.DOTALL)
            # match = pattern.search(str)
            match = pattern.findall(str.split("</think>")[-1])
            if match:
                return json.loads(match[-1])
            else:
                return None

        def get_robot_id(string):
            try:
                robot_id = int(re.search(r"\d+", string).group())
                return robot_id
            except Exception as _:
                return None

        def get_arm_pos(robot_id, all_bot_pos):
            if robot_id in all_bot_pos:
                return all_bot_pos[robot_id].arm_pos
            else:
                return None

        def get_base_pos(robot_id, all_bot_pos):
            if robot_id in all_bot_pos:
                return all_bot_pos[robot_id].base_pos
            else:
                return None

        def parse_action(string):
            try:
                carry_str = string.split(",")[-1]
                left = string.split("->")[0].strip()
                left = eval(left)
                right = string[: -len(carry_str)].split("->")[-1].rstrip(",")
                right = eval(right)
                carry = eval(carry_str.strip())
                return (left, right, carry)
            except Exception as _:
                return None

        def get_step_action(step_action):
            this_step_actions = []
            for k, v in step_action.items():
                robot_id = get_robot_id(k)
                if robot_id is None:
                    raise ValueError(f"Cannot find robot id in {k}")
                # robot_pos = get_arm_pos(robot_id, all_bot_pos)
                base_pos = get_base_pos(robot_id, all_bot_pos)
                movement = parse_action(v)
                if movement is None:
                    raise ValueError(f"Cannot parse action {v}")
                this_step_actions.append(
                    cls(
                        robot_id=robot_id,
                        arm_pos=all_bot_pos[robot_id].arm_pos,
                        base_pos=base_pos,
                        pos_s=movement[0],
                        pos_e=movement[1],
                        carry=movement[2],
                    )
                )
            return this_step_actions

        json_obj = extract_json(str)
        if json_obj is None:
            return None
        else:
            traj_actions = []

            if isinstance(json_obj, list):
                for step_action in json_obj:
                    this_step_actions = get_step_action(step_action)
                    traj_actions.append(this_step_actions)
                return traj_actions
            elif isinstance(json_obj, dict):
                # Only one-step action
                this_step_actions = get_step_action(json_obj)
                return this_step_actions
            else:
                raise ValueError("Invalid json object")

    def hash(self):
        return hash(self.__repr__())


class ExecutionResult:
    def __init__(self, sucess: ExecutionRes, detail=""):
        self.success = sucess
        self.detail = detail

    def __repr__(self):
        return f"ExecutionResult({self.success}, {self.detail})"


class Box1Env(SimulationEnv):
    def __init__(
        self,
        name="Box1Env",
        grid_n=2,
        grid_m=5,
        num_objects=4,
        robot_mode="full",
        robot_as="point",  # point for point robot, arm for robot with arm (base + arm position as a robot)
        movement="straight",  # straight or curve, if curve, then the collission considers the curve
        robot_speed=1.0,
        pos_mode="full",
    ):
        super().__init__(name)

        self.map = {}
        self.objects = {}
        self.targets = {}
        self.robots = {}
        self.grid_n = grid_n
        self.grid_m = grid_m
        self.num_objects = num_objects
        self.robot_mode = robot_mode
        self.robot_as = robot_as
        self.pos_mode = pos_mode
        self.movement = movement
        self.robot_speed = robot_speed
        self.global_timing = 0.0

    def filter_map(self):
        robot_positions = [robot.base_pos for robot in self.robots.values()]
        all_coords = list(self.map.keys())
        # filter by robot position, find all positions that can be reached by one of the robot
        all_rechable_coords = []
        for base_pos in robot_positions:
            reachable_coords = self.get_arm_reachable_positions(base_pos)
            all_rechable_coords.extend(reachable_coords)
        all_rechable_coords = set(all_rechable_coords)
        all_coords = [x for x in all_coords if x in all_rechable_coords]
        return all_coords

    def create_map(self):
        x_list = np.arange(0.25, self.grid_n, 0.5).tolist()
        y_list = np.arange(0.25, self.grid_m, 0.5).tolist()
        coords = list(product(x_list, y_list))
        for coord in coords:
            (x, y) = coord
            self.map[(x, y)] = Status(StatusEnum.Empty, -1, "None")

    def create_object(self, num_objects=None):
        if num_objects is not None:
            num_objects = self.num_objects
        all_coords = list(self.map.keys())
        random.shuffle(all_coords)
        object_coords = all_coords[:num_objects]
        for i in range(self.num_objects):
            (x, y) = object_coords[i]
            self.objects[i] = [x, y]
            self.map[(x, y)] = Status(StatusEnum.ObjTaken, i, f"Object {i}")

    def create_target_position(self):
        # Gather positions that are still empty (cannot be object's original positions)
        available_positions = [
            pos for pos, stat in self.map.items() if stat.status == StatusEnum.Empty
        ]
        if len(available_positions) < len(self.objects):
            raise ValueError("Not enough available positions for target assignment.")
        random.shuffle(available_positions)
        for obj_id in self.objects:
            target_coord = available_positions.pop()
            self.targets[obj_id] = list(target_coord)
            self.map[target_coord] = Status(
                StatusEnum.ObjTaken, obj_id, f"Target {obj_id}"
            )

    def create_robot(self):
        if self.robot_mode.startswith("full"):
            robot_id = 0
            coords = product(
                np.arange(0.0, self.grid_n, 1.0).tolist(),
                np.arange(1.0, self.grid_m, 1.0).tolist(),
            )
        elif self.robot_mode.startswith("minimal"):
            # in this mode, no overlap coverage between any robot
            robot_id = 0
            x_list = np.arange(1.0, self.grid_n, 2.0).tolist() + (
                [] if self.grid_n % 2 == 0 else [self.grid_n - 1]
            )
            y_list = np.arange(1.0, self.grid_m, 2.0).tolist() + (
                [] if self.grid_m % 2 == 0 else [self.grid_m - 1]
            )
            coords = list(
                product(
                    x_list,
                    y_list,
                )
            )
            if self.grid_n >= 4 or self.grid_m >= 4:
                x_list = np.arange(2.0, self.grid_n + 1, 2.0).tolist()
                y_list = np.arange(2.0, self.grid_m + 1, 2.0).tolist()
                coords.extend(list(product(x_list, y_list)))
        elif self.robot_mode.startswith("triu"):
            robot_id = 0
            coords = product(
                np.arange(0.0, self.grid_n, 1.0).tolist(),
                np.arange(1.0, self.grid_m, 1.0).tolist(),
            )
            coords = [(x, y) for x, y in coords if x <= y]
        elif self.robot_mode.startswith("randrobot"):
            robot_id = 0
            base_coords = list(
                product(
                    np.arange(0.0, self.grid_n, 1.0).tolist(),
                    np.arange(1.0, self.grid_m, 1.0).tolist(),
                )
            )
            coords = []
            while len(coords) < len(base_coords) // 2:
                can_reachable_base = []
                for base_coord in coords:
                    connected_coords = self.get_connected_base_positions(base_coord)
                    can_reachable_base.extend(
                        [x for x in connected_coords if x in base_coords]
                    )
                can_reachable_base = list(set(can_reachable_base))

                if len(can_reachable_base) == 0:
                    choices = base_coords
                else:
                    choices = can_reachable_base
                rand_coords = random.choice(choices)
                while rand_coords in coords:
                    rand_coords = random.choice(choices)
                coords.append(rand_coords)

        if "randpos" not in self.robot_mode:
            for coord in coords:
                x, y = coord
                if x + 0.25 > self.grid_n or y + 0.25 > self.grid_m:
                    self.robots[robot_id] = RobotPos([x, y], [x - 0.25, y - 0.25])
                else:
                    self.robots[robot_id] = RobotPos([x, y], [x + 0.25, y + 0.25])
                robot_id += 1
        else:
            seen_coords = set()
            for robot_id in range(len(coords)):
                reachable_pos = self.get_arm_reachable_positions(coords[robot_id])
                x, y = coords[robot_id]
                # randomly chooise a position, if there is no overlap then break
                while True:
                    new_pos = random.choice(reachable_pos)
                    if new_pos not in seen_coords:
                        seen_coords.add(new_pos)
                        break
                self.robots[robot_id] = RobotPos([x, y], new_pos)

    def generate_nooverlap(self, orientation="horizontal"):
        self.create_map()
        all_coords = list(self.map.keys())

        # create robot
        if orientation == "horizontal":
            rob_coords = list(
                product(
                    np.arange(1.0, self.grid_n, 1.0).tolist(),
                    np.arange(1.0, self.grid_m, 1.0).tolist(),
                )
            )
            one_rob_coords = [x for x in rob_coords if x[0] < self.grid_m // 2]
            two_rob_coords = [x for x in rob_coords if x[0] > self.grid_m // 2]
        else:
            rob_coords = list(
                product(
                    np.arange(1.0, self.grid_n, 1.0).tolist(),
                    np.arange(1.0, self.grid_m, 1.0).tolist(),
                )
            )
            one_rob_coords = [x for x in rob_coords if x[1] < self.grid_n // 2]
            two_rob_coords = [x for x in rob_coords if x[1] > self.grid_n // 2]

        robot_id = 0
        self.robots = {}
        for coords in one_rob_coords + two_rob_coords:
            x, y = coords
            if x + 0.25 > self.grid_n or y + 0.25 > self.grid_m:
                self.robots[robot_id] = RobotPos([x, y], [x - 0.25, y - 0.25])
            else:
                self.robots[robot_id] = RobotPos([x, y], [x + 0.25, y + 0.25])
            robot_id += 1

        # create object
        all_coords = list(self.map.keys())
        if orientation == "horizontal":
            one_obj_coords = [x for x in all_coords if x[0] < self.grid_n // 2]
            two_obj_coords = [x for x in all_coords if x[0] > self.grid_n // 2]
        else:
            one_obj_coords = [x for x in all_coords if x[1] < self.grid_m // 2]
            two_obj_coords = [x for x in all_coords if x[1] > self.grid_m // 2]

        random.shuffle(one_obj_coords)
        random.shuffle(two_obj_coords)

        object_coords = []
        tmp0 = one_obj_coords[self.num_objects // 2 :]
        tmp1 = two_obj_coords[self.num_objects // 2 :]
        random.shuffle(tmp0)
        random.shuffle(tmp1)
        one_obj_coords = one_obj_coords[: self.num_objects // 2]
        two_obj_coords = two_obj_coords[: self.num_objects // 2]
        object_coords = one_obj_coords + two_obj_coords
        object_targets = tmp0[: len(one_obj_coords)] + tmp1[: len(two_obj_coords)]
        # for i in range(self.num_objects):
        for i in range(len(object_coords)):
            (x, y) = object_coords[i]
            self.objects[i] = [x, y]
            self.map[(x, y)] = Status(StatusEnum.ObjTaken, i, f"Object {i}")
            (x, y) = object_targets[i]
            self.targets[i] = [x, y]

    def get_arm_reachable_positions(self, base_pos):
        # Right now we only consider 0.25, 0.75
        offsets = [-0.25, -0.75, 0.25, 0.75]
        res = [(base_pos[0] + dx, base_pos[1] + dy) for dx in offsets for dy in offsets]
        res = [x for x in res if 0 < x[0] < self.grid_n and 0 < x[1] < self.grid_m]
        res = [x for x in res if x[0] != base_pos[0] or x[1] != base_pos[1]]
        return res

    def get_connected_base_positions(self, base_pos):
        offsets = [0, 1, -1]
        res = [(base_pos[0] + dx, base_pos[1] + dy) for dx in offsets for dy in offsets]
        res = [x for x in res if 0 <= x[0] <= self.grid_n and 0 <= x[1] <= self.grid_m]
        res = [x for x in res if x[0] != base_pos[0] or x[1] != base_pos[1]]
        return res

    def create(self):
        self.create_map()
        self.create_robot()
        self.filter_map()

        self.create_object()
        self.create_target_position()
        self.global_timing = 0.0

        # import ipdb

        # ipdb.set_trace()
        # print("Done creating " + self.__repr__())

    def has_solution(self):
        for obj_id, obj_pos in self.objects.items():
            target_pos = self.targets[obj_id]
            ok = True
            for robot_id, robot_pos in self.robots.items():
                rechable_pos = self.get_arm_reachable_positions(robot_pos.base_pos)
                if target_pos in rechable_pos:
                    ok = False
                    break
            if not ok:
                return False
        return True

    def is_valid(self):
        # Check if the environment is valid

        # All objects in valid position and appear only once, and can be found in the map
        if len(self.objects) != self.num_objects:
            return False, "Number of objects is not correct"

        for obj_id, obj_pos in self.objects.items():
            if obj_id not in self.targets:
                return False, f"Object {obj_id} does not have target position"
            obj_pos = tuple(obj_pos)
            if obj_pos not in self.map:
                if (
                    obj_pos[0] < 0
                    or obj_pos[0] > self.grid_n
                    or obj_pos[1] < 0
                    or obj_pos[1] > self.grid_m
                ):
                    return False, f"Object {obj_id} is not in the map"
            else:
                if (
                    self.map[obj_pos].status != StatusEnum.ObjTaken
                    or self.map[obj_pos].objid != obj_id
                ):
                    return False, f"Object {obj_id} is not in the correct position"

        # All arms in valid position
        return True, "Valid Environment"

    def update(self, actions: List[Action]):
        # if the action is valid, then update, else return False
        if self.verify(actions):
            self.update_map(actions)
            self.update_objects(actions)
            self.update_robots(actions)
            return True
        else:
            return False

    def visualize(
        self,
        actions: List[Action] = None,
        exec_res: ExecutionResult = None,
        out_file_path=None,
    ):
        # Visualize the current state of the environment
        import pygame
        from simulation.vis_api import TileMap

        pygame.init()
        CELL_SIZE = 150
        PAD = 50
        SCREEN_WIDTH = self.grid_m * CELL_SIZE + 2 * PAD
        SCREEN_HEIGHT = self.grid_n * CELL_SIZE + 2 * PAD
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.display.set_caption("Box1 Visualization")

        tile_map = TileMap(CELL_SIZE)
        screen.fill((0, 0, 0, 0))
        tile_map.draw_map(screen, self.grid_n, self.grid_m)
        tile_map.draw_target_pos(screen, self.targets)
        tile_map.draw_object(screen, self.objects)
        tile_map.draw_robot(screen, self.robots)
        if actions is None:
            actions = []

        tmp_env = copy.deepcopy(self)
        tmp_env.simulate(actions)
        moved_robots = {
            k: v
            for k, v in tmp_env.robots.items()
            if v.arm_pos != self.robots[k].arm_pos
        }
        moved_robots_before = {
            k: v
            for k, v in self.robots.items()
            if v.arm_pos != tmp_env.robots[k].arm_pos
        }
        not_moved_robots = {
            k: v
            for k, v in tmp_env.robots.items()
            if v.arm_pos == self.robots[k].arm_pos
        }
        COLOR = (0, 100, 255, 50)
        tile_map.draw_arm(screen, moved_robots_before, color=COLOR, is_initial=False)
        tile_map.draw_arm(
            screen,
            not_moved_robots,
            color=COLOR,
            is_initial=True,
        )
        if actions:
            tile_map.draw_action(screen, actions)
            tile_map.draw_arm(screen, moved_robots)

        if exec_res:
            if exec_res.success != ExecutionRes.Success:
                # Highlight the failed action region(s) in red
                # If detail is not a tuple, wrap it in a tuple for iteration.
                failed_actions = (
                    exec_res.detail
                    if isinstance(exec_res.detail, tuple)
                    else (exec_res.detail,)
                )
                tile_map.draw_conflict(screen, failed_actions)

        pygame.display.flip()
        if out_file_path is not None:
            pygame.image.save(screen, out_file_path)

        pygame.quit()

    def get_execution_time(self, action):
        # Get the execution time for the action
        start_pos = action.pos_s
        end_pos = action.pos_e
        distance = np.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        )
        return distance / self.robot_speed

    def simulate(self, actions: List[Action]):
        old_env = copy.deepcopy(self)

        # Simulates the movement
        step_timing = 0.0
        for action in actions:
            if action.robot_id not in self.robots:
                return ExecutionResult(ExecutionRes.Fail, action)

            # Move robot arm
            self.robots[action.robot_id].arm_pos = action.pos_e

            # If there is an object in this position
            if (
                old_env.map.get(
                    tuple(action.pos_s), Status(StatusEnum.Empty, -1, "")
                ).status
                == StatusEnum.ObjTaken
                and action.carry
            ):
                obj_id = old_env.map[tuple(action.pos_s)].objid
                # Update the object
                self.objects[obj_id] = action.pos_e
                # Update map
                self.map[tuple(action.pos_e)] = Status(
                    StatusEnum.ObjTaken, obj_id, f"Object {obj_id}"
                )
                self.map[tuple(action.pos_s)] = Status(StatusEnum.Empty, -1, "None")

            action_timing = self.get_execution_time(action)
            if action_timing > step_timing:
                step_timing = action_timing
        self.global_timing += step_timing

    def verify(self, actions: List[Action]) -> ExecutionResult:
        for action in actions:
            if not is_valid(action):
                return ExecutionResult(ExecutionRes.InvalidAction, action)

        collision = has_object_collision(actions)
        if collision:
            return ExecutionResult(ExecutionRes.CollisionObject, collision)

        collision = has_object_collision_after_move(actions, self)
        if collision:
            return ExecutionResult(ExecutionRes.CollisionObject, collision)

        collision = has_robot_collision(actions)
        # collision =
        if collision:
            # There is a collision, return False
            return ExecutionResult(ExecutionRes.CollisionRobot, collision)

        return ExecutionResult(ExecutionRes.Success)

    def scrutiny(self, actions: List[Action]):
        # This function returns all invalid action and conflicts in the actions
        is_invalid = [is_valid(action, self) for action in actions]

        invalid_actions = [
            action for action, valid in zip(actions, is_invalid) if not valid
        ]
        actions = [action for action, valid in zip(actions, is_invalid) if valid]

        collision = has_object_collision(actions, self)
        if collision:
            invalid_actions.append(collision)

        collision = has_robot_collision(actions, self)
        if collision:
            invalid_actions.append(collision)

        if len(invalid_actions) > 0:
            return invalid_actions

    def check_final(self):
        def close_enough(a, b, tol=0.02):
            return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol

        # Check if all objects are in the target positions
        for obj_id in self.objects:
            # if self.objects[obj_id] != self.targets[obj_id]:
            if not close_enough(self.objects[obj_id], self.targets[obj_id]):
                return False
        return True

    def __repr__(self):
        return f"Box1Env({self.name}):\n\tGrid: {self.grid_n}x{self.grid_m}\n\tObjects: {self.num_objects}\n\tRobots: {self.robot_mode}"

    def to_json(self):
        return {
            "name": self.name,
            "grid_n": self.grid_n,
            "grid_m": self.grid_m,
            "num_objects": self.num_objects,
            "robot_mode": self.robot_mode,
            "robot_as": self.robot_as,
            "movement": self.movement,
            "objects": self.objects,
            "targets": self.targets,
            "pos_mode": self.pos_mode,
            "robots": {k: v.to_json() for k, v in self.robots.items()},
        }

    def get_current_state(self):
        return self.to_json()

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
            return cls.load(data)

    def reset_from_json(self, dict):
        self.grid_n = dict["grid_n"]
        self.grid_m = dict["grid_m"]
        self.num_objects = dict["num_objects"]
        self.robot_mode = dict["robot_mode"]
        self.robot_as = dict["robot_as"]
        self.movement = dict["movement"]
        self.pos_mode = dict.get("pos_mode", "full")
        self.objects = dict["objects"]
        self.targets = dict["targets"]
        self.robots = {int(k): RobotPos.load(v) for k, v in dict["robots"].items()}
        self.objects = {int(k): v for k, v in self.objects.items()}
        self.targets = {int(k): v for k, v in self.targets.items()}
        self.create_map()
        for k, v in self.objects.items():
            self.map[tuple(v)] = Status(StatusEnum.ObjTaken, k, f"Object {k}")
        self.global_timing = 0.0

    @classmethod
    def load(cls, dict: Dict[str, Any]):
        def round_pos(pos):
            return (round(pos[0], 2), round(pos[1], 2))

        env = cls(
            name=dict["name"],
            grid_n=dict["grid_n"],
            grid_m=dict["grid_m"],
            num_objects=dict["num_objects"],
            robot_mode=dict["robot_mode"],
            robot_as=dict["robot_as"],
            movement=dict["movement"],
            pos_mode=dict.get("pos_mode", "full"),
        )
        env.objects = dict["objects"]
        env.targets = dict["targets"]
        env.robots = {int(k): RobotPos.load(v) for k, v in dict["robots"].items()}
        env.objects = {int(k): round_pos(v) for k, v in env.objects.items()}
        env.targets = {int(k): round_pos(v) for k, v in env.targets.items()}
        env.create_map()

        for k, v in env.objects.items():
            env.map[tuple(v)] = Status(StatusEnum.ObjTaken, k, f"Object {k}")
        env.global_timing = 0.0
        return env

    def hash(self):
        tmp = self.to_json()
        tmp = json.dumps(tmp)
        return hash(tmp)

    def simulate_all_str(
        self, actions_str: str, return_step_action=False
    ) -> Dict[str, Any]:
        # ? This function try to simulate all actions in the string, used for RL training
        try:
            all_actions = Action.from_str(actions_str, self.robots)
        except Exception:
            return {
                "success": False,
                "detail": "ParseError",
                "traj_len": -1,
                "parallelism": -1,
            }

        if all_actions is None:
            return {
                "success": False,
                "detail": "ParseError",
                "traj_len": -1,
                "parallelism": -1,
            }

        # Simulate
        done = False
        detail = ""
        all_step_traj = []
        max_par = -1
        for step, step_action in enumerate(all_actions):
            step_traj = {}
            to_break = False
            # Update the arm_pos in action
            if step_action is not None:
                try:
                    # print(actions_str)
                    # print(step_action)
                    for action in step_action:
                        action.arm_pos = self.robots[action.robot_id].arm_pos
                    step_traj["actions"] = Action.to_str(step_action)
                    exec_res = self.verify(step_action)

                    if exec_res.success == ExecutionRes.Success:
                        step_traj["status"] = "Success"
                    else:
                        step_traj["status"] = "failed"
                        if exec_res.success == ExecutionRes.CollisionObject:
                            detail = "CollisionObject"
                        elif exec_res.success == ExecutionRes.CollisionRobot:
                            detail = "CollisionRobot"
                        elif exec_res.success == ExecutionRes.InvalidAction:
                            detail = "InvalidAction"
                        else:
                            detail = ""
                        to_break = True
                except Exception:
                    step_traj["status"] = "invalid"
                    detail = "StepParseError"
                    to_break = True
            else:
                step_traj["status"] = "invalid"
                detail = "StepParseError"
                to_break = True

            if not isinstance(step_action, list):
                step_action = [step_action]
            max_par = max(max_par, len(step_action))
            all_step_traj.append(step_traj)
            if to_break:
                break
            self.simulate(step_action)

        done = self.check_final()
        if done:
            result = {"success": True, "detail": "Success"}
        else:
            result = {"success": False, "detail": detail}
        if return_step_action:
            result["traj"] = all_step_traj
        result["traj_len"] = len(all_actions)
        result["parallelism"] = max_par
        return result

    def simulate_one_step_from_str(
        self, action_str: str, return_step_action=False
    ) -> Dict[str, Any]:
        # ? This function try to simulate one step actions in the string
        try:
            all_actions = Action.from_str(action_str, self.robots)
        except Exception:
            return {
                "success": False,
                "detail": "ParseError",
                "traj": [],
                "traj_len": -1,
                "parallelism": -1,
            }

        if all_actions is None:
            return {
                "success": False,
                "detail": "ParseError",
                "traj": [],
                "traj_len": -1,
                "parallelism": -1,
            }

        # Simulate
        done = False
        detail = ""
        all_step_traj = []
        step_traj = {}
        to_break = False
        # Update the arm_pos in action
        step_action = all_actions
        if step_action is not None:
            try:
                for action in step_action:
                    action.arm_pos = self.robots[action.robot_id].arm_pos
                step_traj["actions"] = Action.to_str(step_action)
                exec_res = self.verify(step_action)

                if exec_res.success == ExecutionRes.Success:
                    step_traj["status"] = "success"
                else:
                    step_traj["status"] = "failed"
                    if exec_res.success == ExecutionRes.CollisionObject:
                        detail = "CollisionObject"
                    elif exec_res.success == ExecutionRes.CollisionRobot:
                        detail = "CollisionRobot"
                    elif exec_res.success == ExecutionRes.InvalidAction:
                        detail = "InvalidAction"
                    else:
                        detail = ""
                    to_break = True
            except Exception:
                step_traj["status"] = "invalid"
                detail = "StepParseError"
                to_break = True
        else:
            step_traj["status"] = "invalid"
            detail = "StepParseError"
            to_break = True

        if not to_break:
            self.simulate(step_action)
            done = self.check_final()
            if done:
                result = {"success": True, "detail": "Success"}
            else:
                result = {"success": True, "detail": "StepSuccess"}
            result["parallelism"] = len(step_action)
        else:
            result = {"success": False, "detail": detail}
            result["parallelism"] = -1

        if return_step_action:
            result["traj"] = all_step_traj

        return result


def is_valid(action: Action, env: Box1Env = None):
    def check_robot(action: Action):
        if not action.arm_pos == action.pos_s:
            return False

        # robot cannot reach a position that is not on the grid
        robot_x_diff = -(action.base_pos[0] - action.pos_s[0])
        robot_y_diff = -(action.base_pos[1] - action.pos_s[1])
        if not (-1 < robot_x_diff < 1 and -1 < robot_y_diff < 1):
            return False

        robot_x_diff = -(action.base_pos[0] - action.pos_e[0])
        robot_y_diff = -(action.base_pos[1] - action.pos_e[1])
        if not (-1 < robot_x_diff < 1 and -1 < robot_y_diff < 1):
            return False

        if env is not None:
            if action.pos_s[0] > env.grid_n or action.pos_s[1] > env.grid_m:
                return False

        return True

    if not check_robot(action):
        return False

    return True


def has_object_collision(actions, env: Box1Env = None):
    # This function checks whether there is any collision between objects before or after action
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            if (
                (actions[i].pos_e == actions[j].pos_e)
                and actions[i].carry
                and actions[j].carry
            ):
                # No two objects are put at same position
                return (actions[i], actions[j])

            if (
                actions[i].pos_s == actions[j].pos_s
                and actions[i].carry
                and actions[j].carry
            ):
                # No two objects are put at same position
                return (actions[i], actions[j])
    return None


def has_object_collision_after_move(actions: List[Action], env: Box1Env = None):
    new_env = copy.deepcopy(env)
    new_env.simulate(actions)
    check_res = new_env.is_valid()
    if not check_res[0]:
        # Find the root cause
        pattern = re.compile(r"Object (\d+)")
        missing_id = int(pattern.search(check_res[1]).group(1))
        collision_actions = []
        for action in actions:
            if (action.carry and action.pos_s == env.objects[missing_id]) or (
                action.carry and action.pos_e == env.objects[missing_id]
            ):
                collision_actions.append(action)
        return collision_actions

    return None


def has_robot_collision(actions: List[Action], robots=None):
    # Check if there is any collision of robots between robots
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            # Check if two trajectory overlap

            # We map the position for arm to another system
            area_i = [
                actions[i].base_pos,
                actions[i].pos_s,
                actions[i].pos_e,
            ]
            area_j = [
                actions[j].base_pos,
                actions[j].pos_s,
                actions[j].pos_e,
            ]

            overlap = shapes_overlap_with_intersection(area_i, area_j)

            if overlap[0]:
                crossing_points = overlap[1]
                if len(crossing_points) > 1:
                    return (actions[i], actions[j])
                if not (
                    list(crossing_points[0]) == actions[i].pos_s
                    or list(crossing_points[0]) == actions[j].pos_s
                    or list(crossing_points[0]) == actions[i].base_pos
                    or list(crossing_points[0]) == actions[j].base_pos
                ):
                    return (actions[i], actions[j])

    return None


def shapes_overlap_with_intersection(shape1, shape2):
    """
    Check if two shapes (triangles, lines, or points) in 2D space overlap
    and return the intersection points.

    Args:
        shape1: List of points [(x1, y1), (x2, y2), (x3, y3)]
        shape2: List of points [(x1, y1), (x2, y2), (x3, y3)]

    Returns:
        tuple: (bool, list) - (True if shapes overlap, list of intersection points)
    """

    def is_triangle(points):
        """
        Check if three points form a proper triangle (not collinear).

        Args:
            points: List of three points [(x1, y1), (x2, y2), (x3, y3)]

        Returns:
            bool: True if points form a proper triangle, False if collinear
        """
        if len(points) != 3:
            return False

        (x1, y1), (x2, y2), (x3, y3) = points

        # Calculate area using cross product
        area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

        # If area is close to zero, points are collinear
        return area > 1e-10

    # Ensure both shapes have at least one point
    if not shape1 or not shape2:
        return False, []

    # Convert to proper format
    shape1 = [tuple(p) for p in shape1]
    shape2 = [tuple(p) for p in shape2]

    # Remove duplicate points
    shape1 = list(dict.fromkeys(shape1))
    shape2 = list(dict.fromkeys(shape2))

    # To store intersection points
    intersection_points = []

    # Check for point-point overlap
    if len(shape1) == 1 and len(shape2) == 1:
        if shape1[0] == shape2[0]:
            return True, [shape1[0]]
        return False, []

    # Helper function to check if a point is inside a triangle
    def point_in_triangle(pt, triangle):
        if not is_triangle(triangle):
            return False

        x, y = pt
        (x1, y1), (x2, y2), (x3, y3) = triangle

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(pt, (x1, y1), (x2, y2))
        d2 = sign(pt, (x2, y2), (x3, y3))
        d3 = sign(pt, (x3, y3), (x1, y1))

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    # Helper function to check if point is on a line segment
    def point_on_segment(p, segment):
        p1, p2 = segment

        # Check if point is collinear with segment
        cross_product = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p[0] - p1[0]) * (
            p2[1] - p1[1]
        )
        if abs(cross_product) > 1e-10:
            return False

        # Check if point is within the bounding box of the segment
        if min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[
            1
        ] <= max(p1[1], p2[1]):
            return True

        return False

    # Helper function to find intersection point of two line segments
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            # Lines are parallel
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        # Check if intersection point is on both segments
        if point_on_segment((x, y), line1) and point_on_segment((x, y), line2):
            return (x, y)

        return None

    # Helper function to check if two line segments intersect and find the intersection point
    def segments_intersect(seg1, seg2):
        p1, p2 = seg1
        p3, p4 = seg2

        # Check if any endpoints lie on the other segment
        for p in [p1, p2]:
            if point_on_segment(p, (p3, p4)):
                return True, p

        for p in [p3, p4]:
            if point_on_segment(p, (p1, p2)):
                return True, p

        # Find intersection point
        intersect_point = line_intersection(seg1, seg2)
        if intersect_point:
            return True, intersect_point

        return False, None

    # Case 1: Both are triangles
    if len(shape1) == 3 and len(shape2) == 3:
        if is_triangle(shape1) and is_triangle(shape2):
            # Check if any point of one triangle is inside the other
            for point in shape1:
                if point_in_triangle(point, shape2):
                    intersection_points.append(point)

            for point in shape2:
                if point_in_triangle(point, shape1):
                    intersection_points.append(point)

    # Create edges for both shapes
    edges1 = [(shape1[i], shape1[(i + 1) % len(shape1)]) for i in range(len(shape1))]
    if len(shape1) == 1:
        edges1 = []
    elif len(shape1) == 2:
        edges1 = [(shape1[0], shape1[1])]

    edges2 = [(shape2[i], shape2[(i + 1) % len(shape2)]) for i in range(len(shape2))]
    if len(shape2) == 1:
        edges2 = []
    elif len(shape2) == 2:
        edges2 = [(shape2[0], shape2[1])]

    # Check for point-edge intersections
    for point in shape1:
        for edge in edges2:
            if point_on_segment(point, edge):
                intersection_points.append(point)

    for point in shape2:
        for edge in edges1:
            if point_on_segment(point, edge):
                intersection_points.append(point)

    # Check for edge-edge intersections
    for edge1 in edges1:
        for edge2 in edges2:
            intersects, point = segments_intersect(edge1, edge2)
            if intersects and point:
                intersection_points.append(point)

    # Remove duplicate intersection points (with tolerance)
    unique_points = []
    for p in intersection_points:
        is_duplicate = False
        for up in unique_points:
            if np.sqrt((p[0] - up[0]) ** 2 + (p[1] - up[1]) ** 2) < 1e-10:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(p)

    return len(unique_points) > 0, unique_points
