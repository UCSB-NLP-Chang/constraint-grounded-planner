import json
from typing import Dict

SYSTEM = """Below, I will provide you with a detailed task description, an example environment, and its corresponding ground-truth plan for a multi-robot planning task.

You will **assume the role of a central planner**. Your task is to simulate the step-by-step thinking process that logically leads you to the provided ground-truth plan.

Your thinking should be presented from a **first-person perspective**, clearly demonstrating reasoning, collision avoidance considerations, and planning decisions. For example:

"Let me first analyze this scenario carefully. Robot 1 can reach object A easily without collision, but robot 2 might interfere if it moves simultaneously. To avoid collision, I will move robot 1 first, and then proceed with robot 2. Therefore, the first step in the plan should be..."

## Task Description:
You are a central planner responsible for coordinating multiple robotic arms operating in a grid-like environment. Your goal is to plan and execute efficient, collision-free movements to transport objects to their designated target positions.


*Task Representation:*
* Objective: Move all objects to their specified target locations safely and efficiently.
* Input: A detailed map state containing positions of robots, objects, and target locations.
* Output: A precise movement plan specifying each robot arm's actions for moving objects.


*Position Representation:*
* All positions (robots, objects, targets) are given by their center coordinates, e.g., [0.25, 0.25], [0.75, 1.25].
* Robots have a fixed base location and an extendable arm with a limited reach range.


*Movement Rules:*
* Each robot arm can only move within a limited range relative to its fixed base position:
    * X-axis: from (Base_X - 1.0) to (Base_X + 1.0) (exclusive).
    * Y-axis: from (Base_Y - 1.0) to (Base_Y + 1.0) (exclusive).
* For example:
    * If a robot's base is [1.0, 1.0], its arm can reach [0.25, 0.75] or [1.25, 1.75], but not [0, 0.25] or [2.0, 0.75].
    * Robots may move an object only if their arm aligns exactly with the object's current position, and if explicitly indicated in the action (move_object: True).


## How to Generate Your Response:
Your response must **clearly indicate your thinking process** enclosed in <think> and </think> tags, followed by the generated step of the movement plan.


*Thinking:*
* Clearly describe your analysis and decisions from a first-person perspective.
* Identify potential collisions explicitly and explain how you avoid them.
* Highlight your reasoning for movement choices, considering efficiency and collision avoidance.

*Movement Plan (Output):*
* Your generated step of the movement plan should be in markdown format and contain a JSON dictionary, with robot names as keys and their movement instructions as values, structured as follows:
```json
{
    "robot_name": "start_position -> end_position, move_object"
}
```
* *start_position* and *end_position* represent the *[x, y]* coordinates of the robot arm's movement.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* Robots without actions in the current step should not be included.
Ensure your final step completes the objective of placing all objects at their target positions, and your plan forms a valid JSON array.


## Collision Avoidance Rules:
Your plan must strictly avoid collisions, as follows:
* Robot-Robot Collision:
    * Two robot arms cannot occupy the same position simultaneously.
    * Robot arms cannot have intersecting movement trajectories during a step.
    * For example:
        * Collision occurs if Robot 1 moves [0.75, 0.75]→[0.75, 1.25] and Robot 2 moves [2.25, 1.75]→[0.75, 1.25] (same endpoint).
        * Collision occurs if Robot 1 moves [0.25, 0.25]→[0.75, 0.75] and Robot 2 moves [0.25, 0.75]→[0.75, 0.25] (intersecting paths).
    * Object-Object Collision:
        * Two objects cannot occupy the same position at any time.

## Plan Efficiency Considerations:
* Each step of your plan involves simultaneous robot arm movements from their current positions to specified target positions.
* Each robot arm moves at a constant speed of 0.5 units/time.
* The duration of each step is determined by the longest single-arm movement within that step.
* The total execution time is the sum of all individual step durations.
* You should aim to minimize total execution time while ensuring collision-free movements and successful object placements.
"""

MESSAGE_TEMPLATE = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": """## Example Environment and one step of the ground-Truth Plan:
Below is an example scenario and one step of the ground-truth solution:

```text
{environment}
```

With the above information clearly provided, please start by explicitly presenting your first-person reasoning for this plan enclosed in <think></think> tags.""",
    },
]


def get_object_position(object_positions):
    res = "Object positions:\n" + "\n".join(
        [f"\tObject {objid}: {objpos}" for objid, objpos in object_positions.items()]
    )
    return res


def get_target_positions(target_positions):
    res = "Target positions:\n" + "\n".join(
        [
            f"\tObject {objid} target: {objpos}"
            for objid, objpos in target_positions.items()
        ]
    )
    return res


def get_robot_positions(target_positions):
    print(target_positions)
    res = "Robot positions:\n" + "\n".join(
        [
            f"\tRobot {objid}: base: {objpos['base_pos']}, arm: {objpos['arm_pos']}"
            for objid, objpos in target_positions.items()
        ]
    )
    return res


def get_plan(data):
    plan = {}
    robot_id = (
        data["robot_id"] + 1
    )  # Adjusting index to match example (Robot 1, Robot 2, etc.)
    start_pos = data["pos_s"]
    end_pos = data["pos_e"]
    carry_status = data["carry"]
    plan[f"Robot {robot_id}"] = f"{start_pos} -> {end_pos}, {carry_status}"
    return plan


def Plan2Text(plan: Dict):
    object_positions = plan["env"]["objects"]
    target_positions = plan["env"]["targets"]
    robot_positions = plan["env"]["robots"]
    res = (
        "The current map state is as follows:\n"
        + get_object_position(object_positions)
        + "\n"
        + get_target_positions(target_positions)
        + "\n"
        + get_robot_positions(robot_positions)
    )

    all_step_plans = [get_plan(plan) for plan in plan["plan"]]
    res = (
        res
        + "\n"
        + "The plan for current step is as follows:\n"
        + json.dumps(all_step_plans, indent=2)
    )
    return res
