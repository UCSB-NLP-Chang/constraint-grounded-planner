import json
from typing import Dict

SYSTEM = """Below, I will provide you with a detailed task description, an example environment, and its corresponding ground-truth plan for a multi-robot planning task.

You will **assume the role of a central planner**. Your task is to simulate the step-by-step thinking process that logically leads you to the provided ground-truth plan.

Your thinking should be presented from a **first-person perspective**, clearly demonstrating reasoning, collision avoidance considerations, and planning decisions. For example:

"<think>
Let me first analyze this scenario carefully. Robot 1 can reach object A easily without collision, but robot 2 might interfere if it moves simultaneously. To avoid collision, I will move robot 1 first, and then proceed with robot 2. Therefore, the current step in the plan should be...
</think>"

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
    "robot_name": "start_position -> end_position, move_object",
    "robot_name": "start_position -> end_position, move_object",
}
```
* *start_position* and *end_position* represent the *[x, y]* coordinates of the robot arm's movement.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* Robots without actions in the current step should not be included.
Ensure your final step completes the objective of placing all objects at their target positions, and your plan forms a valid JSON dict.


## Collision Avoidance Rules:
* Robot-Robot Collision:
    * Two robot arms cannot occupy the same position simultaneously.
    * Robot arms cannot cannot intersect with each other or have intersecting movement trajectories during a step movement.
    * For example:
        * Collision occurs if Robot 1 moves [0.75, 0.75] -> 0.75, 1.25] and Robot 2 moves [2.25, 1.75] -> [0.75, 1.25] (same endpoint).
        * Collision occurs if Robot 1 moves [0.25, 0.25] -> [0.75, 0.25] and Robot 2 moves [1.25, 0.25] -> [0.25, 0.75] (intersecting arms as the end position Robot 1 is at the arm, as the end of Robot 2 arm position occupies [0.75, 0.25])
        * Collision occurs if Robot 1 moves [0.25, 0.25] -> [0.75, 0.75] and Robot 2 moves [0.25, 0.75] -> [0.75, 0.25] (intersecting movement as both arms moves across [0.5, 0.5]).
    * Object-Object Collision:
        * Two objects cannot occupy the same position at any time.


## Plan Efficiency Considerations:
* Each step of your plan involves simultaneous robot arm movements from their current positions to specified target positions.
* Each robot arm moves at a constant speed of 0.5 units/time.
* The duration of each step is determined by the longest single-arm movement within that step.
* The total execution time is the sum of all individual step durations.
* You should aim to minimize total execution time while ensuring collision-free movements and successful object placements.


## Example Reasoning Trace
Here is an full example reasoning trace, the synthesized thinking should resemble this style:
```markdown
</think>
Okay, let's tackle this problem step by step. So, the goal is to move Object 0 from [0.75, 1.75] to its target [0.25, 1.75]. We have two robots: Robot 0 with base at [0.0, 1.0] and arm currently at [0.25, 1.25], and Robot 1 with base at [1.0, 1.0] and arm at [1.25, 1.25].

First, I need to figure out which robot can reach the object. Object 0 is at [0.75, 1.75]. Let's check Robot 0's reach. Robot 0's base is at [0.0, 1.0], so their arm can move from (0.0 -1.0) to (0.0 +1.0) on X, which is -1.0 to 1.0. But since positions are centers, maybe the actual reachable area is from 0.0 to 1.0? Wait, the example given earlier said a base at [1.0,1.0] can reach up to 2.0 exclusive. So, for Robot 0's X-axis: from 0.0 -1.0 = -1.0 to 0.0 +1.0 = 1.0. But the object's X is 0.75, which is within -1.0 to 1.0. Similarly, Y-axis for Robot 0 is from 1.0 -1.0 = 0.0 to 1.0 +1.0 = 2.0. The object's Y is 1.75, which is within 0.0 to 2.0. So Robot 0 can reach the object.

Wait, the current arm position of Robot 0 is [0.25, 1.25], which is within its range. But the object is at [0.75, 1.75], which is also within Robot 0's reach. So Robot 0's arm needs to move to the object's position to pick it up. However, the object's current position is [0.75, 1.75]. Let me check if Robot 0 can move its arm there. The base is [0.0,1.0], so X: 0.75 is within -1.0 to 1.0 (yes, since 0.75 <1.0). Y:1.75 is within 0.0 to 2.0. So yes.

Now, Robot 1's current arm position is [1.25, 1.25]. Their base is [1.0,1.0], so X can be from 0.0 to 2.0, Y same. The object is at [0.75,1.75]. X: 0.75 is within 0.0 to 2.0, Y:1.75 is within 0.0 to 2.0. So Robot 1 could also reach the object. But since the target is [0.25,1.75], which is closer to Robot 0's base. Let me check if moving the object via Robot 0 is better.

Wait, the target for Object 0 is [0.25,1.75]. Let's see if Robot 0 can reach that. Robot 0's X range is -1.0 to 1.0. 0.25 is within that. Y is 1.75, which is within 0.0 to 2.0. So yes. So Robot 0 can move the object from its current position to the target.

But right now, the object is at [0.75,1.75]. Robot 0's arm is at [0.25,1.25]. So first, Robot 0 needs to move its arm to the object's position. Then, pick it up and move it towards the target. Alternatively, perhaps Robot 1 can help? Let's see. The target is [0.25,1.75]. Robot 1's base is [1.0,1.0]. The X for the target is 0.25, which is 1.0 -0.75. So 0.25 is within Robot 1's X range (0.0 to 2.0). Y is 1.75, within 0.0 to 2.0. So Robot 1 could also reach the target. But since the object is currently in Robot 0's reach, maybe Robot 0 should handle moving it to the target.

Wait, the object's current position is [0.75,1.75], which is in both robots' reach. But to move it to [0.25,1.75], which is closer to Robot 0's base. Let's see: if Robot 0 can take the object, move it west to the target. But first, the arm needs to be at the object's current position. Right now, Robot 0's arm is at [0.25,1.25], which is not aligned with the object. So the first step for Robot 0 would be to move its arm from [0.25,1.25] to [0.75,1.75], then pick up the object (move_object: True). But in this step, the movement is just moving the arm to the object's position. Then, in the next step, move the arm (with the object) to the target.

But wait, each step is a movement. So in the next step, can Robot 0 move its arm to the object's position, pick it up, and then move towards the target in the same step? Or does each step involve moving the arm to a new position, and whether they carry the object during that movement.

The movement plan's output is for the next step. So in the first step, Robot 0 needs to move its arm from current position [0.25,1.25] to [0.75,1.75] (the object's position), and set move_object to True. But wait, when moving to the object's position, to pick it up, the move_object should be True only when the arm is moving the object. Wait, the movement rules state: Robots may move an object only if their arm aligns exactly with the object's current position, and if explicitly indicated in the action (move_object: True). So to pick up the object, the robot's arm must first be at the object's position, and then when moving, they can move_object: True. So in the current step, if the robot's arm is not at the object's position, they first move their arm to that position (with move_object: False), then in the next step, they can move the object.

So in this case, the object is at [0.75,1.75]. Robot 0's arm is at [0.25,1.25]. So Robot 0 needs to move its arm from [0.25,1.25] to [0.75,1.75], with move_object: False. Once the arm is there, in the next step, they can move the object towards the target.

But wait, can the robot pick up the object in the same step as moving to the object's position? The rules say, "If a robot needs to move an object within its range and the arm is not aligned with the object, the robot should first move its arm to the position of that object." So first, move the arm to the object's position (without moving the object), then in the next step, move the object. So in this step, the movement for Robot 0 would be to go from current arm position to [0.75,1.75], move_object: False. Then, once there, in the next step, move the object.

But is there another possibility? Let's check Robot 1's current position. Robot 1's arm is at [1.25,1.25], and their base is [1.0,1.0]. The object's position is [0.75,1.75], which is within Robot 1's reach (X: 0.75 is between 0.0 and 2.0; Y:1.75 is between 0.0 and 2.0). So Robot 1 could also move to the object. But the target is [0.25,1.75], which is within Robot 0's reach but not necessarily in Robot 1's reach. Wait, Robot 1's X range is 0.0 to 2.0, so 0.25 is within that. Y is 1.75, same as before. So Robot 1 could also move the object to the target. But which robot is better to use?

If Robot 0 is moving the object, the path would be from [0.75,1.75] to [0.25,1.75], which is a horizontal move. The distance is 0.5 units (0.75 to 0.25). That would take 1 unit of time (distance divided by speed 0.5), but since movement is simultaneous, if both robots are moving, the time is the longest movement. Alternatively, if Robot 1 moves the object, the path would be from [0.75,1.75] to [0.25,1.75], which is a horizontal move of 0.5 units as well. But Robot 1's arm is currently at [1.25,1.25]. So to reach the object, Robot 1 would need to move from [1.25,1.25] to [0.75,1.75], which is a diagonal move. The distance is sqrt( (0.5)^2 + (0.5)^2 ) ≈ 0.707, which takes about 1.414 units of time. Then, moving the object to the target would take another 0.5 units. So total time would be longer if using Robot 1. So better to use Robot 0.

So, plan step 1: Robot 0 moves arm to [0.75,1.75] (distance from current position is sqrt( (0.5)^2 + (0.5)^2 ) ≈ 0.707, which would take 0.707 / 0.5 = 1.414 time units. But we need to check for collisions. Are there any other robots moving? Robot 1's current arm is at [1.25,1.25]. Are they moving? If we don't include Robot 1 in the movement plan, their arm stays at [1.25,1.25]. So Robot 0's movement to [0.75,1.75] would not collide with Robot 1's static position. So that's safe.

But what if Robot 1 is moving? Let's see. If we move Robot 0 to [0.75,1.75], and Robot 1 is not moving, then no collision. But perhaps Robot 1 can do something else. Wait, the only object is Object 0. So perhaps there's no need for Robot 1 to act. So the initial step is for Robot 0 to move to the object's position. Then, in the next step, move the object to the target. But in this current step, only Robot 0 is moving.

Wait, but maybe there's a way to have Robot 1 help in some way? But since there's only one object, maybe it's best to just have Robot 0 handle it.

Another consideration: Are there any obstacles or other objects? No, Object 0 is the only one. So the main steps are:

1. Robot 0 moves arm to Object 0's position (0.75,1.75) without moving the object (move_object: False).

2. Once there, Robot 0 moves the object to the target (0.25,1.75) with move_object: True.

But let's check for collisions in step 1. Are there any robots moving into the same position? Robot 1's arm is at [1.25,1.25], which is different from Robot 0's destination [0.75,1.75]. So no collision. Also, their movement paths: Robot 0 is moving from [0.25,1.25] to [0.75,1.75]. The path is a straight line. Robot 1 isn't moving, so no intersection. So this movement is safe.

Thus, the next step plan should involve Robot 0 moving to the object's position. So the JSON would be:

{"Robot 0": "[0.25, 1.25] -> [0.75, 1.75], False"}

But wait, what about Robot 1? Is there any reason to move Robot 1 in this step? Since there's only one object and it's being handled by Robot 0, probably not. So the plan is just Robot 0's movement.

Wait, but the example input had both robots moving. But in this case, maybe only Robot 0 is needed.

Wait, but after Robot 0 moves to the object's position, then in the next step, they can move the object. So this seems correct.

Another check: The object's current position is [0.75,1.75]. When Robot 0 moves its arm there, if move_object is False, does that mean the arm is now at the object's position, but the object hasn't been moved yet? Or does moving the arm there automatically pick up the object? According to the rules: "Robots may move an object only if their arm aligns exactly with the object's current position, and if explicitly indicated in the action (move_object: True)." So moving the arm to the object's position with move_object: False would not move the object. Then, in the next step, when the arm is already aligned, the robot can move the object by setting move_object: True. So the process is correct.

So the next step is Robot 0 moving its arm to the object's position. Then, in the following step, moving the object towards the target.

Thus, the correct next-step plan is only Robot 0 moving to [0.75,1.75] with move_object: False. There's no collision here because Robot 1 is not moving. So the JSON output is as above.
</think>
```
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
