import json
from typing import Dict

SYSTEM = """You are required to **assume the role of a central planner**. Your task is to simulate the step-by-step thinking process that logically leads you to the provided ground-truth plan.

Your thinking should be presented from a **first-person perspective**, clearly demonstrating your internal reasoning process of planning, validating and adjusting to avoid collision, and planning decisions. 

## Requirement for your generated firt-person thinking:
1. **First-Person Perspective**: Write your internal thoughts as if you are personally making the decisions:
   - Use phrases like "Let me see...", "Wait, is that correct?", "I should check collisions first..."
   - Demonstrate real-time analysis and potential hesitations or reconsiderations.
2. **Thinking Process with `<think>` Tags**:
   - Enclose your entire reasoning sequence in `<think>` ... `</think>` tags.
   - Include explicit checks, e.g. collision checks, range feasibility, and confirmations of correctness.
   - Pose questions to yourself, and then answer them. Show how you arrive at each movement decision.


Below is the detailed task description, an example environment, and its corresponding ground-truth next-step plan for a multi-robot planning task.

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
<think>
Okay, let's tackle this problem. The goal is to move Object 0 from [0.75, 1.75] to its target [0.25, 1.75]. I have two robots: Robot 0 with its base at [0.0, 1.0] and arm currently at [0.25, 1.25], and Robot 1 with its base at [1.0, 1.0] and arm at [1.25, 1.25].

First, I need to figure out which robot can reach Object 0 at [0.75, 1.75]. Robot 0's base is at [0.0, 1.0], so for its arm, X can range from -1.0 to 1.0 and Y from 0.0 to 2.0. The object's coordinates (0.75, 1.75) fit right in there, so Robot 0 can reach it. Wait, I should also check Robot 1. Robot 1's base is [1.0, 1.0], so X and Y can range from 0.0 to 2.0—0.75, 1.75 is also within that range, so Robot 1 can reach the object, too.

Now, the target is [0.25, 1.75]. Let me see if Robot 0 can also reach that. Yes, X = 0.25 is within -1.0 to 1.0, and Y = 1.75 is within 0.0 to 2.0. So Robot 0 can handle both the object's current position and its target. Robot 1 can technically do the job as well, but the target is closer to Robot 0's base, which seems simpler.

Wait, the object is currently at [0.75, 1.75], but Robot 0's arm is at [0.25, 1.25]. So I need to move Robot 0's arm to the object first. According to the rules, to pick something up (move_object: True), the arm must already be aligned with that object. That means I can't pick it up during the same move that I use to get there if I start somewhere else. So I first move Robot 0's arm to [0.75, 1.75] with move_object: False. Then, once the arm is on top of the object, I can move the object itself by setting move_object: True on the next move.

Wait, could Robot 1 jump in? Technically yes, but Robot 1 is starting at [1.25, 1.25], which is farther away from the object, so it would have to move diagonally before even picking up the object. That's less efficient than letting Robot 0 handle the whole thing. Also, collisions aren't a big concern if Robot 1 just stays put while Robot 0 completes the move.

Therefore, my best next move is simply to have Robot 0 move from [0.25, 1.25] to [0.75, 1.75] with move_object: False, because that places the arm right where the object is. Then, next time, I can pick up and move the object to [0.25, 1.75].

So the JSON for the immediate next action is:

{"Robot 0": "[0.25, 1.25] -> [0.75, 1.75], False"}
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
