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
[
{
    "robot_name": "start_position -> end_position, move_object"
    "robot_name": "start_position -> end_position, move_object",
},
{
    "robot_name": "start_position -> end_position, move_object",
}
]
```
* *start_position* and *end_position* represent the *[x, y]* coordinates of the robot arm's movement.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* Robots without actions in the current step should not be included.
Ensure your final step completes the objective of placing all objects at their target positions, and your plan forms a valid JSON array.


## Collision Avoidance Rules:
Your plan must strictly avoid collisions, as follows:
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
Here is an full example reasoning trace, your synthesized thinking should resemble this style:
<think>
Alright, let me really think through this step by step, questioning myself as I go to make sure I'm not missing anything.

Okay, so we have two objects to move and six robots. First, let me understand the positions:

**Objects:**
- Object 0: starts at [0.25, 0.75], needs to go to [0.25, 2.25]
- Object 1: starts at [1.25, 2.25], needs to go to [1.25, 0.75]

**Robots:**
Each has a base and an initial arm position. Their arms can move within 1 unit of their base in any direction.

Wait, how exactly does their reach work? If a robot is at base [x,y], its arm can reach anywhere where:
- X coordinate is between x-1.0 and x+1.0
- Y coordinate is between y-1.0 and y+1.0

But looking at the initial positions, some coordinates are decimals like 0.25. Does that mean the grid is divided into smaller units? Or is it just floating-point precision? Hmm, maybe it doesn't matter - we just need to know what each robot can reach.

Let me list which robots can reach which positions.

**For Object 0 at [0.25, 0.75]:**
- Robot 0: base [0.0, 1.0]. Can reach X from -1.0 to 1.0, Y from 0.0 to 2.0. So yes, it can reach [0.25, 0.75].
- Robot 1: base [0.0, 2.0]. Y reach is 1.0 to 3.0. So Y=0.75 is too low. Can't reach.
- Similarly, Robots 2-5 can't reach Y=0.75.
So only Robot 0 can pick up Object 0 initially.

**For Object 1 at [1.25, 2.25]:**
- Robot 4 is already there! Its arm is at [1.25, 2.25] holding Object 1. So that's easy - Robot 4 can move it.

Now, where do they need to go?

**Object 0 to [0.25, 2.25]:**
- Robot 1 can reach here (Y reach 1.0-3.0)
- Robot 2 could also reach it (Y reach 2.0-4.0), but Robot 1 is closer.

**Object 1 to [1.25, 0.75]:**
- Robot 3 can reach here (Y reach 0.0-2.0)
- No other robots can reach that low on Y.

Okay, so the basic idea is:
- Robot 0 picks up Object 0 and moves it up to where Robot 1 can take over
- Robot 4 moves Object 1 down to where Robot 3 can take over

But how exactly? Let me think about Object 1 first since Robot 4 is already holding it.

**Moving Object 1:**
Robot 4 is at [1.25, 2.25] holding Object 1. Needs to get to [1.25, 0.75], but Robot 4 can't reach that low (its minimum Y is 1.0). So we need to hand off to Robot 3.

Where can we hand off? Somewhere both can reach - so Y between 1.0 and 2.0 (Robot 4's Y reach is 1.0-3.0, Robot 3's is 0.0-2.0).

Let's say we choose Y=1.75 as the handoff point.

But wait - Robot 3's initial arm position is [1.25, 1.25]. If Robot 4 moves Object 1 to [1.25, 1.75], then Robot 3 would need to move up to [1.25, 1.75] to take it. But during that move, Robot 4 would still be at [1.25, 1.75], causing a collision.

Hmm, how to avoid that? Maybe:
1. First, have Robot 3 move away from [1.25, 1.25] to [1.25, 0.75]
2. Then Robot 4 moves Object 1 down to [1.25, 1.75]
3. Then Robot 3 moves back up to [1.25, 1.75] to pick it up
4. Then Robot 3 moves it down to [1.25, 0.75]

That would work but takes 4 steps just for Object 1. Is there a better way?

Alternatively, maybe we can have Robot 4 move Object 1 to [1.25, 1.25], but then Robot 3 is already there - that's a problem. Unless... can Robot 3 be somewhere else initially? Wait no, it starts at [1.25, 1.25].

Hmm, maybe we need to accept that we need multiple steps for the handoff.

Now let's think about Object 0.

**Moving Object 0:**
Robot 0 needs to pick it up at [0.25, 0.75], move it up to where Robot 1 can take it, say [0.25, 1.75], then Robot 1 can take it to [0.25, 2.25].

But similar problem - after Robot 0 moves Object 0 to [0.25, 1.75], Robot 1 would need to move to [0.25, 1.75] to take it, but Robot 0 is already there.

So we'd need:
1. Robot 0 picks up at [0.25, 0.75]
2. Moves to [0.25, 1.75]
3. Robot 0 moves away to [0.25, 1.25]
4. Robot 1 moves to [0.25, 1.75] to take it
5. Robot 1 moves to [0.25, 2.25]

Again, many steps. Can we combine these?

Maybe a combined plan where we interleave the steps:

Step 1:
- Robot 0 moves to [0.25, 0.75] (but doesn't pick up yet)
- Robot 4 moves Object 1 to [1.25, 1.75]

Step 2:
- Robot 0 picks up Object 0 and moves to [0.25, 1.75]
- Robot 3 moves to [1.25, 0.75] (getting out of the way)
- Robot 4 moves back to [1.25, 2.25] (leaving Object 1 at [1.25, 1.75])

Step 3:
- Robot 1 moves to [0.25, 1.75] to take Object 0
- Robot 3 moves back to [1.25, 1.75] to take Object 1

Step 4:
- Robot 1 moves Object 0 to [0.25, 2.25]
- Robot 3 moves Object 1 to [1.25, 0.75]

Does this avoid collisions? Let me check:

In Step 2:
- Robot 0 moves to [0.25, 1.75]
- Robot 3 moves to [1.25, 0.75]
- Robot 4 moves to [1.25, 2.25]
No collisions here - they're all moving to different spots.

In Step 3:
- Robot 1 moves to [0.25, 1.75] where Object 0 is (Robot 0 has moved away)
- Robot 3 moves to [1.25, 1.75] where Object 1 is (Robot 4 has moved away)
Still no collisions.

This seems to work! It's a bit like a dance where robots move in sequence to avoid bumping into each other.

The key insights are:
1. Only certain robots can reach certain positions due to their reach limits
2. Handoffs between robots need to be carefully coordinated to avoid collisions
3. Sometimes robots need to move away first before others can move in
4. We can optimize by having multiple robots move in the same step when their paths don't cross

After carefully working through the possible collisions and reach limitations, this 4-step plan should successfully move both objects to their targets.
</think>
"""

MESSAGE_TEMPLATE = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": """## Example Environment and Ground-Truth Plan:
Below is an example scenario and its ground-truth solution:

```text
{environment}
```

With the above information clearly provided, please start by explicitly presenting your first-person reasoning for the whole plan enclosed in <think></think> tags. Limit your thinking length within 2000 tokens.""",
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
    res = "Robot positions:\n" + "\n".join(
        [
            f"\tRobot {objid}: base: {objpos['base_pos']}, arm: {objpos['arm_pos']}"
            for objid, objpos in target_positions.items()
        ]
    )
    return res


def get_plan(plans):
    plan = {}
    for data in plans:
        robot_id = (
            # data["robot_id"] + 1
            data["robot_id"]
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
        "The initial map state is as follows:\n"
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
        + "The ground-truth plan is as follows:\n"
        + json.dumps(all_step_plans, indent=2)
    )
    return res
