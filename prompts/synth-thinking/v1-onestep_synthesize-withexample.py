import json
from typing import Dict

SYSTEM = """
You are a central planner tasked with coordinating multiple robotic arms in a grid-like environment to move objects to their target positions. You will receive:

1. **A detailed environment** with the locations of robots, objects, and targets.
2. **A ground-truth plan** that successfully moves all objects to their targets.

Your job is to produce a **thinking-out-loud narrative** in the first person, capturing your internal process of planning, validating, and adjusting to ensure collision-free, efficient movements. After this thought process, output a **single plan step** in valid JSON format.

#### Requirements for Your Response

1. **First-Person Perspective**: Write your internal thoughts as if you are personally making the decisions:
   - Use phrases like "Let me see...", "Wait, is that correct?", "I should check collisions first..."
   - Demonstrate real-time analysis and potential hesitations or reconsiderations.

2. **Thinking Process with `<think>` Tags**:
   - Enclose your entire reasoning sequence in `<think>` ... `</think>` tags.
   - Include explicit checks, e.g. collision checks, range feasibility, and confirmations of correctness.
   - Pose questions to yourself, and then answer them. Show how you arrive at each movement decision.

3. **Action Output**:
   - Provide a movement step in a valid JSON dictionary (in *markdown* format), where each key is a robot name, and the value is a string describing the movement from start to end coordinates plus a boolean for whether the robot moves an object.
   - Only include robots that perform an action in this step.
   - Example format:
     ```json
     {
       "Robot A": "[x1, y1] -> [x2, y2], True",
       "Robot B": "[x3, y3] -> [x4, y4], False"
     }
     ```

4. **Collision & Feasibility Checks**:
   - Describe how you check for robot-robot collisions (both end positions and trajectories).
   - Describe how you ensure objects do not collide.
   - Include range checks to confirm each robot's arm remains within allowable boundaries.

5. **Efficiency Considerations**:
   - Briefly acknowledge if your chosen movement is optimal or near-optimal in terms of step durations or total plan time.
   - If you spot a potential inefficiency, mention it, but explain your final decision anyway.

#### Task Description (Reference)
- Each robot has a limited arm reach around its base: `(Base - 1.0, Base + 1.0)` in the X and Y directions.
- A robot can move an object if and only if the arm's position matches the object's position and `move_object` is `True`.
- Collision is strictly disallowed for robots, arms, and objects at any point in movement.

#### Example of Thought Process
Here is a concise example of how your output might look:
```markdown
</think>
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
