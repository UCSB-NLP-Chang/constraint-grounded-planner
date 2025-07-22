import json
from typing import Dict, List

SYSTEM = """You are required to **assume the role of a central planner**. Your task is to simulate the step-by-step thinking process that logically leads you to the provided ground-truth plan.

Your thinking should be presented from a **first-person perspective**, clearly demonstrating your internal reasoning process of planning, validating and adjusting to avoid collision, and planning decisions. 

## Requirement for your generated firt-person thinking:
1. **First-Person Perspective**: Write your internal thoughts as if you are personally making the decisions:
   - Use phrases like "Let me see...", "Wait, is that correct?", "I should check collisions first...", "Can I parallel two robot movements to make the plan more efficient?"
   - Demonstrate real-time analysis and potential hesitations or reconsiderations.
2. **Thinking Process with `<think>` Tags**:
   - Enclose your entire reasoning sequence in `<think>` ... `</think>` tags.
   - Make sure you have explicit checks, e.g. collision checks, range feasibility, and confirmations of correctness. You can start the explicit checks with "Wait", "Hmm", "let me check", etc.
   - Make sure to pose questions to yourself, and then answer them. Show how you arrive at each movement decision.
   - You must include multiple explict checks and self-questioning in your thinking process.


Below is the detailed task description. You can learn the rules for the task from these descriptions. 
   
## Task Description:
You are a central planner responsible for coordinating multiple robotic arms operating in a grid-like environment. Your goal is to plan and execute efficient, collision-free movements to transport objects to their designated target positions.


*Task Representation:*
* Objective: Move all objects to their specified target locations safely and efficiently.
* Input: A detailed map state containing positions of robots, objects, and target locations.
* Output: A precise movement plan specifying each robot arm's actions for moving objects.


*Position Representation:*
* All positions (robots, objects, targets) are given by their center coordinates, e.g., [0.55, 1.65], [2.75, 0.55].
* Robots have a fixed base location and an extendable arm with a limited reach range.


*Movement Rules:*
* Each robot arm can only move within a circular band around its fixed base position:  
  - Let d = sqrt((X - Base_X)**2 + (Y - Base_Y)**2).  
  - The arm may reach (X, Y) only if 0.4 < d < 0.8

* For example:  
  - If a robot's base is at [1.1, 1.1]:  
    - It can reach [0.55, 0.55] since sqrt((1.1 - 0.55)**2 + (1.1 - 0.55)**2) around 0.77 < 0.8
    - It can reach [0.6, 1.1] since sqrt((0.6 - 1.1)**2 + (1.1 - 1.1) ** 2)) = 0.5 > 0.4
    - It cannot reach [2.0, 1.1] because sqrt(0.9**2 + 0**2) = 0.9, which exceeds 0.8
    - It cannot reach [2.25, 0.65] because sqrt(1.15**2 + 0.45**2) around 1.23, which exceeds 0.8
  - If a robot needs to move an object within its range and the arm is not aligned with the object, the robot should first move its arm to the position of that object. By aligning, it meas the distance between object center and arm position is less than 0.1
  - When you plan a move, please follow following rules:
    - First check that the proposed target lies within the circular band 0.5 < d < 0.8.  
    - If it does not, adjust your plan or reject that movement.  
    - If the arm is not yet aligned with an object it needs to move and that object lies within the band, plan a preliminary move to position the arm aligned with the object before picking it up.



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
    "robot_name1": "Move end_position, move_object",
    "robot_name2": "Move end_position, move_object"
}
```
* *end_position* represent the target *[x, y]* coordinates of the robot arm end point of the movement around circular path. Note that only the arm moves while its base remains fixed.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* One robot can only be moved once in each step, which means that no repeated keys are allowed in the same step.
* Robots without actions in the current step should not be included.
Ensure your final step completes the objective of placing all objects at their target positions, and your plan forms a valid JSON array.


## Collision Avoidance Rules:
Your plan must strictly avoid collisions, as follows:
* Robot-Robot Collision
  * Each robot arm always swings along a smooth **circular** path around its base.
  * Two robot arms cannot occupy the same position at the end of a move.  
  * Their curved paths must not cross or share any point during the move.  
  * Sometimes a robot needs to move its arm to a safe position to avoid collision between another robot that move its arm to reach an object.
  * **Example:**  
    * robot_0 swings from [0.25, 0.25] to [0.75, 0.75] and robot_1 swings from [0.25, 0.75] to [0.75, 0.25] at the same time. Both arcs pass through [0.5, 0.5], causing a collision.
* Object-Object Collision
  * Two objects cannot occupy the same (x, y) at any time.  
  * If you move more than one object at once, they must have different drop-off points and non-crossing straight-line paths.
* Robot-Object Collision
  * An arm's circular path must not sweep through any object it isn't carrying.  
  * Before moving, confirm the curved trajectory does not pass over another object's position.


## Example Reasoning Trace
You should not copy from below example, but here is an full example reasoning trace, the synthesized thinking should resemble this style:
```markdown
Let me analyze the initial state of the robotic arms and their targets. I have two robots:
- **Robot 0**:  
  - Base position: [1.1, 1.1]  
  - Arm position: [1.23, 0.61]
- **Robot 1**:  
  - Base position: [1.1, 2.2]  
  - Arm position: [1.23, 1.71]
There is one object in the environment:

- **Object 0**: located at [0.55, 1.65], with a target destination of [0.55, 2.75].

My first task is to determine which robot can reach Object 0. I'll calculate the distance between each robot's arm and the object:

- **Robot 0** to Object 0:  
  `sqrt((1.23 - 0.55)^2 + (0.61 - 1.65)^2) ≈ 1.34` → This exceeds the reach limit (0.8), so Robot 0 cannot reach the object.

- **Robot 1** to Object 0:  
  `sqrt((1.23 - 0.55)^2 + (1.71 - 1.65)^2) ≈ 0.826` → Slightly above 0.8, which might be borderline depending on the exact threshold. I need to test whether the robot can get slightly closer to meet the threshold for picking up the object.

Next, I consider moving **Robot 1's** arm closer to the object for alignment. The goal is to reduce the distance to less than 0.1 to allow for grasping.

Now I should evaluate if this first move is feasible:

- **Path from [1.23, 1.71] to [0.55, 1.65]**:  
  - The move is downward and to the left. There's no indication of obstacles or other robots along this path.
  - Since Robot 0 is stationary at [1.1, 1.1], and no other objects block the direct route, the path seems collision-free.

So, for the **first step**, I will focus on moving Robot 1's arm to align with Object 0 at [0.55, 1.65]. This move doesn't involve carrying the object yet—just getting into position.

I'll now try this move before thinking about any subsequent steps.
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

With the above information clearly provided, please start by explicitly presenting your first-person reasoning for this plan enclosed in <think></think> tags. Make sure you include explicit checks and self-questioning in your thinking process. Your reasoning should be clear and easy to follow, as if you are explaining it to someone else. Make your thinking short, limit your thinking length within 1000 tokens.""",
    },
]


def convert_tup(tup):
    # return f"[{tup[0]:.2f}, {tup[1]:.2f}]"
    return f"[{round(float(tup[0]), 2)}, {round(float(tup[1]), 2)}]"


def get_object_position(object_positions):
    res = "Object positions:\n" + "\n".join(
        [
            f"\tObject {objid}: {convert_tup(objpos)}"
            for objid, objpos in object_positions.items()
        ]
    )
    return res


def get_target_positions(target_positions):
    res = "Target positions:\n" + "\n".join(
        [
            # f"\tObject {objid} target: {objpos}"
            f"\tObject {int(objid.split('_')[-1])} target: {convert_tup(objpos)}"
            for objid, objpos in target_positions.items()
            if objpos is not None
        ]
    )
    return res


def get_robot_positions(target_positions):
    res = "Robot positions:\n" + "\n".join(
        [
            # f"\tRobot {objid}: base: {objpos[0]}, arm: {objpos[1]}"
            f"\tRobot {objid}: base: {objpos['base_pos']}, arm: {convert_tup(objpos['arm_pos'])}"
            for objid, objpos in target_positions.items()
        ]
    )
    return res


def Map2Text(map, object_positions, target_positions, robot_positions):
    res = (
        get_object_position(object_positions)
        + "\n"
        + get_target_positions(target_positions)
        + "\n"
        + get_robot_positions(robot_positions)
    )
    return res


def Plan2Text(obs, target_positions, plan: list):
    res = "The initial map state is as follows:\n" + describe_obs(obs, target_positions)
    # import ipdb

    # ipdb.set_trace()
    plan_new = convert_gt_plan(plan)
    res = (
        res
        + "\n"
        + "The ground-truth plan is as follows:\n"
        + json.dumps(plan_new, indent=2)
    )
    return res


def describe_obs(obs, target_positions):
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

    object_positions = {}
    for obj_name, obj_info in obs["objects"].items():
        # object_positions[obj_name] = obj_info["xpos"][:2].tolist()
        obj_id = int(obj_name.split("_")[-1])
        object_positions[obj_id] = obj_info["xpos"][:2]

    robot_names = [x for x in obs.keys() if x.startswith("robot_")]
    robot_positions = {}
    for robot_name in robot_names:
        robot_id = int(robot_name.split("_")[-1])
        robot_positions[robot_id] = {
            "base_pos": obs[robot_name]["base_xpos"][:2],
            "arm_pos": obs[robot_name]["ee_xpos"][:2],
        }

    return Map2Text({}, object_positions, target_positions, robot_positions)


def convert_gt_plan(gt_plan: List[str]) -> List[Dict[str, str]]:
    """
    Convert each string in gt_plan (which may contain one or more
    commands separated by '\\n') into a dict mapping "Robot N" → formatted command.
    Returns a list of dicts, one per input string.
    """
    result: List[Dict[str, str]] = []
    for entry in gt_plan:
        step_dict: Dict[str, str] = {}
        # split into individual commands (handles single- or multi-line entries)
        for line in entry.splitlines():
            line = line.strip()
            if not line:
                continue
            # e.g. "robot_0: Move (0.55, 0.55) False"
            robot_tag, rest = line.split(": ", 1)
            idx = int(robot_tag.split("_", 1)[1])  # get the integer index
            # swap "("→"[" and ")"→"]"
            formatted = rest.replace("(", "[").replace(")", "]")
            step_dict[f"Robot {idx}"] = formatted
        result.append(step_dict)
    return result[0]
