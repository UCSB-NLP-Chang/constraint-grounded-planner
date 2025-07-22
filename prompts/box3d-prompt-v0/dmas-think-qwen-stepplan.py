SYSTEM_PLAN_ONE_STEP = """You are a central planner responsible for coordinating robotic arms in a grid-like environment to transport objects to their designated targets. Each robot is stationed at the corner of a 1.1x1.1 square and uses its arm to move objects. Your task is to generate an efficient and collision-free next-step movement plan plan for controlling these robots, targeting at moving all objects to their target positions.


## Task Description:
*Task Representation:*
* Objective: Move all objects to their specified target locations safely and efficiently.
* Input: A detailed map state containing positions of robots, objects, and target locations.
* Output: A precise movement plan specifying each robot arm's actions for moving objects.


*Position Representation:*
* All positions (robots, objects, targets) are given by their center coordinates, e.g., [0.55, 1.65], [2.75, 0.55].
* Robots have a fixed base location and an extendable arm with a limited reach range.


*Movement Rules:*
Your generated movement must strictly consider the reachability of each robot arm, detaild rule in following:
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
* Think carefully and try to identify potential collisions explicitly in the analysis and explain how you avoid them.
* Highlight your reasoning for movement choices, considering efficiency and collision avoidance.


*Movement Plan (Output):*
* Your generated step of the movement plan should be in markdown format and contain a JSON dictionary, with robot names as keys and their movement instructions as values, structured as follows:
```json
{
    "robot_name1": "Move end_position, move_object",
    "robot_name2": "Move end_position, move_object"
},
```
* *end_position* represent the target *[x, y]* coordinates of the robot arm end point of the movement around circular path. Note that only the arm moves while its base remains fixed.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* Robots without actions in a certain step should not be included.
* One robot can only be moved once in each step, which means that no repeated keys are allowed in the same step.
Ensure your output forms a valid JSON dictionary of next-step plan.


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


## Plan Efficiency Considerations:
* The exeuction time of your plan involves simultaneous robot arm movements from their current positions to specified target positions.
* Each robot arm moves at a constant speed of 0.5 units/time.
* The duration of the plan is determined by the longest single-arm movement within it.
* You should aim to minimize the execution time while ensuring collision-free movements and successful object placements.


## Example Input & Output:
* Input (Current environment):
Object positions:
    Object 1: [0.55, 1.65]
Target positions:
    Object 1 target: [1.65, 0.55]
Robot positions:
    Robot 1: base [1.1, 1.1], arm [1.24, 0.61]
    Robot 2: base [1.1, 2.2], arm [1.24, 1.71]

* Output:
<think> Let's understand the scenerio ... </think>
```json
{
  "Robot 1": "Move [0.55, 1.65] False"
}
```
"""


MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": SYSTEM_PLAN_ONE_STEP,
    },
    {
        "role": "user",
        "content": (
            "Given the information above, now consider following environtment:\nInput:\n{mapstate}\nGenerate the next-step plan for moving the robots."
        ),
    },
]


# def convert_tup(tup):
#     # return f"[{tup[0]:.2f}, {tup[1]:.2f}]"
#     return f"[{round(tup[0], 2)}, {round(tup[1], 2)}]"


def convert_tup(tup):
    # return f"[{tup[0]:.2f}, {tup[1]:.2f}]"
    return f"[{round(float(tup[0]), 2)}, {round(float(tup[1]), 2)}]"


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
            f"\tRobot {objid}: base: {objpos[0]}, arm: {objpos[1]}"
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
