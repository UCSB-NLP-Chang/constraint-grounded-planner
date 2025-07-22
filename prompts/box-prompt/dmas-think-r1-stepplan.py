SYSTEM_PLAN_ONE_STEP = """You are a central planner responsible for coordinating robotic arms in a grid-like environment to transport objects to their designated targets. Each robot is stationed at the corner of a 1x1 square and uses its arm to move objects. Your task is to generate an efficient and collision-free next-step movement plan plan for controlling these robots, targeting at moving all objects to their target positions.


## Task Description:
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
    * If a robot's base is [1.0, 1.0], its arm can reach [0.25, 0.75] or [1.25, 1.75], but not [0, 0.25] or [2.25, 1.75] because 2.25 - 1.0 = 1.25 > 1.0 and 0 - 1.0 = -1.0 <= -1.
    * Robots may move an object only if their arm aligns exactly with the object's current position, and if explicitly indicated in the action (move_object: True).
    * If a robot needs to move an object within its range and the arm is not aligned with the object, the robot should first move its arm to the position of that object.


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
    "robot_name1": "start_position -> end_position, move_object"
    "robot_name2": "start_position -> end_position, move_object",
},
```
* *start_position* and *end_position* represent the *[x, y]* coordinates of the robot arm before and after the movement. Note that only the arm moves while base remains fixed.
* *move_object* is a boolean indicating whether the robot moves an object (*True*) or simply moves its arm without carrying an object (*False*).
* Robots without actions in a certain step should not be included.
* One robot can only be moved once in each step, which means that no repeated keys are allowed in the same step.
Ensure your output forms a valid JSON dictionary of next-step plan.



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
* The exeuction time of your plan involves simultaneous robot arm movements from their current positions to specified target positions.
* Each robot arm moves at a constant speed of 0.5 units/time.
* The duration of the plan is determined by the longest single-arm movement within it.
* You should aim to minimize the execution time while ensuring collision-free movements and successful object placements.


## Example Input & Output:
* Input (Current environment):
Object positions:
    Object 1: [0.75, 0.75]
    Object 2: [1.75, 0.25]
Target positions:
    Object 1 target: [2.25, 0.75]
    Object 2 target: [0.25, 1.25]
Robot positions:
    Robot 1: base [1.0, 1.0], arm [0.75, 0.75]
    Robot 2: base [2.0, 0.0], arm [1.75, 0.75]

* Output:
<think> Let's understand the scenerio ... </think>
```json
{"Robot 1": "[0.75, 0.75] -> [1.25, 0.25], True", "Robot 2": "[1.75, 0.75] -> [1.75, 0.25], False"}
```
"""


MESSAGE_TEMPLATE = [
    {
        "role": "user",
        "content": (
            SYSTEM_PLAN_ONE_STEP
            + "\n\n"
            + "The current state of the environment is:\n{mapstate}\nNow generate the next-step plan for moving the robots. Make sure your output contains the thinking and the markdown json dict."
        ),
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
