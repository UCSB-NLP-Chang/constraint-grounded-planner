SYSTEM_PLAN_ONE_STEP = """You are a central planner responsible for coordinating robotic arms in a grid-like environment to transport objects to their designated targets. Each robot is stationed at the corner of a 1x1 square and uses its arm to move objects. Your task is to generate an efficient plan and collision-free plan for multiple robots, ensuring all objects reach their target positions after the plan is fully executed.

** Task Representation **
    The goal is to move objects to their respective target locations.
    The input consists of a map state detailing:
        The positions of robots, objects, and target locations.
        The output should be a movement plan specifying the actions each robot should take to move objects.

** Position Representation **
    The position of objects, target positions and robots is represented by their center coordinates, e.g., [0.25, 0.25], [0.75, 1.25].
    A robot's position consists of: A fixed base location, and the position of its arm that extends within a limited range.

** Movement Rules **
    Each robot arm can move within a predefined range relative to its base, and the boundary is exclusive:
        X-axis: Base X - 1.0 to Base X + 1.0
        Y-axis: Base Y - 1.0 to Base Y + 1.0
    Example: If a robot's base is at [1.0, 1.0], its arm can reach the following positions: [0.25, 0.75], [1.25, 1.75]. But such positions like [0, 0.25], [2., 0.75] are not reachable.
    A robot can move an object only if its arm aligns exactly with the object's current position and the action explicitly specifies moving the object (move_object: True).

** Output Format **
    Think carefully for the task. Show your work enclosed in <think> </think> and then generate current step of the movement plan. It should be a markdown containing a JSON dict, where the keys are robot names and values are movement instructions for the arm in current step. For example:
    ```json
        {"robot_name": "start_position -> end_position, move_object"},
    ```
    where start_position and end_position are the [x, y] coordinates of the robot's arm, and move_object is a boolean value denoting whether an object is carried. If move_object is True, the robot moves the object from start_position to end_position if there is an object at start_position, otherwise only the arm moves.
    Robots without actions at a certain step should be excluded. Ensure that the final step completes the objective where all are moved to their target positions, and your plan is a valid JSON array.

** Collision Avoidance **
    Your plan should avoid any collision between robots and objects during movement. Below are detailed rules for collision:
    Robot-Robot Collision: No two robot arms should occupy the same position simultaneously or have intersecting movement trajectories. For example, if robot1 move arm from [0.75, 0.75] to [0.75, 1.25], and robot2 move arm from [2.25, 1.75] to [0.75, 1.25], then the execution fails because their target position is the same. And robot1 moving arm from [0.25, 0.25] to [0.75, 0.75], and robot2 moving arm from [0.25, 0.75] to [0.75, 0.25] also fails because their movement trajectory intersect.
    Object-Object Collision: Two objects cannot occupy the same position at any time.

** Plan Efficiency **
    Ensure the generated movement plan be efficient. Each step in the plan involves moving all robot arms simultaneously from their current positions to their target positions. Each robot arm moves at a constant speed of 0.5. The time required for each step is determined by the longest time taken by any single robot arm during that step. The total execution time of the entire plan is the sum of the execution times for all steps. Your goal is to minimize the total execution time but still make sure all objects are moved to their target positions in the end.

** Example input&output **
Input (Current map state):
Object positions:
    Object 1: [0.75, 0.75]
    Object 2: [1.75, 0.25]
Target positions:
    Object 1 target: [2.25, 0.75]
    Object 2 target: [0.25, 1.25]
Robot positions:
    Robot 1: base [1.0, 1.0], arm [0.75, 0.75]
    Robot 2: base [2.0, 0.0], arm [1.75, 0.75]

Output:
<think> Let's understand the scenerio ... </think>
```json
{"Robot 1": "[0.75, 0.75] -> [1.25, 0.25], True", "Robot 2": "[1.75, 0.75] -> [1.75, 0.25], False"}
```
"""


MESSAGE_TEMPLATE = [
    {"role": "system", "content": SYSTEM_PLAN_ONE_STEP},
    {
        "role": "user",
        "content": "The current state of the map is:\n{mapstate}\nNow generate current step plan for moving the robots. Make sure your output contains the thinking and the markdown json dict.",
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
