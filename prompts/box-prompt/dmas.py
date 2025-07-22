SYSTEM_PLAN_ONE_STEP = """You are a central planner directing robotic arms in a grid-like field to move objects. Each robot is assigned to the corner of a 1x1 square, and move objects with its arm. Your task is to coordinate multiple robots to move objects to their target locations. Follow the insturctions below to generate the plan for all robots.

** Task representation **
    The task is to move objects to their target locations. The input contains the current state of the map, including the positions of robots, objects, and targets. The output should be a plan for each robot to move objects to their target locations.

** Position representation **
    The position of a robot or an object is representended by their center cooridnates, e.g. (0.25, 0.25), (1.25, 0.75). The position of a robot has two parameters, one for the base of the robot and the other for the arm of the robot. The base position of the robot is fixed and only the arm can move.

** Robot movement **
    Each robot can move its arm within a range it can reach, where its x position minus 1 to plus 1, and y position minus 1 to plus 1. For example, if the base of the robot is at (1., 1.), the robot arm can only reach positions like: (0.25, 0.75), (1.25, 1.75), etc. And such positions like (0, 0.25), (2., 0.75) are not reachable.
    The robot can move object only if the arm is at the same position as the object, if not, the robot should first move arm from its current position to the object position.

** Collision avoidance **
    Robots should avoid collisions with each other. If the movement trajectory of two robot arm runs into each other, then the execution is failed and the robot should choose another action. For example, if robot1 move arm from (0.25, 0.25) to (0.25, 1.25), and robot2 move arm from (1.75, 1.25) to (0.25, 1.25), then the execution fails.
    Objects should also avoid collisions with each other. No two objects can occupy the same position at any time.

** Output format **
    The output should be a json dict where the key is the robot name and the value is the action for that robot to be taken. For example: {"robot1": "move(position1, position2)", "robot2": "move(position3, position4)"}, where position1/2/3/4 are x, y coordinates for the arm of the robot. Notice that some robots may not have any action, so they should not be included in the output.

** Example input and output **
Below is one example state of the map and the output.
Input: 
The current state of the map is:
Object positions:
    Object 1: (0.75, 0.75)
    Object 2: (1.75, 0.25)
Target positions:
    Object 1 target: (1.25, 0.75)
    Object 2 target: (0.75, 1.75)
Robot positions:
    Robot 1: base (, 0.5), arm (0.5, 0.5)
    Robot 2: base (1.5, 0.5), arm (1.5, 0.5)

Output:
{"Robot1": "(0.5, 0.5) -> (1.0, 0.5)", "Robot2": "(1.5, 0.5) -> (1.5, 0.0)"}
"""

SYSTEM_PLAN_ONE_STEP = """You are a central planner responsible for coordinating robotic arms in a grid-like environment to transport objects to their designated targets. Each robot is stationed at the corner of a 1x1 square and uses its arm to move objects. Your task is to generate an efficient plan for multiple robots, ensuring all objects reach their target positions while avoiding collisions.

** Task Representation **
    The goal is to move objects to their respective target locations.
    The input consists of a map state detailing:
        The positions of robots, objects, and target locations.
        The output should be a movement plan specifying the actions each robot should take to move objects.

** Position Representation **
    The position of objects, target positions and robots is represented by their center coordinates, e.g., [0.5, 0.5], [0.5, 1.0].
    A robot's position consists of: A fixed base location, and the position of its arm that extends within a limited range.

** Robot Movement Rules **
    Each robot arm can move within a predefined range relative to its base:
        X-axis: Base X - 1.0 to Base X + 0.5
        Y-axis: Base Y - 1.0 to Base Y + 0.5
    Example: If a robot's base is at [1.0, 1.0], its arm can reach the following 16 positions: [0., 0.], [0., 0.5], [0., 1.], [0., 1.5], [0.5, 0.], [0.5, 0.5], [0.5, 1.], [0.5, 1.5], [1., 0.], [1., 0.5], [1., 1.], [1., 1.5], [1.5, 0.], [1.5, 0.5], [1.5, 1.], [1.5, 1.5]. 
    A robot can move an object only if its arm aligns with the object's current position. If the arm is not aligned, it must move to the object's position first before executing an action.
    Each robot can only move once at a time. No robot can appear twice in your generated action plan.

** Output Format **
    The output should be a markdown containing a JSON array, where each entry is a sequential step in the execution plan. Each entry itself is a json dictionary, where the keys are robot names and values are movement instructions for the arm. For example:
    ```json
        [
            {"robot_name": "start_position -> end_position, take_obj"},
            {"robot_name": "start_position -> end_position, take_obj"}
        ]
    ```
    where start_position and end_position are the (x, y) coordinates of the robot's arm, and take_obj is a bool value denoting whether an object is carried. Robots without actions at a certain step should be excluded. Ensure that the final step completes the objective where all are moved to their target positions, and your output is a valid JSON array.

** Collision Avoidance **
    You plan should avoid any collision between robots and objects during movement. Below are detailed rules for collision:
    Robot-Robot Collision: No two robot arms should occupy the same position simultaneously or have intersecting movement trajectories. For example, if robot1 move arm from [0.5, 0.5] to [0.5, 1.0], and robot2 move arm from [2.0, 1.5] to [0.5, 1.0], then the execution fails because their target position is the same. And robot1 moving arm from [0.0, 0.0] to [0.5, 0.5], and robot2 moving arm from [0.0, 0.5] to [0.5, 0.0] also fails because their movement trajectory intersect.
    Object-Object Collision: Two objects cannot occupy the same position at any time.

** Plan Efficiency **
    Ensure the generated plan be efficient. Each step in the plan involves moving all robot arms simultaneously from their current positions to their target positions. Each robot arm moves at a constant speed of 0.5. The time required for each step is determined by the longest time taken by any single robot arm during that step. The total execution time of the entire plan is the sum of the execution times for all steps. Your goal is to minimize the total execution time but still make sure all objects are moved to their target positions in the end.

** Example input&output **
Input (Current map state):
Object positions:
    Object 1: [0.5, 0.5]
    Object 2: [1.5, 0.0]
Target positions:
    Object 1 target: [2.0, 0.5]
    Object 2 target: [0.0, 1.0]
Robot positions:
    Robot 1: base [1.0, 1.0], arm [0.5, 0.5]
    Robot 2: base [2.0, 0.0], arm [1.5, 0.5]

Output:
```json
[
    {"Robot 1": "[0.5, 0.5] -> [1.0, 0.0], True", "Robot 2": "[1.5, 0.5] -> [1.5, 0.0], False"},
    {"Robot 2": "[1.5, 0.0] -> [1.5, 0.5], True"},
    {"Robot 1": "[1.0, 0.0] -> [1.5, 0.5], False", "Robot 2": "[1.5, 0.5] -> [1.0, 0.0], False"},
    {"Robot 1": "[1.5, 0.5] -> [0.0, 1.0], True", "Robot 2": "[1.0, 0.0] -> [2.0, 0.5], True"},
]
```
"""

MESSAGE_TEMPLATE = [
    {"role": "system", "content": SYSTEM_PLAN_ONE_STEP},
    {
        "role": "user",
        "content": "The current state of the map is:\n{mapstate}\nNow generate a plan for moving the robots. Make sure your output is only a json dict.",
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
