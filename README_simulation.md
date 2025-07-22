# Box3D Simulation Quickstart

This repo includes a MuJoCo-based 3D boxnet environment. The snippet below starts a single Box3D simulation, runs one movement step, and writes a short video to `debug/execute.mp4`.

## Setup
```bash
conda create -n rob python=3.10 -y
pip install -e .
```

## Run a 3D simulation and move a robot
```python
from simulation.simenv.box3d.box3d import Box3DEnv

env = Box3DEnv(grid_n=3, grid_m=3, num_objects=1, robot_mode="full", do_render=True)
# do_render=False if not wanting to render video
env.reset()

print("robots:", env.env.robot_names)
base = env.get_base_pos("robot_0")
target = (base[0] + 0.2, base[1] + 0.2)

action = f"robot_0: Move ({target[0]:.2f}, {target[1]:.2f}) False"
result = env.simulate_one_step(action)
print(result)
PY
```

Notes:
- Action format: `robot_X: Move (x, y) False` (use `True` only if the robot is carrying an object).
- The rendered video is saved to `debug/execute.mp4`.

## Multi-step / multi-robot actions
You can pass multiple lines (one per robot) or a list of step strings:

```python
steps = [
    "robot_0: Move (0.10, 0.10) False\nrobot_1: WAIT",
    "robot_0: Move (0.20, 0.20) False\nrobot_1: WAIT",
]
result = env.simulate_full_step(steps)
print(result)
```
