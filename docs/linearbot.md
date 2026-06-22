# LinearBot Setup Guide

LinearBot is a mobile bimanual robot (holonomic base + dual 6-DOF arms). Its assets
are hosted on HuggingFace at **[TreeePlanter/linearbot-assets](https://huggingface.co/datasets/TreeePlanter/linearbot-assets)**.

## 1. Get the robot assets

Downloads directly into `$MLSPACES_ASSETS_DIR/robots/linearbot` (or `~/robot_assets/robots/linearbot`
if the env var is not set). Set `MLSPACES_ASSETS_DIR` in your shell profile to wherever you keep
MlSpaces assets:

```bash
# ~/.bashrc or ~/.zshrc
export MLSPACES_ASSETS_DIR=/path/to/mlspaces_assets
```

```bash
bash scripts/assets/pull_linearbot.sh
```

## 2. Run the MuJoCo viewer

```bash
source /path/to/molmospaces/.venv/bin/activate
python scripts/view_linearbot.py
```

## 3. Install CuRobo (Linux + CUDA GPU only)

The `curobo` extra in `pyproject.toml` pulls in the Treeeplanter fork (`v1.0`) plus
`cuda-python` and `cuda-core`. You need the CUDA toolkit and torch with CUDA support
on the system before compiling.

Follow the CUDA + torch prerequisites from the main [README § Installing cuRobo](../README.md#installing-curobo),
then install the `curobo` extra into the uv environment:

```bash
uv pip install -e ".[mujoco,curobo]"
```

Verify:

```bash
python -c "import curobo; print('curobo ok')"
```

## 4. Run the CuRobo planner test (direct, requires GPU)

Requires a CUDA GPU. Plans random bimanual Cartesian goals in a loop and
streams the trajectory into the MuJoCo viewer.

```bash
python scripts/test_linearbot_curobo.py
```

## 5. Run the CuRobo planner via gRPC server/client

```bash
# Terminal 1 — start the server (loads CuRobo, warms up CUDA graphs)
python -m molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner_server \
    --robot-config $MLSPACES_ASSETS_DIR/robots/linearbot/curobo_config/linearbot_custom.yml \
    --port 10001 --warmup

# Terminal 2 — connect the client test
python scripts/test_linearbot_curobo_server.py --no-spawn --address localhost:10001
```

Or let the test script manage the server automatically:
```bash
python scripts/test_linearbot_curobo_server.py
```

## 6. Use LinearBot in your own code

```python
from pathlib import Path
from molmo_spaces.configs.robot_configs import LinearBotConfig
from molmo_spaces.robots.linearbot import LinearBot

ROBOT_DIR = Path(os.environ["MLSPACES_ASSETS_DIR"]) / "robots" / "linearbot"
cfg = LinearBotConfig(robot_dir=ROBOT_DIR)

# Add to a MuJoCo scene
from mujoco import MjSpec
spec = MjSpec()
LinearBot.add_robot_to_scene(cfg, spec, prefix="robot_0/",
                              pos=[0.0, 0.0, 0.12], quat=[1.0, 0.0, 0.0, 0.0])
model = spec.compile()
```

Motion planning (CuRobo v2, requires GPU):

```python
from molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner import (
    LinearBotCuroboPlanner, LinearBotCuroboPlannerConfig,
)

planner = LinearBotCuroboPlanner(LinearBotCuroboPlannerConfig(
    robot_config_path=str(ROBOT_DIR / "curobo_config" / "linearbot_custom.yml"),
))
planner.warmup()

# goal_poses: {frame_name: [x, y, z, qw, qx, qy, qz]}
traj, ok = planner.motion_plan(current_joint_positions, goal_poses={
    "gripper_R": [0.5, -0.3, 1.4, 1.0, 0.0, 0.0, 0.0],
    "gripper_L": [0.5,  0.3, 1.4, 1.0, 0.0, 0.0, 0.0],
})
```

Or via the gRPC client (server must be running):

```python
from molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner_client import (
    LinearBotCuroboClient,
)

client = LinearBotCuroboClient(address="localhost:10001")
traj, ok = client.motion_plan(current_joint_positions, goal_poses={...})
```

## Key conventions

| Thing | Value |
|---|---|
| Robot base position | `pos=[0, 0, 0.12]` (12 cm ground clearance for wheels) |
| Robot base orientation | `quat=[1, 0, 0, 0]` (identity — no rotation offset) |
| CuRobo FK frame | equals MuJoCo world frame exactly — no coordinate offset |
| Planned joints (DOF) | 16 (base xyz+θ, lift, 6×L arm, 6×R arm — gripper joints locked) |
| Tool frames | `["gripper_L", "gripper_R"]` |
| Pose format | `[x, y, z, qw, qx, qy, qz]` |

## Updating the assets

```bash
bash scripts/assets/push_linearbot.sh
```

Uploads everything in `$ROBOT_DIR` (default: `MLSPACES_ASSETS_DIR/robots/linearbot`) to HF.
