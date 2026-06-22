"""Test LinearBot CuRobo planner via the gRPC server/client protocol.

Identical behaviour to test_linearbot_curobo.py, but the planner runs in a
subprocess server; this script acts as the gRPC client.

Usage (single terminal — server is managed automatically):
    source /home/shuo/research/molmospaces/.venv/bin/activate
    python scripts/test_linearbot_curobo_server.py

Or start server manually and connect to it:
    # terminal 1
    python -m molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner_server \\
        --robot-config /path/to/linearbot_custom.yml --port 10001 --warmup
    # terminal 2
    python scripts/test_linearbot_curobo_server.py --address localhost:10001 --no-spawn
"""

import argparse
import subprocess
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjSpec
from pathlib import Path

from molmo_spaces.robots.linearbot import LinearBot
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotRobotView
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.configs.robot_configs import LinearBotConfig
from molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner_client import (
    LinearBotCuroboClient,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROBOT_DIR   = Path("/home/shuo/research/resources/mlspaces_assets/robots/linearbot")
CUROBO_YAML = ROBOT_DIR / "curobo_config" / "linearbot_custom.yml"
MJ_PREFIX   = "robot_0/"

ROBOT_BASE_Z = 0.12
FRAME_RGBA: dict[str, np.ndarray] = {
    "gripper_R": np.array([1.0, 0.2, 0.2, 0.4], dtype=np.float32),
    "gripper_L": np.array([0.2, 0.4, 1.0, 0.4], dtype=np.float32),
}
MARKER_SIZE = np.array([0.03, 0.03, 0.03], dtype=np.float64)


# ─── Server lifecycle ─────────────────────────────────────────────────────────

def spawn_server(port: int) -> subprocess.Popen:
    """Launch the planner server as a subprocess, return the Popen handle."""
    cmd = [
        sys.executable, "-m",
        "molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner_server",
        "--robot-config", str(CUROBO_YAML),
        "--port", str(port),
        "--warmup",
    ]
    print(f"Spawning server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_for_server(client: LinearBotCuroboClient, timeout: float = 120.0) -> None:
    """Poll health until the server responds or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = client.health()
            if resp.get("status") == "ok":
                print(f"Server ready — tool_frames={resp['tool_frames']}")
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


# ─── MuJoCo scene ─────────────────────────────────────────────────────────────

def build_mj_model(cfg: LinearBotConfig) -> mujoco.MjModel:
    spec = MjSpec()
    spec.option.gravity    = [0, 0, -9.81]
    spec.option.timestep   = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.01], rgba=[0.5, 0.5, 0.5, 1],
    )
    LinearBot.add_robot_to_scene(
        cfg, spec, prefix=MJ_PREFIX,
        pos=[0.0, 0.0, ROBOT_BASE_Z],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    LinearBot.apply_control_overrides(spec, cfg)
    model = spec.compile()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


# ─── Goal cube renderer ────────────────────────────────────────────────────────

def draw_goal_cubes(v: mujoco.viewer.Handle, goals: dict[str, list[float]]) -> None:
    """Draw target EE cubes in the viewer's user scene. CuRobo FK == MuJoCo world."""
    v.user_scn.ngeom = 0
    for frame, pose in goals.items():
        if v.user_scn.ngeom >= v.user_scn.maxgeom:
            break
        if frame not in FRAME_RGBA:
            continue
        pos  = np.array(pose[:3], dtype=np.float64)
        quat = np.array(pose[3:], dtype=np.float64)
        mat  = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        mujoco.mjv_initGeom(
            v.user_scn.geoms[v.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            MARKER_SIZE, pos, mat,
            FRAME_RGBA[frame].astype(np.float32),
        )
        v.user_scn.ngeom += 1


# ─── Trajectory → MuJoCo move-group targets ───────────────────────────────────

_MG_JOINTS: dict[str, list[str]] = {
    "base":      ["base_x", "base_y", "base_theta"],
    "lift":      ["lift_joint"],
    "right_arm": [f"joint{i}_R" for i in range(1, 7)],
    "left_arm":  [f"joint{i}_L" for i in range(1, 7)],
}
_JNAME_TO_MG: dict[str, tuple[str, int]] = {
    jn: (mg_id, i)
    for mg_id, jnames in _MG_JOINTS.items()
    for i, jn in enumerate(jnames)
}


def curobo_wp_to_mg_targets(
    wp: list[float],
    joint_names: list[str],
    controllers: dict[str, JointPosController],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {
        mg_id: ctrl.target.copy() for mg_id, ctrl in controllers.items()
    }
    for ci, jname in enumerate(joint_names):
        if jname in _JNAME_TO_MG:
            mg_id, local_i = _JNAME_TO_MG[jname]
            targets[mg_id][local_i] = wp[ci]
    return targets


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="localhost:10001",
                        help="gRPC server address (host:port)")
    parser.add_argument("--port", type=int, default=10001,
                        help="Port to use when spawning the server")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Do not spawn a server subprocess; connect to existing server")
    args = parser.parse_args()

    server_proc = None
    if not args.no_spawn:
        server_proc = spawn_server(args.port)
        args.address = f"localhost:{args.port}"

    client = LinearBotCuroboClient(address=args.address)

    try:
        print("Waiting for server...")
        wait_for_server(client, timeout=180.0)

        joint_names = client.joint_names()
        tool_frames = client.tool_frames()
        print(f"joint_names ({len(joint_names)}): {joint_names}")
        print(f"tool_frames: {tool_frames}\n")

        # --- MuJoCo setup ---
        mj_cfg = LinearBotConfig(robot_dir=ROBOT_DIR)
        model  = build_mj_model(mj_cfg)
        data   = mujoco.MjData(model)

        robot_view  = LinearBotRobotView(data, namespace=MJ_PREFIX)
        controllers = {
            mg_id: JointPosController(robot_view.get_move_group(mg_id))
            for mg_id in robot_view.move_group_ids()
        }

        for mg_id, qpos in mj_cfg.init_qpos.items():
            robot_view.get_move_group(mg_id).joint_pos = np.asarray(qpos)
        mujoco.mj_forward(model, data)
        for ctrl in controllers.values():
            ctrl.set_to_stationary()

        jname_to_qposadr = {
            model.joint(i).name.removeprefix(MJ_PREFIX): model.jnt_qposadr[i]
            for i in range(model.njnt)
        }
        HOME_QPOS = [
            float(data.qpos[jname_to_qposadr[jn]]) if jn in jname_to_qposadr else 0.0
            for jn in joint_names
        ]
        print(f"HOME_QPOS: {[round(v, 4) for v in HOME_QPOS]}")

        home_ee = client.fk(HOME_QPOS)
        print("Home EE poses (world frame):")
        for frame, pose in home_ee.items():
            print(f"  {frame}: xyz={np.round(pose[:3], 3)}")
        print()

        rng = np.random.default_rng(seed=0)

        _DELTA: dict[str, dict[str, tuple[float, float]]] = {
            "gripper_R": {"x": (-0.20, 0.20), "y": (-0.20, 0.05), "z": (-0.15, 0.20)},
            "gripper_L": {"x": (-0.20, 0.20), "y": (-0.05, 0.20), "z": (-0.15, 0.20)},
        }

        current_qpos: list[float] = list(HOME_QPOS)
        traj_waypoints: list[list[float]] = []
        wp_idx       = 0
        steps_per_wp = 80
        step_in_wp   = 0
        current_goals: dict[str, list[float]] = {}
        plan_count   = 0

        def plan_next() -> None:
            nonlocal traj_waypoints, wp_idx, step_in_wp, current_goals, current_qpos, plan_count
            plan_count += 1

            ee = client.fk(HOME_QPOS)
            goals: dict[str, list[float]] = {}
            for frame, bounds in _DELTA.items():
                p = list(ee[frame])
                p[0] += float(rng.uniform(*bounds["x"]))
                p[1] += float(rng.uniform(*bounds["y"]))
                p[2] += float(rng.uniform(*bounds["z"]))
                goals[frame] = p

            current_goals = goals

            print(f"[Plan {plan_count}] goals:")
            for frame, p in goals.items():
                print(f"  {frame}: xyz={np.round(p[:3], 3)}")

            t0 = time.perf_counter()
            traj, ok = client.motion_plan(current_qpos, goals)
            dt = time.perf_counter() - t0

            if not ok:
                print(f"  FAILED ({dt:.2f}s)\n")
                return

            traj_waypoints[:] = traj
            wp_idx       = 0
            step_in_wp   = 0
            current_qpos = list(traj[-1])
            print(f"  SUCCESS ({dt:.2f}s) — {len(traj)} waypoints\n")

        plan_next()

        step       = 0
        plan_every = int(8.0 / model.opt.timestep)

        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.distance  = 3.5
            v.cam.elevation = -20
            while v.is_running():
                if traj_waypoints and wp_idx < len(traj_waypoints):
                    targets = curobo_wp_to_mg_targets(
                        traj_waypoints[wp_idx], joint_names, controllers
                    )
                    for mg_id, tgt in targets.items():
                        controllers[mg_id].set_target(tgt)
                    step_in_wp += 1
                    if step_in_wp >= steps_per_wp:
                        step_in_wp = 0
                        wp_idx    += 1

                if step > 0 and step % plan_every == 0:
                    plan_next()

                for ctrl in controllers.values():
                    ctrl.robot_move_group.ctrl = ctrl.compute_ctrl_inputs()

                mujoco.mj_step(model, data)
                draw_goal_cubes(v, current_goals)
                v.sync()
                step += 1

    finally:
        client.close()
        if server_proc is not None:
            print("Terminating server subprocess...")
            server_proc.terminate()
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
