"""Test LinearBot CuRobo v2 planner in a minimal MuJoCo scene.

    source /home/shuo/research/molmospaces/.venv/bin/activate
    python scripts/test_linearbot_curobo.py

What it does:
  1. Builds CuRobo v2 MotionPlanner from the existing robot YAML.
  2. Opens a MuJoCo viewer with LinearBot at home pose.
  3. Runs FK to get the home gripper poses.
  4. Plans both grippers to independent random 3-D goals, repeated every 8 s.
  5. Streams the planned trajectory into the MuJoCo controllers.
  6. Renders target EE poses as transparent red (right) / blue (left) cubes via user_scn.

Frame convention:
  Robot is placed at identity rotation. CuRobo FK outputs poses in the URDF base frame,
  which equals the MuJoCo world frame (both use the same world-aligned base axes).
  The only conversion needed is adding ROBOT_BASE_Z to the Z coordinate.
"""

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
from molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner import (
    LinearBotCuroboPlanner,
    LinearBotCuroboPlannerConfig,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROBOT_DIR   = Path("/home/shuo/research/resources/mlspaces_assets/robots/linearbot")
CUROBO_YAML = ROBOT_DIR / "curobo_config" / "linearbot_custom.yml"
MJ_PREFIX   = "robot_0/"

# ─── Robot placement ──────────────────────────────────────────────────────────
# Wheel joints sit ~0.12 m below the base body → use that as ground clearance.
ROBOT_BASE_Z = 0.12
# Goal marker colours (RGBA, alpha 0.4 = transparent)
FRAME_RGBA: dict[str, np.ndarray] = {
    "gripper_R": np.array([1.0, 0.2, 0.2, 0.4], dtype=np.float32),
    "gripper_L": np.array([0.2, 0.4, 1.0, 0.4], dtype=np.float32),
}
MARKER_SIZE = np.array([0.03, 0.03, 0.03], dtype=np.float64)


# ─── Frame conversion helpers ─────────────────────────────────────────────────

def urdf_pose_to_world(pose_7d: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CuRobo 7D pose [x,y,z,qw,qx,qy,qz] to MuJoCo world frame.

    CuRobo FK outputs positions in the URDF base frame, which equals the MuJoCo
    world frame exactly (verified: MuJoCo xpos matches CuRobo FK at the same config).
    ROBOT_BASE_Z only lifts the visual mesh above the floor — it is not a FK offset.
    """
    pos  = np.array(pose_7d[:3], dtype=np.float64)
    quat = np.array(pose_7d[3:], dtype=np.float64)
    return pos, quat


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
    """Draw target EE cubes in the viewer's user scene (no model bodies needed)."""
    v.user_scn.ngeom = 0
    for frame, pose in goals.items():
        if v.user_scn.ngeom >= v.user_scn.maxgeom:
            break
        pos_w, quat_w = urdf_pose_to_world(pose)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat_w)
        mujoco.mjv_initGeom(
            v.user_scn.geoms[v.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            MARKER_SIZE,
            pos_w,
            mat,
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
# Gripper joints intentionally excluded: joint7_L has an inverted sign convention
# ([0, +0.0475]) vs all other finger joints (negative range).
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
    """Split a flat CuRobo waypoint into per-move-group target arrays (no grippers)."""
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
    # --- MuJoCo setup ---
    mj_cfg = LinearBotConfig(robot_dir=ROBOT_DIR)
    model  = build_mj_model(mj_cfg)
    data   = mujoco.MjData(model)

    robot_view  = LinearBotRobotView(data, namespace=MJ_PREFIX)
    controllers = {
        mg_id: JointPosController(robot_view.get_move_group(mg_id))
        for mg_id in robot_view.move_group_ids()
    }

    # Initialise to home pose
    for mg_id, qpos in mj_cfg.init_qpos.items():
        robot_view.get_move_group(mg_id).joint_pos = np.asarray(qpos)
    mujoco.mj_forward(model, data)
    for ctrl in controllers.values():
        ctrl.set_to_stationary()

    # --- CuRobo planner ---
    planner_cfg = LinearBotCuroboPlannerConfig(robot_config_path=str(CUROBO_YAML))
    planner     = LinearBotCuroboPlanner(planner_cfg)
    print("Warming up CuRobo CUDA graphs...")
    planner.warmup()
    print("Done.\n")

    # Build HOME_QPOS dynamically from the MuJoCo state so joint ordering is guaranteed.
    jname_to_qposadr = {
        model.joint(i).name.removeprefix(MJ_PREFIX): model.jnt_qposadr[i]
        for i in range(model.njnt)
    }
    HOME_QPOS = [
        float(data.qpos[jname_to_qposadr[jn]]) if jn in jname_to_qposadr else 0.0
        for jn in planner.joint_names
    ]
    print(f"CuRobo joint_names ({len(planner.joint_names)}): {planner.joint_names}")
    print(f"HOME_QPOS: {[round(v, 4) for v in HOME_QPOS]}\n")

    # Print home EE poses for reference
    home_ee = planner.fk(HOME_QPOS)
    print("Home EE poses (URDF frame):")
    for frame, pose in home_ee.items():
        w_pos, _ = urdf_pose_to_world(pose)
        print(f"  {frame}: urdf_xyz={np.round(pose[:3], 3)}  world_xyz={np.round(w_pos, 3)}")
    print()

    rng = np.random.default_rng(seed=0)

    _DELTA: dict[str, dict[str, tuple[float, float]]] = {
        "gripper_R": {"x": (-0.20,  0.20), "y": (-0.20,  0.05), "z": (-0.15, 0.20)},
        "gripper_L": {"x": (-0.20,  0.20), "y": (-0.05,  0.20), "z": (-0.15, 0.20)},
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

        home_ee = planner.fk(HOME_QPOS)
        goals: dict[str, list[float]] = {}
        for frame, bounds in _DELTA.items():
            p = list(home_ee[frame])
            p[0] += float(rng.uniform(*bounds["x"]))
            p[1] += float(rng.uniform(*bounds["y"]))
            p[2] += float(rng.uniform(*bounds["z"]))
            goals[frame] = p

        current_goals = goals

        print(f"[Plan {plan_count}] random goals:")
        for frame, p in goals.items():
            w_pos, _ = urdf_pose_to_world(p)
            print(f"  {frame}: world_xyz={np.round(w_pos, 3)}")

        t0 = time.perf_counter()
        traj, ok = planner.motion_plan(current_qpos, goals)
        dt = time.perf_counter() - t0

        if not ok:
            print(f"  FAILED ({dt:.2f}s)")
            print(planner.diagnose_collision(current_qpos))
            print()
            return

        traj_waypoints[:] = traj
        wp_idx        = 0
        step_in_wp    = 0
        current_qpos  = list(traj[-1])
        print(f"  SUCCESS ({dt:.2f}s) — {len(traj)} waypoints\n")

    # Plan before opening viewer
    plan_next()

    step       = 0
    plan_every = int(8.0 / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance  = 3.5
        v.cam.elevation = -20
        while v.is_running():
            # Stream trajectory
            if traj_waypoints and wp_idx < len(traj_waypoints):
                targets = curobo_wp_to_mg_targets(
                    traj_waypoints[wp_idx], planner.joint_names, controllers
                )
                for mg_id, tgt in targets.items():
                    controllers[mg_id].set_target(tgt)
                step_in_wp += 1
                if step_in_wp >= steps_per_wp:
                    step_in_wp = 0
                    wp_idx    += 1

            # Re-plan after dwell period
            if step > 0 and step % plan_every == 0:
                plan_next()

            # Apply PD controllers
            for ctrl in controllers.values():
                ctrl.robot_move_group.ctrl = ctrl.compute_ctrl_inputs()

            mujoco.mj_step(model, data)
            draw_goal_cubes(v, current_goals)
            v.sync()
            step += 1


if __name__ == "__main__":
    main()
