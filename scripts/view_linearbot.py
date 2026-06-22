"""LinearBot viewer with joint target/actual plotting for controller tuning.

    source /home/shuo/research/molmospaces/.venv/bin/activate
    python scripts/view_linearbot.py

Close the MuJoCo window to generate the per-joint target-vs-actual plot.
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mujoco import MjSpec
from pathlib import Path
from collections import defaultdict

from molmo_spaces.robots.linearbot import LinearBot
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotRobotView
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.configs.robot_configs import LinearBotConfig

ROBOT_DIR = Path("/home/shuo/research/resources/mlspaces_assets/robots/linearbot")
PREFIX = "robot_0/"

# Each entry holds (duration_s, targets_dict).
# Groups not mentioned keep their previous target.
# Lift range: [-0.729, 0.330] m
# Gripper: use LinearBot.gripper_targets(g), g=0.0 → open, g=1.0 → closed.
gripper = LinearBot.gripper_targets


def base_ramp(
    start: list[float], end: list[float], n: int = 10, dt: float = 0.3
) -> list[tuple[float, dict]]:
    """Interpolate base pose from start→end in n steps of dt seconds each."""
    return [
        (dt, {"base": np.array(p)})
        for p in np.linspace(start, end, n + 1)[1:]
    ]


_T0 = 0.0

SEQUENCE = [
    (2.0, {  # home — robot faces world +X
        "base":      np.array([0.0,  0.0,  _T0]),
        "right_arm": np.array([0.0, -1.047, -1.047, 0.0,  0.0,  0.0]),
        "left_arm":  np.array([0.0,  1.047,  1.047, 0.0,  0.0,  0.0]),
        "lift":      np.array([0.0]),
        **gripper(1.0),
    }),
    # *base_ramp([0, 0, _T0],             [1, 0, _T0]),             # drive forward 1 m (+X)
    # *base_ramp([1, 0, _T0],             [1, 1, _T0]),             # strafe left 1 m (+Y)
    # *base_ramp([1, 1, _T0],             [1, 1, _T0 + np.pi/2]),   # rotate 90° CCW
    # *base_ramp([1, 1, _T0 + np.pi/2],  [0, 0, _T0]),             # return to origin
    (3.0, {  # lift up + reach forward
        "right_arm": np.array([0.0, -0.5, -0.5, 0.0,  0.8,  0.0]),
        "left_arm":  np.array([0.0,  0.5,  0.5, 0.0, -0.8,  0.0]),
        "lift":      np.array([0.25]),
    }),
    (3.0, {  # open grippers fully
        **gripper(0.0),
    }),
    (3.0, {  # wide arm spread + wrist rotation
        "right_arm": np.array([ 1.2, -0.8, -0.5,  0.5,  0.3,  1.0]),
        "left_arm":  np.array([-1.2,  0.8,  0.5, -0.5, -0.3, -1.0]),
        "lift":      np.array([0.1]),
    }),
    (3.0, {  # close grippers + high lift
        "right_arm": np.array([0.0, -0.3, -0.8, 0.0,  1.2,  0.5]),
        "left_arm":  np.array([0.0,  0.3,  0.8, 0.0, -1.2, -0.5]),
        "lift":      np.array([0.3]),
        **gripper(1.0),
    }),
    (3.0, {  # half-open grippers + wrist sweep
        "right_arm": np.array([0.5, -1.0, -1.0,  1.5,  1.0, -1.5]),
        "left_arm":  np.array([-0.5, 1.0,  1.0, -1.5, -1.0,  1.5]),
        "lift":      np.array([0.2]),
        **gripper(0.5),
    }),
    (3.0, {  # open grippers at low lift
        "lift":      np.array([0.05]),
        **gripper(0.0),
    }),
    (2.0, {  # back to home
        "right_arm": np.array([0.0, -1.047, -1.047, 0.0,  0.0,  0.0]),
        "left_arm":  np.array([0.0,  1.047,  1.047, 0.0,  0.0,  0.0]),
        "lift":      np.array([0.0]),
        **gripper(1.0),
    }),
]

PLOT_GROUPS = ["base", "lift", "right_arm", "left_arm", "right_gripper", "left_gripper"]

# Mocap target cubes: show FK positions at the current joint targets
TARGET_BODIES = {
    "target_gripper_R": ("robot_0/gripper_R", [1.0, 0.2, 0.2, 0.5]),
    "target_gripper_L": ("robot_0/gripper_L", [0.2, 0.4, 1.0, 0.5]),
    "target_base":      ("robot_0/base",       [0.2, 0.8, 0.2, 0.5]),
}


def build_model(cfg: LinearBotConfig) -> mujoco.MjModel:
    spec = MjSpec()
    spec.option.gravity = [0, 0, -9.81]
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.1],
        rgba=[0.5, 0.5, 0.5, 1],
    )
    LinearBot.add_robot_to_scene(cfg, spec, prefix=PREFIX, pos=[0, 0, 0.12],
                                 quat=[1, 0, 0, 0])
    LinearBot.apply_control_overrides(spec, cfg)

    for name, (_, rgba) in TARGET_BODIES.items():
        body = spec.worldbody.add_body()
        body.name = name
        body.mocap = True
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.04, 0.04, 0.04]
        geom.rgba = rgba
        geom.contype = 0
        geom.conaffinity = 0

    model = spec.compile()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def plot_results(
    times: list[float],
    actuals: dict[str, list],
    targets_log: dict[str, list],
) -> None:
    t = np.array(times)

    # Build subplot grid: one column per move group, rows = max joints in any group
    n_groups = len(PLOT_GROUPS)
    n_rows = max(len(actuals[g][0]) for g in PLOT_GROUPS if actuals[g])
    fig, axes = plt.subplots(
        n_rows, n_groups,
        figsize=(4 * n_groups, 2.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    fig.suptitle("LinearBot — joint target (dashed) vs actual (solid)", fontsize=12)

    for col, mg_id in enumerate(PLOT_GROUPS):
        act_arr = np.array(actuals[mg_id])     # (T, n_joints)
        tgt_arr = np.array(targets_log[mg_id]) # (T, n_joints)
        n_joints = act_arr.shape[1]
        for row in range(n_rows):
            ax = axes[row][col]
            if row < n_joints:
                ax.plot(t, act_arr[:, row], color="steelblue", linewidth=1.2, label="actual")
                ax.step(t, tgt_arr[:, row], color="tomato", linewidth=1.0,
                        linestyle="--", where="post", label="target")
                _BASE_LABELS = ["x (m)", "y (m)", "θ (rad)"]
                ylabel = _BASE_LABELS[row] if mg_id == "base" else f"j{row + 1} (rad)"
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(mg_id, fontsize=9, fontweight="bold")
                    ax.legend(fontsize=7, loc="upper right")
            else:
                ax.set_visible(False)
        axes[n_rows - 1][col].set_xlabel("time (s)", fontsize=8)

    plt.tight_layout()
    out = Path("linearbot_joint_tracking.png")
    fig.savefig(out, dpi=150)
    print(f"Plot saved → {out.resolve()}")
    plt.show()


def main() -> None:
    cfg = LinearBotConfig(robot_dir=ROBOT_DIR)
    model = build_model(cfg)
    data = mujoco.MjData(model)

    robot_view = LinearBotRobotView(data, namespace=PREFIX)
    controllers: dict[str, JointPosController] = {
        mg_id: JointPosController(robot_view.get_move_group(mg_id))
        for mg_id in robot_view.move_group_ids()
    }

    # Set initial pose and seed targets from it
    for mg_id, qpos in cfg.init_qpos.items():
        robot_view.get_move_group(mg_id).joint_pos = np.asarray(qpos)
    mujoco.mj_forward(model, data)
    for ctrl in controllers.values():
        ctrl.set_to_stationary()

    # Secondary data + view used for FK at the target state (not the current state)
    target_data = mujoco.MjData(model)
    target_robot_view = LinearBotRobotView(target_data, namespace=PREFIX)

    # Resolve IDs once
    mocap_ids: dict[str, int] = {}
    src_body_ids: dict[str, int] = {}
    for mocap_name, (src_body, _) in TARGET_BODIES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        mocap_ids[mocap_name] = model.body_mocapid[bid]
        src_body_ids[mocap_name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, src_body)

    def update_target_mocap() -> None:
        """Run FK at the current controller targets and move the marker cubes."""
        for mg_id, ctrl in controllers.items():
            target_robot_view.get_move_group(mg_id).joint_pos = ctrl.target
        mujoco.mj_forward(model, target_data)
        for mocap_name, mid in mocap_ids.items():
            bid = src_body_ids[mocap_name]
            data.mocap_pos[mid]  = target_data.xpos[bid]
            data.mocap_quat[mid] = target_data.xquat[bid]  # wxyz

    update_target_mocap()  # initialise markers at home

    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")
    for mg_id in PLOT_GROUPS:
        print(f"  ctrl_range {mg_id}: {controllers[mg_id].ctrl_range.tolist()}")

    # Build phase schedule (cumulative step counts)
    dt = model.opt.timestep
    phase_ends = []
    cumulative = 0
    for duration, _ in SEQUENCE:
        cumulative += int(duration / dt)
        phase_ends.append(cumulative)

    # Logging buffers
    log_every = 10  # log every 10 ms
    times: list[float] = []
    actuals: dict[str, list] = defaultdict(list)
    targets_log: dict[str, list] = defaultdict(list)

    step = 0
    current_phase = -1

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.5
        v.cam.elevation = -20
        while v.is_running():
            # Advance phase
            phase = next((i for i, end in enumerate(phase_ends) if step < end), len(SEQUENCE) - 1)
            if phase != current_phase:
                current_phase = phase
                _, tgt = SEQUENCE[phase]
                for mg_id, target in tgt.items():
                    controllers[mg_id].set_target(target)
                update_target_mocap()
                t_sim = step * dt
                print(f"t={t_sim:.2f}s → phase {phase}: {list(tgt.keys())}")

            # Apply controls
            for ctrl in controllers.values():
                ctrl.robot_move_group.ctrl = ctrl.compute_ctrl_inputs()

            mujoco.mj_step(model, data)

            # Log
            if step % log_every == 0:
                times.append(step * dt)
                for mg_id in PLOT_GROUPS:
                    mg = robot_view.get_move_group(mg_id)
                    actuals[mg_id].append(mg.joint_pos.copy())
                    targets_log[mg_id].append(controllers[mg_id].target.copy())

            v.sync()
            step += 1

    if times:
        plot_results(times, actuals, targets_log)


if __name__ == "__main__":
    main()
