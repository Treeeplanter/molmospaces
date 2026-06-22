"""
LinearBot robot view implementation.

The LinearBot is a mobile bimanual robot with:
- Two 6-DOF revolute arms with 2-DOF prismatic grippers
- A 1-DOF prismatic lift column
- A mobile base with holonomic x/y/theta virtual joint control

Kinematic tree (URDF):
  base
  └── lift_joint (prismatic) → lift_link
      ├── [fixed joint0_R] → arm_base_R → joint1_R..joint6_R → gripper_R → joint7_R/8_R → tip_left_R/tip_right_R
      └── [fixed joint0_L] → arm_base_L → joint1_L..joint6_L → gripper_L → joint7_L/8_L → tip_left_L/tip_right_L
  base (also parent of 4 swerve wheels: fl/fr/rl/rr _steer_joint + _wheel_joint)
"""

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class LinearBotArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """6-DOF revolute arm of the LinearBot."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        suffix = "_R" if side == "right" else "_L"
        joint_ids = [model.joint(f"{namespace}joint{i}{suffix}").id for i in range(1, 7)]
        act_ids = [model.actuator(f"{namespace}joint{i}{suffix}_act").id for i in range(1, 7)]
        ee_body_name = f"{namespace}gripper_R" if side == "right" else f"{namespace}gripper_L"
        root_body_name = f"{namespace}arm_link1_R" if side == "right" else f"{namespace}arm_link1_L"
        self._ee_body_id = model.body(ee_body_name).id
        root_body_id = model.body(root_body_name).id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_body_id

    @property
    def leaf_frame_type(self) -> Literal["body"]:
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class LinearBotGripperGroup(MJCFFrameMixin, GripperGroup):
    """2-DOF prismatic gripper of the LinearBot."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        suffix = "_R" if side == "right" else "_L"
        joint_ids = [model.joint(f"{namespace}joint{j}{suffix}").id for j in [7, 8]]
        act_ids = [model.actuator(f"{namespace}joint{j}{suffix}_act").id for j in [7, 8]]
        palm_name = f"{namespace}gripper_R" if side == "right" else f"{namespace}gripper_L"
        # Track first finger tip as EE frame
        finger_name = f"{namespace}tip_left_R" if side == "right" else f"{namespace}tip_left_L"
        self._ee_body_id = model.body(finger_name).id
        root_body_id = model.body(palm_name).id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_body_id

    @property
    def leaf_frame_type(self) -> Literal["body"]:
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)

    def set_gripper_ctrl_open(self, open: bool) -> None:
        limits = self.ctrl_limits
        self.ctrl = limits[:, 1] if open else limits[:, 0]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        limits = self.joint_pos_limits
        return float(limits[:, 0].sum()), float(limits[:, 1].sum())

    @property
    def inter_finger_dist(self) -> float:
        return float(np.abs(self.joint_pos).sum())


class LinearBotLiftGroup(SimplyActuatedMoveGroup):
    """1-DOF prismatic lift column of the LinearBot (lift_joint)."""

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}lift_joint").id]
        act_ids = [model.actuator(f"{namespace}lift_joint_act").id]
        root_body_id = model.body(f"{namespace}base").id
        self._lift_body_id = model.body(f"{namespace}lift_link").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._lift_body_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)

    def get_jacobian(self) -> np.ndarray:
        import mujoco

        J = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, J[:3], J[3:], self._lift_body_id)
        return J


class LinearBotHoloBaseGroup(HoloJointsRobotBaseGroup):
    """Virtual holonomic base for planar x/y/theta control."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        world_site_id = model.site(f"{namespace}world").id
        base_site_id = model.site(f"{namespace}base_site").id
        joint_ids = [model.joint(f"{namespace}base_{ax}").id for ax in ["x", "y", "theta"]]
        act_ids = [model.actuator(f"{namespace}base_{ax}_act").id for ax in ["x", "y", "theta"]]
        root_body_id = model.body(f"{namespace}base").id
        super().__init__(mj_data, world_site_id, base_site_id, joint_ids, act_ids, root_body_id)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()


class LinearBotRobotView(RobotView):
    """Complete LinearBot robot view assembling all move groups."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = LinearBotHoloBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "lift": LinearBotLiftGroup(mj_data, base, namespace=namespace),
            "left_arm": LinearBotArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": LinearBotArmGroup(mj_data, "right", base, namespace=namespace),
            "left_gripper": LinearBotGripperGroup(mj_data, "left", base, namespace=namespace),
            "right_gripper": LinearBotGripperGroup(mj_data, "right", base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "linearbot"

    @property
    def base(self) -> LinearBotHoloBaseGroup:
        return self.get_move_group("base")
