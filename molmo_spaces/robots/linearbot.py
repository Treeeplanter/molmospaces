from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotRobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class LinearBot(Robot):
    """LinearBot: mobile bimanual robot with prismatic lift and holonomic base.

    Move groups:
      base          - holonomic x/y/theta virtual joints
      lift          - prismatic lift column (lift_joint)
      left_arm      - 6-DOF left arm (joint1_L..joint6_L)
      right_arm     - 6-DOF right arm (joint1_R..joint6_R)
      left_gripper  - 2-DOF left gripper (joint7_L, joint8_L)
      right_gripper - 2-DOF right gripper (joint7_R, joint8_R)

    The MJCF (linearbot_holobase.xml) already embeds the robot_0/ namespace on all
    body/joint/site/actuator names.  attach_body() automatically transfers the arm,
    gripper, and lift actuators.  Only the world-reference site and the three holonomic
    base actuators (which target that site) are added programmatically, because they
    live at worldbody scope rather than inside the attached body.
    """

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)
        self._namespace = self.exp_config.robot_config.robot_namespace
        self._robot_view = LinearBotRobotView(mj_data, self.namespace)

        self._controllers: dict[str, Controller] = {
            mg_id: JointPosController(self._robot_view.get_move_group(mg_id))
            for mg_id in [
                "base",
                "lift",
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
            ]
        }
        assert set(self._controllers.keys()).issubset(set(self._robot_view.move_group_ids()))

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def robot_view(self) -> LinearBotRobotView:
        return self._robot_view

    @property
    def kinematics(self):
        raise NotImplementedError("LinearBot kinematics not yet implemented")

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("LinearBot parallel kinematics not implemented")

    # Gripper finger travel from URDF joint limits.
    # URDF joint directions (open → closed):
    #   joint7_R: 0.0 → -0.0475   joint8_R: 0.0 → -0.0475
    #   joint7_L: 0.0 → +0.0475   joint8_L: 0.0 → -0.0475  (j7_L is inverted)
    GRIPPER_TRAVEL: float = 0.0475

    @staticmethod
    def gripper_targets(g: float) -> dict[str, np.ndarray]:
        """Joint targets for both grippers. g=0.0 → fully open, g=1.0 → fully closed."""
        g = float(np.clip(g, 0.0, 1.0))
        t = LinearBot.GRIPPER_TRAVEL
        return {
            "right_gripper": np.array([-t * (1 - g), -t * (1 - g)]),
            "left_gripper":  np.array([ t * g,       -t * (1 - g)]),
        }

    def get_arm_move_group_ids(self) -> list[str]:
        return ["left_arm", "right_arm"]

    def get_world_pose_tf_mat(self) -> np.ndarray:
        return self.robot_view.get_move_group("base").pose

    def reset(self) -> None:
        self.set_joint_pos(self.exp_config.robot_config.init_qpos)
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        # Body names in the MJCF already carry the robot_0/ namespace prefix.
        return "robot_0/base"

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "BaseRobotConfig"):
        # robot_model_root_name already includes the hardcoded namespace, so strip
        # it from the config before delegating (same pattern as RBY1).
        tmp = robot_config.model_copy(deep=True)
        tmp.robot_namespace = ""
        super().apply_control_overrides(spec, tmp)

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        assert prefix == "robot_0/", f"LinearBot namespace must be 'robot_0/', got {prefix!r}"
        # Body names in the MJCF already carry the robot_0/ prefix, so attach with prefix="".
        # attach_body() also transfers all actuators targeting joints/sites inside the body
        # (lift + arm + gripper actuators), so no need to add them manually.
        super().add_robot_to_scene(
            robot_config=robot_config,
            spec=spec,
            prefix="",
            pos=pos,
            quat=quat,
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )

        # The world-reference site must live in the parent worldbody so it stays at
        # the scene origin regardless of robot placement.
        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])

        # Base site actuators: not transferred by attach_body because their refsite
        # (robot_0/world) is not inside the robot body hierarchy.
        Kp_xy  = 12000
        Kd_xy  = 500
        Kp_yaw = 5000
        Kd_yaw = 300

        def add_slider_act(
            name: str, ctrlrange: float, gainprm: float, biasprm: list[float], gear_idx: int
        ) -> None:
            act = spec.add_actuator()
            act.name = f"{prefix}{name}"
            act.target = f"{prefix}base_site"
            act.refsite = f"{prefix}world"
            act.ctrlrange = np.array([-ctrlrange, ctrlrange])
            act.gainprm[0] = gainprm
            act.biasprm[: len(biasprm)] = biasprm
            act.trntype = mujoco.mjtTrn.mjTRN_SITE
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gear = [0] * 6
            gear[gear_idx] = 1
            act.gear = gear

        add_slider_act("base_x_act",     25,     Kp_xy,  [0, -Kp_xy,  -Kd_xy],  0)
        add_slider_act("base_y_act",     25,     Kp_xy,  [0, -Kp_xy,  -Kd_xy],  1)
        add_slider_act("base_theta_act", 2 * np.pi, Kp_yaw, [0, -Kp_yaw, -Kd_yaw], 5)
