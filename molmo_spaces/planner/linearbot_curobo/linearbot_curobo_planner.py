"""CuRobo v2 motion planner for LinearBot.

Uses MotionPlanner (v2 API) with multi-tool_frame support so both arms are
planned jointly in a single planner instance — no left/right split needed.

Key v2 differences from the old curobo_planner.py (v1):
  - MotionPlanner + MotionPlannerCfg.create()  (was MotionGen + MotionGenConfig)
  - GoalToolPose dict {frame_name: Pose}        (was separate left/right Pose)
  - SceneCfg(cuboid=[...])                      (was WorldConfig(cuboid=[...]))
  - plan_pose(goal_tool_poses, current_state)   (was motion_gen.plan(...))
  - tool_frames property lists all EE frames
"""

import logging
from typing import Optional

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo.types import GoalToolPose, JointState, Pose

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.planner.abstract import Planner
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class LinearBotCuroboPlannerConfig(Config):
    # Path to the LinearBot robot YAML understood by CuRobo v2 RobotCfg.create().
    robot_config_path: str

    # Tool frames to plan for — must match link names declared in the robot YAML.
    # Both arms are handled by one planner; specify which subset to use per call.
    tool_frames: list[str] = ["gripper_R", "gripper_L"]

    # MotionPlannerCfg.create() parameters
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.005       # metres
    orientation_tolerance: float = 0.05     # radians
    collision_cache: dict = {"mesh": 2, "cuboid": 40}
    optimizer_collision_activation_distance: float = 0.01
    max_attempts: int = 5
    use_cuda_graph: bool = True


class LinearBotCuroboPlanner(Planner):
    """Single MotionPlanner instance covering both LinearBot arms."""

    def __init__(self, config: LinearBotCuroboPlannerConfig) -> None:
        self.config = config
        self._build_planner()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_planner(self) -> None:
        cfg = MotionPlannerCfg.create(
            robot=self.config.robot_config_path,
            num_ik_seeds=self.config.num_ik_seeds,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            collision_cache=self.config.collision_cache,
            optimizer_collision_activation_distance=(
                self.config.optimizer_collision_activation_distance
            ),
            use_cuda_graph=self.config.use_cuda_graph,
        )
        self.planner = MotionPlanner(cfg)
        # Pre-allocated collision cache slots are uninitialised and can appear
        # as obstacles at the origin. Clear them so the world starts empty.
        self.planner.clear_scene_cache()
        log.info("LinearBotCuroboPlanner ready. tool_frames=%s", self.planner.tool_frames)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        return self.planner.joint_names

    @property
    def tool_frames(self) -> list[str]:
        return self.planner.tool_frames

    def warmup(self) -> None:
        self.planner.warmup()

    def reset(self) -> None:
        self.planner.reset_seed()

    def motion_plan(
        self,
        joint_position: list[float],
        goal_poses: dict[str, list[float]],
        scene_objects: Optional[list[MlSpacesObject]] = None,
    ) -> tuple[list[list[float]], bool]:
        """Plan a trajectory to Cartesian goals for one or both arms.

        Args:
            joint_position: Current joint positions (len == len(joint_names)).
            goal_poses: Dict mapping tool_frame name → 7D pose [x,y,z, qw,qx,qy,qz].
                        Supply only the frames you want to constrain; the rest are free.
            scene_objects: Optional scene obstacles to add to the collision world.

        Returns:
            (trajectory, success) where trajectory is a list of joint-position
            waypoints (each a list of floats), empty on failure.
        """
        if scene_objects is not None:
            self.update_world(scene_objects)

        current_state = self._make_joint_state(joint_position)
        goal_tool_pose = self._make_goal_tool_pose(goal_poses)

        result = self.planner.plan_pose(
            goal_tool_pose,
            current_state,
            max_attempts=self.config.max_attempts,
        )

        if result is None or not result.success.any():
            return [], False

        # interpolated_trajectory includes locked joints (full robot DOF).
        # Extract only the 16 active planned joints so current_qpos stays consistent.
        b, s = result.success.nonzero(as_tuple=True)
        traj_pos = result.interpolated_trajectory.position[b[0], s[0]]  # [H, full_dof]
        all_names = result.interpolated_trajectory.joint_names
        active_idx = [all_names.index(jn) for jn in self.planner.joint_names]
        traj = traj_pos[:, active_idx].cpu().numpy().tolist()
        return traj, True

    def update_world(self, scene_objects: list[MlSpacesObject]) -> None:
        """Replace the collision world with cuboid AABBs of the given scene objects."""
        cuboids = []
        for obj in scene_objects:
            dims = (np.array([0.3, 0.4, 0.02]) * 2).tolist() \
                if obj.name == "scene/table" \
                else (obj.aabb_size * 2).tolist()
            cuboids.append(Cuboid(
                name=obj.name,
                pose=pose_mat_to_7d(obj.pose).tolist(),
                dims=dims,
            ))
        self.planner.update_world(SceneCfg(cuboid=cuboids))

    def update_world_from_obstacles(self, obstacles: list[dict]) -> None:
        """Replace collision world from pre-serialised obstacle dicts (used by server)."""
        cuboids = [
            Cuboid(name=o["name"], pose=o["pose"], dims=o["dims"])
            for o in obstacles
        ]
        self.planner.update_world(SceneCfg(cuboid=cuboids))

    def diagnose_collision(self, joint_position: list[float]) -> str:
        """Return a human-readable collision summary using only the pairs CuRobo checks.

        Reports world-collision distance, self-collision distance, and any
        penetrating sphere pairs from SelfCollisionKinematicsCfg (excluded pairs removed).
        """
        q_2d = torch.tensor([joint_position], dtype=torch.float32, device="cuda")
        lines: list[str] = []

        # --- feasibility check via graph planner (same check used in planning) ---
        gp = self.planner.graph_planner
        if gp is not None:
            feasible = gp.check_samples_feasibility(q_2d)
            lines.append(f"  graph feasibility: {feasible.cpu().numpy()}")

        # --- sphere-pair check against CuRobo's actual collision_pairs ---
        collisions = self._find_collision_pairs(q_2d)
        if collisions:
            lines.append("  penetrating CuRobo-checked sphere pairs (worst first):")
            for link1, link2, pen in collisions[:8]:
                lines.append(f"    {link1} ↔ {link2}  pen={pen*1000:.1f}mm")
        else:
            lines.append("  no penetration in CuRobo-checked self-collision pairs")

        return "\n".join(lines)

    def _find_collision_pairs(
        self, q_2d: "torch.Tensor"
    ) -> list[tuple[str, str, float]]:
        """Sphere-pair penetration check using only the pairs CuRobo's self-collision kernel checks.

        Reads collision_pairs from SelfCollisionKinematicsCfg, which already excludes
        adjacent links and any pairs in self_collision_ignore.
        """
        kin = self.planner.kinematics
        sc_cfg = kin.get_self_collision_config()  # SelfCollisionKinematicsCfg

        # Pairs CuRobo actually checks: [num_checks, 2] of sphere indices
        checked_pairs = sc_cfg.collision_pairs.cpu().numpy()   # int32, [N, 2]
        sphere_padding = sc_cfg.sphere_padding.cpu().numpy()   # [num_spheres]

        # Sphere world positions at this config (filter_valid=False preserves indices)
        sph_batch = kin.get_robot_as_spheres(q_2d, filter_valid=False)
        spheres = sph_batch[0]

        kin_cfg = kin.config.kinematics_config
        idx_to_link = {v: k for k, v in kin_cfg.link_name_to_idx_map.items()}
        lsm = kin_cfg.link_sphere_idx_map.cpu().numpy()

        # Index spheres by their integer id (embedded in name "curobo/robot_sphere_N")
        pos_r: dict[int, tuple[np.ndarray, float, str]] = {}
        for s in spheres:
            si = int(s.name.rsplit("_", 1)[-1])
            link_name = idx_to_link.get(int(lsm[si]), f"unk{si}")
            r = float(s.radius) + float(sphere_padding[si])
            pos_r[si] = (np.array(s.pose[:3]), r, link_name)

        collisions: list[tuple[str, str, float]] = []
        for ia, ib in checked_pairs:
            if ia not in pos_r or ib not in pos_r:
                continue
            pa, ra, la = pos_r[ia]
            pb, rb, lb = pos_r[ib]
            if ra <= 0 or rb <= 0:
                continue
            pen = ra + rb - float(np.linalg.norm(pa - pb))
            if pen > 0:
                collisions.append((la, lb, pen))

        return sorted(collisions, key=lambda x: -x[2])

    def fk(self, joint_position: list[float]) -> dict[str, list[float]]:
        """Forward kinematics. Returns {frame_name: [x,y,z, qw,qx,qy,qz]}."""
        state = self._make_joint_state(joint_position)
        kin_state = self.planner.compute_kinematics(state)
        # tool_poses.get_link_pose returns Pose with .position/.quaternion [B*H, 3/4]
        result = {}
        for name in self.planner.tool_frames:
            link_pose = kin_state.tool_poses.get_link_pose(name)
            pos  = link_pose.position[0].cpu().numpy().tolist()
            quat = link_pose.quaternion[0].cpu().numpy().tolist()
            result[name] = pos + quat
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_joint_state(self, joint_position: list[float]) -> JointState:
        pos = torch.tensor([joint_position], dtype=torch.float32, device="cuda")
        return JointState.from_position(pos, joint_names=self.joint_names)

    def _make_goal_tool_pose(self, goal_poses: dict[str, list[float]]) -> GoalToolPose:
        """Build GoalToolPose from {frame: [x,y,z, qw,qx,qy,qz]} dict."""
        # ordered_tool_frames must match planner.tool_frames order exactly,
        # otherwise CuRobo assigns goal poses to the wrong EE frames.
        frames = [f for f in self.planner.tool_frames if f in goal_poses]
        pose_dict = {}
        for frame in frames:
            pose_7d = goal_poses[frame]
            pos = torch.tensor([pose_7d[:3]], dtype=torch.float32, device="cuda")
            quat = torch.tensor([pose_7d[3:]], dtype=torch.float32, device="cuda")
            pose_dict[frame] = Pose(position=pos, quaternion=quat, name=frame,
                                    normalize_rotation=False)
        return GoalToolPose.from_poses(pose_dict, ordered_tool_frames=frames, num_goalset=1)
