"""gRPC client for the LinearBot CuRobo v2 planner server.

Usage:
    client = LinearBotCuroboClient("localhost:10001")
    traj, ok = client.motion_plan(
        joint_position=[...],
        goal_poses={
            "gripper_R": [x, y, z, qw, qx, qy, qz],
            "gripper_L": [x, y, z, qw, qx, qy, qz],
        },
    )

host:port format only — no http:// prefix (stripped automatically for compat).
"""

import json

import grpc

_SERVICE = "linearbot_curobo_planner.LinearBotCuroboPlanner"
_MAX_MSG = 512 * 1024 * 1024

_RPC_NAMES = [
    "Health",
    "JointNames",
    "ToolFrames",
    "Fk",
    "MotionPlan",
    "UpdateWorld",
    "Warmup",
    "Reset",
]


def _encode(obj: dict) -> bytes:
    return json.dumps(obj).encode()


def _decode(data: bytes) -> dict:
    return json.loads(data)


class LinearBotCuroboClient:
    def __init__(self, address: str = "localhost:10001") -> None:
        # Strip http:// or https:// prefix if present (backward compat)
        for scheme in ("https://", "http://"):
            if address.startswith(scheme):
                address = address[len(scheme):]
        self._address = address
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_MSG),
                ("grpc.max_receive_message_length", _MAX_MSG),
            ],
        )
        self._stubs: dict[str, grpc.UnaryUnaryMultiCallable] = {
            name: self._channel.unary_unary(
                f"/{_SERVICE}/{name}",
                request_serializer=_encode,
                response_deserializer=_decode,
            )
            for name in _RPC_NAMES
        }

    def _call(self, method: str, payload: dict) -> dict:
        return self._stubs[method](payload)

    # ------------------------------------------------------------------
    # RPC wrappers
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check server liveness. Returns {"status": "ok", "tool_frames": [...]}."""
        return self._call("Health", {})

    def joint_names(self) -> list[str]:
        return self._call("JointNames", {})["joint_names"]

    def tool_frames(self) -> list[str]:
        return self._call("ToolFrames", {})["tool_frames"]

    def warmup(self) -> None:
        """Trigger CUDA graph capture on the server (blocking)."""
        self._call("Warmup", {})

    def reset(self) -> None:
        """Reset planner random seed."""
        self._call("Reset", {})

    def fk(self, joint_position: list[float]) -> dict[str, list[float]]:
        """Forward kinematics.

        Returns:
            Dict mapping tool_frame name → [x, y, z, qw, qx, qy, qz].
        """
        return self._call("Fk", {"joint_position": joint_position})["poses"]

    def motion_plan(
        self,
        joint_position: list[float],
        goal_poses: dict[str, list[float]],
    ) -> tuple[list[list[float]], bool]:
        """Plan a trajectory to Cartesian goals.

        Args:
            joint_position: Current joint positions.
            goal_poses: Dict mapping tool_frame name → 7D pose [x,y,z, qw,qx,qy,qz].
                        Supply only the frames you want to constrain (right arm only,
                        left arm only, or both simultaneously).

        Returns:
            (trajectory, success). trajectory is a list of joint-position waypoints.
            Empty list on failure.
        """
        resp = self._call("MotionPlan", {
            "joint_position": joint_position,
            "goal_poses": goal_poses,
        })
        return resp["trajectory"], resp["success"]

    def update_world(self, obstacles: list[dict]) -> dict:
        """Replace the collision world on the server.

        Args:
            obstacles: List of dicts with keys:
                "name"  (str)             — unique obstacle identifier
                "pose"  ([x,y,z,qw,qx,qy,qz])  — obstacle pose in world frame
                "dims"  ([dx, dy, dz])    — full extents in metres (not half-extents)

        Returns:
            {"status": "ok", "num_obstacles": N}
        """
        return self._call("UpdateWorld", {"obstacles": obstacles})

    def close(self) -> None:
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
