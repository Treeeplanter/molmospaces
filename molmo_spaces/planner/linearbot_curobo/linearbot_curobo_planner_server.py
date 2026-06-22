"""gRPC server wrapping LinearBotCuroboPlanner (CuRobo v2).

Launch:
    python -m molmo_spaces.planner.linearbot_curobo_planner_server \\
        --robot-config /path/to/linearbot_curobo.yml \\
        --port 10001

One server per node. Workers on the same machine connect via localhost:<port>.
All GPU operations are serialised through a single dedicated GPU thread so
GPU memory stays flat regardless of client count.

Wire format: JSON over gRPC (no .proto compilation required).
"""

import argparse
import concurrent.futures as cf
import json
import logging
import queue
import threading
from concurrent import futures

import grpc

from molmo_spaces.planner.linearbot_curobo.linearbot_curobo_planner import (
    LinearBotCuroboPlanner,
    LinearBotCuroboPlannerConfig,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SERVICE = "linearbot_curobo_planner.LinearBotCuroboPlanner"
_MAX_MSG = 512 * 1024 * 1024

# ---------------------------------------------------------------------------
# Single GPU worker thread
# ---------------------------------------------------------------------------

_gpu_queue: queue.Queue = queue.Queue()


def _gpu_worker() -> None:
    while True:
        fn, fut = _gpu_queue.get()
        try:
            fut.set_result(fn())
        except Exception as e:
            fut.set_exception(e)


def _run_on_gpu(fn):
    fut = cf.Future()
    _gpu_queue.put((fn, fut))
    return fut.result()


# ---------------------------------------------------------------------------
# Wire encoding
# ---------------------------------------------------------------------------


def _encode(obj: dict) -> bytes:
    return json.dumps(obj).encode()


def _decode(data: bytes) -> dict:
    return json.loads(data)


# ---------------------------------------------------------------------------
# Global planner instance (initialised in main)
# ---------------------------------------------------------------------------

_planner: LinearBotCuroboPlanner | None = None


# ---------------------------------------------------------------------------
# gRPC handler
# ---------------------------------------------------------------------------


class _Handler(grpc.GenericRpcHandler):
    def service_name(self) -> str:
        return _SERVICE

    def _dispatch(self, method: str, req: dict, context) -> dict:
        handlers = {
            "Health":               self.Health,
            "JointNames":           self.JointNames,
            "ToolFrames":           self.ToolFrames,
            "Fk":                   self.Fk,
            "MotionPlan":           self.MotionPlan,
            "UpdateWorld":          self.UpdateWorld,
            "Warmup":               self.Warmup,
            "Reset":                self.Reset,
        }
        handler = handlers.get(method)
        if handler is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, f"Unknown method: {method}")
        return handler(req, context)

    def service(self, handler_call_details):
        method = handler_call_details.method.split("/")[-1]
        return grpc.unary_unary_rpc_method_handler(
            lambda req_bytes, ctx: _encode(
                self._dispatch(method, _decode(req_bytes), ctx)
            ),
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        )

    # --- RPC methods ---

    def Health(self, req: dict, context) -> dict:
        return {"status": "ok", "tool_frames": _planner.tool_frames}

    def JointNames(self, req: dict, context) -> dict:
        return {"joint_names": _planner.joint_names}

    def ToolFrames(self, req: dict, context) -> dict:
        return {"tool_frames": _planner.tool_frames}

    def Warmup(self, req: dict, context) -> dict:
        def _run():
            _planner.warmup()
            return {"status": "ok"}
        return _run_on_gpu(_run)

    def Reset(self, req: dict, context) -> dict:
        def _run():
            _planner.reset()
            return {"status": "ok"}
        return _run_on_gpu(_run)

    def Fk(self, req: dict, context) -> dict:
        if "joint_position" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'joint_position'")

        def _run():
            poses = _planner.fk(req["joint_position"])
            return {"poses": poses}

        return _run_on_gpu(_run)

    def MotionPlan(self, req: dict, context) -> dict:
        for key in ("joint_position", "goal_poses"):
            if key not in req:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing '{key}'")

        def _run():
            traj, success = _planner.motion_plan(
                joint_position=req["joint_position"],
                goal_poses=req["goal_poses"],       # {frame: [x,y,z,qw,qx,qy,qz]}
                scene_objects=None,                 # obstacles set via UpdateWorld
            )
            return {"trajectory": traj, "success": success}

        return _run_on_gpu(_run)

    def UpdateWorld(self, req: dict, context) -> dict:
        if "obstacles" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'obstacles'")
        for obj in req["obstacles"]:
            if not all(k in obj for k in ("name", "pose", "dims")):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"obstacle missing required fields (name, pose, dims): {obj}",
                )

        def _run():
            _planner.update_world_from_obstacles(req["obstacles"])
            return {"status": "ok", "num_obstacles": len(req["obstacles"])}

        return _run_on_gpu(_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global _planner

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", required=True,
                        help="Path to LinearBot CuRobo v2 robot YAML")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--workers", type=int, default=4,
                        help="gRPC thread-pool workers (GPU ops still serialised)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup after init (captures CUDA graphs)")
    args = parser.parse_args()

    log.info("Building LinearBotCuroboPlanner from %s", args.robot_config)
    cfg = LinearBotCuroboPlannerConfig(robot_config_path=args.robot_config)

    def _init():
        global _planner
        _planner = LinearBotCuroboPlanner(cfg)
        if args.warmup:
            log.info("Warming up CUDA graphs...")
            _planner.warmup()
            log.info("Warmup complete.")

    threading.Thread(target=_gpu_worker, daemon=True).start()
    _run_on_gpu(_init)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    server.add_generic_rpc_handlers([_Handler()])
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info("LinearBot CuRobo v2 server listening on port %d", args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
