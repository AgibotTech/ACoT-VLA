import asyncio
import http
import logging
import time
import traceback

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


# corobot's image key naming
_COROBOT_IMG_KEYS = ("head", "hand_left", "hand_right")
# how policies (trained against the v3 dataset pipeline) name the same images
_POLICY_IMG_KEY_MAP = {
    "head": "top_head",
    "hand_left": "hand_left",
    "hand_right": "hand_right",
}


def _get(d: dict, key: str):
    """Get a key from a msgpack-decoded dict, tolerating bytes vs str keys."""
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        return d[bkey]
    return None


def _decode_corobot_jpeg(d):
    """Decode a corobot-style JPEG dict {encoding, image_data, height, width} → HWC RGB uint8.

    Pass through if it's already a numpy array (legacy clients).
    """
    if isinstance(d, np.ndarray):
        return d
    if not isinstance(d, dict):
        raise ValueError(f"Unexpected image payload type: {type(d)}")
    encoding = _get(d, "encoding")
    raw = _get(d, "image_data")
    if raw is None:
        # Fallback: pi-style {"__jpeg__": True, "data": ...}
        raw = _get(d, "data")
    if raw is None:
        raise ValueError(f"Image dict missing image_data/data; keys={list(d.keys())}")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding and encoding.upper() not in ("JPEG", "PNG"):
        raise ValueError(f"Unsupported image encoding: {encoding}")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC BGR
    if img is None:
        raise ValueError("Failed to JPEG-decode image payload")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# corobot robot_type tag -> layout family that assemble/repack branch on.
_ROBOT_TYPE_TO_FAMILY = {
    "arx": "arx",
    "agilex": "aloha",
    "G1_omnipicker": "gseries",
    "G1_120s": "gseries",
    "G2_omnipicker": "gseries",
    "G2_90d": "gseries",
    "G2_crsB_omnipicker": "gseries",
}


def _detect_embodiment(states: dict) -> str:
    """Legacy fallback for clients without robot_type: 12-DOF arm -> arx, else gseries.

    arx and aloha share a layout, so guessing arx for a 12-DOF arm is safe.
    """
    arm = _get(states, "arm_joint_states")
    if arm is not None:
        n = np.asarray(arm).reshape(-1).shape[0]
        if n == 12:
            return "arx"
    return "gseries"


def _resolve_embodiment(params: dict) -> str:
    """Pick the layout family from the client's robot_type tag, else fall back."""
    rt = _get(params, "robot_type")
    if isinstance(rt, bytes):
        rt = rt.decode("utf-8")
    if rt:
        family = _ROBOT_TYPE_TO_FAMILY.get(rt)
        if family is not None:
            return family
        logger.warning("unknown robot_type=%r; falling back to arm-length heuristic", rt)
    return _detect_embodiment(_get(params, "states") or {})


def _assemble_state(states: dict, embodiment: str = "gseries") -> np.ndarray:
    """Reassemble corobot's split state (arm_joint_states + gripper_states) into the
    32-d flat vector the policy expects. Anything not provided is zero-padded.

    arx / aloha (grouped): [0:6] left arm, [6:12] right arm, [12] left grip, [13] right grip
    gseries (G1/G2): [0:14] arms (7+7), [14:16] grippers, [16:21] waist, head after waist
    """
    state = np.zeros(32, dtype=np.float64)

    if embodiment in ("arx", "aloha"):
        arm = _get(states, "arm_joint_states")
        if arm is not None:
            arm = np.asarray(arm, dtype=np.float64).reshape(-1)
            n = min(12, arm.shape[0])
            state[:n] = arm[:n]
        grip = _get(states, "gripper_states")
        if grip is not None:
            grip = np.asarray(grip, dtype=np.float64).reshape(-1)
            n = min(2, grip.shape[0])
            state[12:12 + n] = grip[:n]
        return state

    arm = _get(states, "arm_joint_states")
    if arm is not None:
        arm = np.asarray(arm, dtype=np.float64).reshape(-1)
        n = min(14, arm.shape[0])
        state[:n] = arm[:n]

    grip = _get(states, "gripper_states")
    if grip is not None:
        grip = np.asarray(grip, dtype=np.float64).reshape(-1)
        n = min(2, grip.shape[0])
        state[14:14 + n] = grip[:n]

    waist = _get(states, "waist_joint_states")
    waist_n = 0
    if waist is not None:
        waist = np.asarray(waist, dtype=np.float64).reshape(-1)
        waist_n = min(5, waist.shape[0])  # 2 (G1) or 5 (G2)
        state[16:16 + waist_n] = waist[:waist_n]

    head = _get(states, "head_joint_states")
    if head is not None:
        head = np.asarray(head, dtype=np.float64).reshape(-1)
        # G1 layout puts head at [18:20]; G2 has waist filling 16..20 so park head after.
        head_start = 18 if waist_n <= 2 else (16 + waist_n)
        if head_start + head.shape[0] <= state.shape[0]:
            state[head_start:head_start + head.shape[0]] = head
    return state


def _corobot_params_to_obs(params: dict, embodiment: str = "gseries") -> dict:
    """Translate the corobot JSON-RPC params block into the flat dict the policy expects."""
    images_in = _get(params, "images") or {}
    images_out = {}
    for k in _COROBOT_IMG_KEYS:
        v = _get(images_in, k)
        if v is None:
            continue
        images_out[_POLICY_IMG_KEY_MAP[k]] = _decode_corobot_jpeg(v)

    states_in = _get(params, "states") or {}
    state = _assemble_state(states_in, embodiment)

    prompt = _get(params, "prompt") or ""
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    obs = {
        "state": state,
        "images": images_out,
        "prompt": prompt,
        "task_name": _get(params, "task_name") or "",
        "episode_idx": _get(params, "episode_idx") or 0,
        "episode_done": bool(_get(params, "episode_done") or False),
        "task_progress": _get(params, "task_progress") or [],
    }
    return obs


def _policy_action_to_corobot_result(action_dict: dict, embodiment: str = "gseries") -> dict:
    """Repack the policy's `{"actions": (chunk, dim)}` into the corobot result envelope.

    Grippers go via the effector channel; arm values carry only real joints (no
    padding), since the client concatenates left+right and slices the first arm_dim.

    arx / aloha (grouped): left_arm[:,0:6] right_arm[:,6:12] eff[:,12:13]/[:,13:14]
    gseries: left_arm[:,0:7] right_arm[:,7:14] eff[:,14:15]/[:,15:16], waist[:,20:21] if dim>=21
    """
    actions = action_dict.get("actions")
    if actions is None:
        bkey = b"actions"
        if bkey in action_dict:
            actions = action_dict[bkey]
    if actions is None:
        raise ValueError(f"Policy output missing 'actions' key; got: {list(action_dict.keys())}")

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D action chunk, got shape {actions.shape}")

    _, dim = actions.shape
    if dim < 14:
        raise ValueError(f"Action dim {dim} < 14, cannot split into left/right arms")

    waist_vals = None
    if embodiment in ("arx", "aloha"):
        left_arm_vals  = actions[:, 0:6].tolist()
        right_arm_vals = actions[:, 6:12].tolist()
        left_eff_vals  = actions[:, 12:13].tolist()
        right_eff_vals = actions[:, 13:14].tolist()
    else:
        if dim < 16:
            raise ValueError(f"G-series action dim {dim} < 16, expected 7+7 arms + grippers")
        left_arm_vals  = actions[:, 0:7].tolist()
        right_arm_vals = actions[:, 7:14].tolist()
        left_eff_vals  = actions[:, 14:15].tolist()
        right_eff_vals = actions[:, 15:16].tolist()
        if dim >= 21:
            # Full waist 16..20 in joint order: dims 16..19 are frozen to the current
            # pose server-side, dim 20 is the learned joint. Client maps positionally.
            waist_vals = actions[:, 16:21].tolist()

    result = {
        "left_arm": {"kind": "JOINT_ABS", "values": left_arm_vals},
        "right_arm": {"kind": "JOINT_ABS", "values": right_arm_vals},
        "left_effector": left_eff_vals,
        "right_effector": right_eff_vals,
    }
    if waist_vals is not None:
        result["waist"] = {"kind": "JOINT_ABS", "values": waist_vals}
    return result


class WebsocketPolicyServer:
    """Serves a policy over WebSocket using the **corobot** JSON-RPC protocol.

    Wire format
    -----------
    Request (client → server):
        {"method": "infer", "params": {images, states, prompt, task_name, ...}}
    Success response (server → client):
        {"result": {left_arm, right_arm, left_effector, right_effector, [waist]}}
    Failure response:
        {"error": "<traceback string>"}

    There is **no metadata first packet** (corobot clients don't expect one).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,  # accepted for backward-compat; unused by corobot
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        if metadata:
            logger.info("Server metadata=%s — ignored by corobot protocol", metadata)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened (corobot)")
        packer = msgpack_numpy.Packer()

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                req = msgpack_numpy.unpackb(await websocket.recv())

                method = _get(req, "method")
                if isinstance(method, bytes):
                    method = method.decode("utf-8")
                if method != "infer":
                    await websocket.send(
                        packer.pack({"error": f"unknown method: {method!r}"})
                    )
                    continue

                params = _get(req, "params") or {}
                embodiment = _resolve_embodiment(params)

                try:
                    obs = _corobot_params_to_obs(params, embodiment)
                except Exception:
                    await websocket.send(
                        packer.pack({"error": f"failed to parse params:\n{traceback.format_exc()}"})
                    )
                    continue

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                try:
                    result = _policy_action_to_corobot_result(action, embodiment)
                except Exception:
                    await websocket.send(
                        packer.pack({"error": f"failed to repack policy output:\n{traceback.format_exc()}"})
                    )
                    continue

                resp = {"result": result}
                # Server timing piggy-backed alongside result for compat with PiPolicy clients
                # that ignored these fields; corobot clients should also ignore unknown keys.
                resp["server_timing"] = {"infer_ms": infer_time * 1000.0}
                model_ms = (_get(action, "policy_timing") or {}).get("infer_ms")
                if model_ms is not None:
                    resp["server_timing"]["model_ms"] = float(model_ms)
                if prev_total_time is not None:
                    resp["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(resp))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(packer.pack({"error": traceback.format_exc()}))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
