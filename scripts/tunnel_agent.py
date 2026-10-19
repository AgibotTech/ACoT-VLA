"""tunnel_agent —— ACoT-VLA 在 Simubotix 挑战赛隧道协议下的推理入口。

替代旧的 ``serve_policy.py``(host/port 监听式 WebSocket 服务器):本脚本
反向拨号到平台 Gateway,按 tunnel-protocol.zh-CN.md 描述的 5 类控制帧
推进 QUEUED → WARMUP → RUNNING → DRAINING 状态机,并把每个二进制
数据帧 (msgpack 编码的 obs) 交给 openpi 推理后,再用 msgpack 回写。

启动示例:
    uv run scripts/tunnel_agent.py \\
        --access-token   $CHALLENGE_TOKEN \\
        --job-uuid       <POST /api/challenge/job 返回的 uuid> \\
        --gateway-url    wss://<host>/api/challenge/tunnel

默认配置:
    policy.config = pi05_genie_sim_10_mini_task_20260312
    policy.dir    = checkpoints/30000

可选 ``--config`` / ``--ckpt-dir`` 覆盖默认值。

帧载荷与 ``openpi.serving.websocket_policy_server`` 保持兼容:仍是
``msgpack_numpy`` 编码的 obs/action 字典,模拟器侧 ``openpi_client`` 不需
要改。
"""
from __future__ import annotations

import argparse
import asyncio
import enum
import json
import logging
import os
import struct
import sys
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Set, Union
from urllib.parse import urlencode, urlparse, urlunparse

# 与 serve_policy.py 一致:在导入 jax/openpi 之前修正 XLA flags。
os.environ.setdefault("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm=false" not in os.environ["XLA_FLAGS"]:
    os.environ["XLA_FLAGS"] += " --xla_gpu_enable_triton_gemm=false"

import cv2
import numpy as np
import websockets
from websockets.asyncio.client import connect as ws_connect

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import msgpack_numpy


logger = logging.getLogger("tunnel_agent")

# one-shot guards for debug logging (obs in / action out)
_OBS_DUMPED = False
_RESP_DUMPED_HOLDER = [False]


# ---------------------------------------------------------------------------
# 数据帧编解码 —— 与 Go pkg/tunnel 逐字节兼容
# 布局: [uint32 BE sessionID 长度][sessionID 字节][payload 字节]
# ---------------------------------------------------------------------------

MAX_SESSION_ID_LEN = 256


def encode_data_frame(session_id: str, payload: bytes) -> bytes:
    sid = session_id.encode("utf-8")
    if len(sid) > MAX_SESSION_ID_LEN:
        raise ValueError(
            f"session_id exceeds {MAX_SESSION_ID_LEN} bytes: got {len(sid)}"
        )
    return struct.pack(">I", len(sid)) + sid + payload


def decode_data_frame(frame: bytes) -> tuple[str, bytes]:
    if len(frame) < 4:
        raise ValueError("tunnel: data frame malformed (too short)")
    (sid_len,) = struct.unpack(">I", frame[:4])
    if sid_len > len(frame) - 4:
        raise ValueError("tunnel: data frame malformed (sid_len out of bounds)")
    return frame[4 : 4 + sid_len].decode("utf-8"), bytes(frame[4 + sid_len :])


# ---------------------------------------------------------------------------
# 新网关 obs 适配 —— 网关下发 JSON-RPC 信封:
#   {"method":"infer","params":{
#       "images": {"head"|"hand_left"|"hand_right": {"encoding":"JPEG",
#                  "image_data":<bytes>,"height":H,"width":W}},
#       "states": {"head_joint_states":[], "arm_joint_states":[14],
#                  "waist_joint_states":[5], "gripper_states":[2]},
#       "prompt": str, "task_name": str, "robot_type": str, ...}}
# 而 openpi 策略 (Go1Inputs) 期望扁平 obs:
#   {"state": ndarray[16], "images": {"top_head"|"hand_left"|"hand_right": HWC RGB uint8},
#    "prompt": str, "task_name": str}
# 这里做协议适配,模型代码保持不动。
# ---------------------------------------------------------------------------

# 网关相机名 -> 模型相机名
_CAMERA_RENAME = {"head": "top_head", "hand_left": "hand_left", "hand_right": "hand_right"}


def _decode_image_field(cam: dict) -> np.ndarray:
    """{"encoding":"JPEG","image_data":bytes,...} -> HWC RGB uint8。"""
    arr = np.frombuffer(cam["image_data"], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC BGR
    if img is None:
        raise ValueError("image decode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC RGB uint8


def _adapt_obs(obs: dict) -> dict:
    """把网关 JSON-RPC obs 信封 {"method":"infer","params":{...}} 转成策略期望的扁平 obs。"""
    params = obs["params"]

    # --- 图像: 逐相机解码并改名 ---
    raw_images = params.get("images", {}) or {}
    images: dict[str, np.ndarray] = {}
    for src, dst in _CAMERA_RENAME.items():
        cam = raw_images.get(src)
        if cam is not None:
            images[dst] = _decode_image_field(cam)

    # --- 状态: 按训练 state_keys 顺序拼 (joint, left_eff, right_eff, waist) ---
    # state_indices=None(info.json 缺失)=> 模型期望已选好的向量,布局与训练一致:
    #   arm_joint_states(14) + gripper_states(2) + waist_joint_states(5) = 21 (G2)。
    # 一套拼法通吃三个榜单: include_waist=False 的 config(instruction/spatial)其
    # state_mask 会把 16+ 维(即 waist)清零,等价于不带 waist;include_waist=True 的
    # config(manip)正需要 waist 落在 dims16-20。
    states = params.get("states", {}) or {}
    arm = np.asarray(states.get("arm_joint_states", []), dtype=np.float32).reshape(-1)
    grip = np.asarray(states.get("gripper_states", []), dtype=np.float32).reshape(-1)
    waist = np.asarray(states.get("waist_joint_states", []), dtype=np.float32).reshape(-1)
    state = np.concatenate([arm, grip, waist])

    adapted: dict[str, Any] = {"images": images, "state": state}
    if params.get("prompt") is not None:
        adapted["prompt"] = params["prompt"]
    if params.get("task_name") is not None:
        adapted["task_name"] = params["task_name"]
    return adapted


def _build_action_response(action: dict) -> dict:
    """把策略输出转成 genie-sim 期望的响应信封。

    genie-sim corobotpolicy._parse_result / infer 期望:
      {"result": {
          "left_arm":  {"kind":"JOINT_ABS","values": [[7]*H]},
          "right_arm": {"kind":"JOINT_ABS","values": [[7]*H]},
          "left_effector":  [[1]*H],
          "right_effector": [[1]*H],
          "waist": {"kind":"JOINT_ABS","values": [[4]*H]}   # 仅 waist 任务
      }}
    动作布局: [0:7]=左臂 [7:14]=右臂 [14]=左夹 [15]=右夹 [16:]=腰部(仅 manip 含)。
    instruction/spatial 输出 16 维;manip 的 waist 任务输出 21 维(含 waist 5)。

    注意: genie-sim 用纯 msgpack(非 msgpack_numpy)解包,因此这里所有数组必须
    转成原生 Python list,否则对端拿到的是 ext 编码的乱码。
    """
    acts = np.asarray(action.get("actions"), dtype=np.float32)
    if acts.ndim == 1:
        acts = acts[None, :]

    result: dict[str, Any] = {
        "left_arm":  {"kind": "JOINT_ABS", "values": acts[:, 0:7].tolist()},
        "right_arm": {"kind": "JOINT_ABS", "values": acts[:, 7:14].tolist()},
        "left_effector":  acts[:, 14:15].tolist(),
        "right_effector": acts[:, 15:16].tolist(),
    }
    # manip 的 waist 任务(如 sorting_packages)输出含腰部维度;其余任务被裁到 16 维。
    if acts.shape[1] > 16:
        result["waist"] = {"kind": "JOINT_ABS", "values": acts[:, 16:].tolist()}

    return {"result": result}


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class State(str, enum.Enum):
    QUEUED = "QUEUED"
    WARMUP = "WARMUP"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"


class TunnelExhausted(RuntimeError):
    """run() 在达到 max_retries 后仍无法建立健康连接时抛出。"""


FrameHandler = Callable[[str, bytes], Union[bytes, Awaitable[bytes]]]


# ---------------------------------------------------------------------------
# 策略 + 推理 handler
# ---------------------------------------------------------------------------


class PolicyHandler:
    """惰性加载 openpi 策略,并把 msgpack 帧转成 obs / action。

    warmup 阶段服务端会调用一次 ``handle("", b"")``;此时 frame 为空,我们
    只把权重加载完成,不真正跑一遍 forward(避免无 obs 时构造假数据)。
    实际推理在 RUNNING 阶段、收到带真实 obs 的数据帧时进行。
    """

    def __init__(self, config_name: str, ckpt_dir: str, default_prompt: str | None = None) -> None:
        self._config_name = config_name
        self._ckpt_dir = ckpt_dir
        self._default_prompt = default_prompt
        self._policy = None
        self._packer = msgpack_numpy.Packer()

    def load(self) -> None:
        if self._policy is not None:
            return
        logger.info(
            "loading policy: config=%s dir=%s", self._config_name, self._ckpt_dir
        )
        t0 = time.monotonic()
        self._policy = _policy_config.create_trained_policy(
            _config.get_config(self._config_name),
            self._ckpt_dir,
            default_prompt=self._default_prompt,
        )
        logger.info("policy loaded in %.2fs", time.monotonic() - t0)

    def __call__(self, session_id: str, frame: bytes) -> bytes:
        # warmup 探针:服务端用 session_id="" 且空 payload 触发一次预热调用。
        if not frame:
            self.load()
            return b""

        if self._policy is None:
            # 兜底:若 warmup 控制帧丢失,首帧也要保证模型已加载。
            self.load()

        try:
            obs = msgpack_numpy.unpackb(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("session=%s msgpack decode failed: %s", session_id, exc)
            raise

        obs = _adapt_obs(obs)

        # one-shot log of the adapted obs shape, to confirm the interface mapping.
        global _OBS_DUMPED
        if not _OBS_DUMPED:
            _OBS_DUMPED = True
            try:
                img_shapes = {k: getattr(v, "shape", None) for k, v in obs.get("images", {}).items()}
                logger.info(
                    "adapted obs: state=%s images=%s prompt=%r task=%r",
                    getattr(obs.get("state"), "shape", None), img_shapes,
                    obs.get("prompt"), obs.get("task_name"),
                )
            except Exception as _e:  # noqa: BLE001
                logger.error("adapted obs log failed: %s", _e)

        infer_t0 = time.monotonic()
        action = self._policy.infer(obs)
        infer_ms = (time.monotonic() - infer_t0) * 1000

        # 新网关/genie-sim 期望 {"result": {left_arm,right_arm,left_effector,...}} 信封,
        # 用纯 msgpack 解包 —— 全部转 list,见 _build_action_response。
        response = _build_action_response(action)

        if not _RESP_DUMPED_HOLDER[0]:
            _RESP_DUMPED_HOLDER[0] = True
            r = response["result"]
            logger.info(
                "action response: left_arm=%dx%d right_arm=%dx%d eff=(%d,%d) infer_ms=%.1f",
                len(r["left_arm"]["values"]), len(r["left_arm"]["values"][0]) if r["left_arm"]["values"] else 0,
                len(r["right_arm"]["values"]), len(r["right_arm"]["values"][0]) if r["right_arm"]["values"] else 0,
                len(r["left_effector"]), len(r["right_effector"]), infer_ms,
            )

        return self._packer.pack(response)


# ---------------------------------------------------------------------------
# TunnelClient —— 与 simubotix_agent.TunnelClient 等价的实现
# ---------------------------------------------------------------------------


class TunnelClient:
    def __init__(
        self,
        url: str,
        access_token: str,
        job_uuid: str,
        on_frame: FrameHandler,
        agent_id: str | None = None,
        on_session_close: Callable[[str], None] | None = None,
    ) -> None:
        self._url = url
        self._access_token = access_token
        self._job_uuid = job_uuid
        self.agent_id = agent_id or str(uuid.uuid4())
        self._on_frame = on_frame
        self._on_session_close = on_session_close
        self.state: State = State.QUEUED
        self.state_history: list[State] = [State.QUEUED]
        self.active_sessions: Set[str] = set()
        self._drained = False
        self._inflight: Set[asyncio.Task] = set()

    def _build_ws_url(self) -> str:
        parts = urlparse(self._url)
        new_query = urlencode({"job": self._job_uuid, "agent": self.agent_id})
        if parts.query:
            new_query = parts.query + "&" + new_query
        return urlunparse(parts._replace(query=new_query))

    def _set_state(self, new_state: State) -> None:
        self.state = new_state
        self.state_history.append(new_state)
        logger.info("state -> %s", new_state.value)

    async def _invoke_handler(self, session_id: str, payload: bytes) -> bytes:
        # openpi 推理是同步、CPU/GPU 阻塞型,丢到工作线程跑,避免阻塞 recv 循环。
        return await asyncio.to_thread(self._on_frame, session_id, payload)

    async def run(
        self,
        max_retries: int = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 30.0,
    ) -> None:
        attempt = 0
        while True:
            self._drained = False
            try:
                await self.run_once()
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                last_error: Exception = exc
                logger.warning("tunnel error: %s", exc)
            else:
                if self._drained:
                    return
                last_error = RuntimeError("WS closed without drain")
            if attempt >= max_retries:
                raise TunnelExhausted(
                    f"giving up after {attempt + 1} attempts: {last_error}"
                ) from last_error
            backoff = min(max_backoff, initial_backoff * (2 ** attempt))
            logger.info("reconnect in %.1fs (attempt %d)", backoff, attempt + 1)
            await asyncio.sleep(backoff)
            attempt += 1

    async def run_once(self) -> None:
        url = self._build_ws_url()
        logger.info("dialing gateway: %s", url)
        async with ws_connect(
            url,
            additional_headers={"Authorization": f"Bearer {self._access_token}"},
            ping_interval=20,
            ping_timeout=10,
            max_size=None,
        ) as ws:
            try:
                await self._recv_loop(ws)
            except websockets.exceptions.ConnectionClosedOK:
                return

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, str):
                await self._handle_control(ws, msg)
                if self.state == State.DRAINING:
                    await ws.close()
                    return
            elif isinstance(msg, (bytes, bytearray)):
                self._spawn(self._handle_data(ws, bytes(msg)))
                logger.info("data frame received")

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return task

    async def _handle_control(self, ws, msg: str) -> None:
        try:
            frame = json.loads(msg)
        except json.JSONDecodeError:
            return
        ctrl_type = frame.get("type")
        session_id = frame.get("session_id", "")

        if ctrl_type == "warmup":
            self._set_state(State.WARMUP)
            try:
                await self._invoke_handler("", b"")
            except Exception as exc:  # noqa: BLE001
                logger.info("warmup failed: %s: %s", type(exc).__name__, exc)
                traceback.print_exc(file=sys.stderr)
                return
            await ws.send(json.dumps({"type": "ready"}))
            self._set_state(State.RUNNING)

        elif ctrl_type == "session_open":
            if session_id:
                self.active_sessions.add(session_id)
                logger.info("session open: %s", session_id)

        elif ctrl_type == "session_close":
            if session_id:
                self.active_sessions.discard(session_id)
                if self._on_session_close is not None:
                    try:
                        self._on_session_close(session_id)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc(file=sys.stderr)
                logger.info("session close: %s", session_id)

        elif ctrl_type == "drain":
            self.active_sessions.clear()
            self._drained = True
            self._set_state(State.DRAINING)

    async def _handle_data(self, ws, frame: bytes) -> None:
        try:
            session_id, payload = decode_data_frame(frame)
        except ValueError as exc:
            logger.warning("malformed data frame dropped: %s", exc)
            return
        try:
            response = await self._invoke_handler(session_id, payload)
        except Exception:  # noqa: BLE001
            logger.error(
                "handler error on session=%s\n%s", session_id, traceback.format_exc()
            )
            return
        try:
            await ws.send(encode_data_frame(session_id, response))
        except websockets.exceptions.ConnectionClosed:
            return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_CONFIG = "pi05_genie_sim_10_mini_task_20260312"
DEFAULT_CKPT_DIR = "checkpoints/30000"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tunnel_agent",
        description="ACoT-VLA 隧道协议推理 Agent",
    )
    parser.add_argument(
        "--access-token",
        default=os.environ.get("CHALLENGE_TOKEN", ""),
        help="登录凭据,缺省读取 CHALLENGE_TOKEN 环境变量",
    )
    parser.add_argument(
        "--job-uuid",
        required=True,
        help="POST /api/challenge/job 响应中的 uuid 字段",
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="agent 进程标识,缺省时本进程随机生成 UUIDv4",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("SIMUBOTIX_GATEWAY_URL", ""),
        help="Gateway WS 地址,缺省读取 SIMUBOTIX_GATEWAY_URL 环境变量",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"openpi training config 名(默认: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIR,
        help=f"模型权重路径(默认: {DEFAULT_CKPT_DIR})",
    )
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="obs 中无 prompt 字段时使用的兜底 prompt",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="重连尝试上限",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    ns = parse_args(argv)
    if not ns.gateway_url:
        logger.info("需要 --gateway-url 或 SIMUBOTIX_GATEWAY_URL 环境变量")
        return 1
    if not ns.access_token:
        logger.info("需要 --access-token 或 CHALLENGE_TOKEN 环境变量")
        return 1

    handler = PolicyHandler(
        config_name=ns.config,
        ckpt_dir=ns.ckpt_dir,
        default_prompt=ns.default_prompt,
    )

    logger.info(
        "tunnel_agent: gateway=%s job=%s config=%s ckpt=%s",
        ns.gateway_url,
        ns.job_uuid,
        ns.config,
        ns.ckpt_dir,
    )

    client = TunnelClient(
        url=ns.gateway_url,
        access_token=ns.access_token,
        job_uuid=ns.job_uuid,
        agent_id=ns.agent_id,
        on_frame=handler,
    )
    try:
        asyncio.run(client.run(max_retries=ns.max_retries))
    except KeyboardInterrupt:
        return 130
    except TunnelExhausted as exc:
        logger.info("隧道重连耗尽: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
