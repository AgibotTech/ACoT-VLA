"""Serve a pi0/pi0.5 checkpoint with server-owned (Anvil-style) Real-Time Chunking.

The server owns the action queue and the execution clock and streams one absolute
command per tick; see src/openpi/serving/rtc_stream_server.py for the protocol.
The request/response corobot endpoint (scripts/serve_policy.py, method infer)
is unchanged.

    uv run scripts/serve_rtc_stream.py \
        --config pi05_g2_ipw_vision_action \
        --dir checkpoints/pi05_g2_ipw_vision_action/InfPoseWaistVisionGcmd/56000 \
        --port 8001 --rtc.clock wall
"""

import dataclasses
import logging

import tyro

# Unlike serve_policy.py, this does NOT set --xla_gpu_enable_triton_gemm=false. On the
# Workstation RTX PRO 6000 (Blackwell) that flag makes one pi0.5 forward 376 ms instead
# of 54 ms, while changing actions by at most 0.0036 rad (arms, bf16 noise). RTC needs
# the short latency. Export XLA_FLAGS yourself to get the old behaviour back.
from openpi.models import pi0
from openpi.policies import policy_config as _policy_config
from openpi.serving import rtc_stream_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Command-line arguments for serve_rtc_stream.py (parsed by tyro; nested ``--rtc.*``)."""

    # Training config name (e.g. "pi05_g2_ipw_vision_action").
    config: str
    # Checkpoint step directory (contains params/ and assets/).
    dir: str
    host: str = "0.0.0.0"
    port: int = 8001
    default_prompt: str | None = None
    # New diffusion noise every inference (LeRobot/Anvil). --no-fresh-noise reuses one fixed
    # key, like serve_policy.py's infer.
    fresh_noise: bool = True
    rtc: rtc_stream_server.RtcStreamConfig = dataclasses.field(default_factory=rtc_stream_server.RtcStreamConfig)


def main(args: Args) -> None:
    """Load the checkpoint, resolve the waist mode and serve the RTC stream (blocking).

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If the config is not Pi0/pi0.5 or ``--rtc.waist`` is not a valid mode.
    """
    train_config = _config.get_config(args.config)
    if not isinstance(train_config.model, pi0.Pi0Config):
        raise SystemExit(
            f"{args.config} uses {type(train_config.model).__name__}; server-side RTC supports Pi0/pi0.5 only "
            "(ACoT's coarse head cannot take an RTC prefix)."
        )

    unmasked = bool(getattr(train_config.data, "unmask_waist", False))
    if args.rtc.waist == "auto":
        waist = "model" if unmasked else "hold"
    elif args.rtc.waist in ("model", "hold", "off"):
        waist = args.rtc.waist
    else:
        raise SystemExit(f"--rtc.waist must be auto, model, hold or off, got {args.rtc.waist!r}")
    if waist == "model" and not unmasked:
        logging.warning("waist=model on %s, whose waist dims 16:19 were masked in training", args.config)
    logging.info(
        "waist: %s",
        {
            "model": "dims 16:21 from the model",
            "hold": "dims 16:20 held at the measured state, dim 20 from the model",
            "off": "dims 16:21 held at the measured state",
        }[waist],
    )

    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt, fresh_noise=args.fresh_noise
    )
    logging.info("diffusion noise: %s", "fresh every call" if args.fresh_noise else "fixed key")
    engine = rtc_stream_server.RtcEngine(policy, args.rtc, waist=waist)
    rtc_stream_server.RtcStreamServer(engine, args.rtc, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
