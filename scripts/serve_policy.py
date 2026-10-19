import dataclasses
import enum
import logging
import os
import socket

import tyro

# GEMMs on cuBLAS instead of XLA's Triton kernels. On the Workstation RTX PRO 6000 this makes
# one pi0.5 forward ~376 ms instead of ~54 ms; OPENPI_KEEP_TRITON_GEMM=1 keeps Triton.
os.environ.setdefault("XLA_FLAGS", "")
if (
    os.environ.get("OPENPI_KEEP_TRITON_GEMM") != "1"
    and "--xla_gpu_enable_triton_gemm=false" not in os.environ["XLA_FLAGS"]
):
    os.environ["XLA_FLAGS"] += " --xla_gpu_enable_triton_gemm=false"

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    VLABENCH = "vlabench"
    LIBEROPLUS = "liberoplus"
    G2SIM = "g2sim"
    CALVIN = "calvin"
    G1 = "g1"
    S2R_SELECT_COLOR = "s2r_select_color"
    S2R_SIZE_RECOGNIZE = "s2r_size_recognize"
    S2R_GRASP_TARGETS = "s2r_grasp_targets"
    S2R_ORGANIZE_ITEMS = "s2r_organize_items"
    S2R_SORT_FRUIT = "s2r_sort_fruit"
    S2R_PACK_IN_SUPERMARKET = "s2r_pack_in_supermarket"
    S2R_BIMANUAL_CHIP_HANDOVER = "s2r_bimanual_chip_handover"
    S2R_PLACE_BLOCK_INTO_DRAWER = "s2r_place_block_into_drawer"
    PI05_GENIE_SIM_INSTRUCTION_AND_ROBUST = "pi05_genie_sim_instruction_and_robust"
    PI05_GENIE_SIM_SPATIAL = "pi05_genie_sim_spatial"
    PI05_GENIE_SIM_MANIP = "pi05_genie_sim_manip"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    host: str = "0.0.0.0"
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # New diffusion noise every inference instead of one fixed key (checkpoint policies only).
    fresh_noise: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="acot_libero_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/exp_name/40000",
    ),
    EnvMode.LIBEROPLUS: Checkpoint(
        config="acot_libero_plus_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_plus_action_cot_explicit_implicit_co_fusion/exp_name/100000",
    ),
    EnvMode.VLABENCH: Checkpoint(
        config="acot_vlabench_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_vlabench_action_cot_explicit_implicit_co_fusion/exp_name/60000",
    ),
    EnvMode.G2SIM: Checkpoint(
        config="acot_icra_simulation_challenge_reasoning_to_action",
        dir="./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/exp_name/30000",
    ),
    EnvMode.S2R_SELECT_COLOR: Checkpoint(
        config="s2r_select_color",
        dir="./checkpoints/select_color/",
    ),
    EnvMode.S2R_SIZE_RECOGNIZE: Checkpoint(
        config="s2r_size_recognize",
        dir="./checkpoints/size_recognize/",
    ),
    EnvMode.S2R_GRASP_TARGETS: Checkpoint(
        config="s2r_grasp_targets",
        dir="./checkpoints/grasp_targets/",
    ),
    EnvMode.S2R_ORGANIZE_ITEMS: Checkpoint(
        config="s2r_organize_items",
        dir="./checkpoints/organize_items/",
    ),
    EnvMode.S2R_SORT_FRUIT: Checkpoint(
        config="s2r_sort_fruit",
        dir="./checkpoints/sort_fruit/",
    ),
    EnvMode.S2R_PACK_IN_SUPERMARKET: Checkpoint(
        config="s2r_pack_in_supermarket",
        dir="./checkpoints/pack_in_supermarket/",
    ),
    EnvMode.S2R_BIMANUAL_CHIP_HANDOVER: Checkpoint(
        config="s2r_bimanual_chip_handover",
        dir="./checkpoints/bimanual_chip_handover/",
    ),
    EnvMode.S2R_PLACE_BLOCK_INTO_DRAWER: Checkpoint(
        config="s2r_place_block_into_drawer",
        dir="./checkpoints/place_block_into_drawer/",
    ),
     EnvMode.PI05_GENIE_SIM_INSTRUCTION_AND_ROBUST: Checkpoint(
        config="pi05_genie_sim_instruction_and_robust_20260526",
        dir="./checkpoints/instruction_and_robust/",
    ),
    EnvMode.PI05_GENIE_SIM_SPATIAL: Checkpoint(
        config="pi05_genie_sim_spatial_20260528",
        dir="./checkpoints/spatial/",
    ),
    EnvMode.PI05_GENIE_SIM_MANIP: Checkpoint(
        config="pi05_genie_sim_manip_20260613",
        dir="./checkpoints/manipulation/",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                fresh_noise=args.fresh_noise,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
