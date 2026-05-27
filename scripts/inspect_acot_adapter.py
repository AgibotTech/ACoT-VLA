"""Inspect ACoT-VLA adapter requirements without loading checkpoints.

This script is intentionally lightweight: it reads training/data config metadata,
describes transform expectations, and optionally validates a small ``.npz`` sample.
It does not instantiate model weights, start a policy server, or run inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import json
import pathlib
import sys
from typing import Any

import numpy as np

MASK_EXPLANATIONS = {
    "state_mask": "State dimensions intentionally zeroed before model input.",
    "action_mask": "Action dimensions intentionally zeroed before training/inference transforms.",
    "delta_action_mask": "Action dimensions converted between absolute and delta action spaces.",
}

GRIPPER_MASK_NOTE = (
    "Practical note from the maintainers: gripper-like binary state can sometimes be hidden so the model "
    "learns opening/closing behavior from visual context and coarse action guidance instead of directly "
    "conditioning on the gripper state."
)

PROMPT_KEYS = ("prompt", "task_instruction", "instruction")


def summarize_mask(
    name: str, mask: Sequence[bool] | np.ndarray | None, sample_dim: int | None = None
) -> dict[str, Any]:
    """Return a JSON-safe summary for a boolean mask."""
    explanation = MASK_EXPLANATIONS.get(name, "Boolean mask used by an adapter transform.")
    if mask is None:
        return {
            "name": name,
            "present": False,
            "length": 0,
            "masked_count": 0,
            "masked_indices": [],
            "explanation": explanation,
            "warnings": [],
        }

    values = np.asarray(mask, dtype=bool).reshape(-1)
    masked_indices = np.flatnonzero(values).astype(int).tolist()
    warnings = []
    if sample_dim is not None and values.size > sample_dim:
        warnings.append(f"{name} length {values.size} is larger than sample dimension {sample_dim}.")
    if sample_dim is not None and masked_indices and max(masked_indices) >= sample_dim:
        warnings.append(f"{name} masks at least one index outside sample dimension {sample_dim}.")

    return {
        "name": name,
        "present": True,
        "length": int(values.size),
        "masked_count": int(values.sum()),
        "masked_indices": masked_indices,
        "explanation": explanation,
        "warnings": warnings,
    }


def validate_sample(
    sample: Mapping[str, np.ndarray],
    *,
    action_horizon: int | None,
    action_dim: int | None,
    coarse_action_horizon: int | None = None,
    expected_cameras: Sequence[str] = (),
    expected_image_keys: Sequence[str] = (),
    requires_prompt: bool = True,
    masks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a small sample dictionary against adapter-level expectations."""
    errors: list[str] = []
    warnings: list[str] = []
    keys = tuple(sample.keys())

    state_key = _first_present(sample, ("state", "observation/state", "observation.state"))
    action_key = _first_present(sample, ("actions", "action"))
    coarse_key = _first_present(sample, ("coarse_actions", "coarse/action", "coarse.action"))

    if state_key is None:
        errors.append("Missing state array. Expected one of: state, observation/state, observation.state.")
        state_dim = None
    else:
        state = np.asarray(sample[state_key])
        state_dim = int(state.shape[-1]) if state.ndim else None
        if state.ndim == 0:
            errors.append(f"{state_key} must have at least one dimension.")
        _extend_mask_warnings(warnings, masks, "state_mask", state_dim)

    if action_key is None:
        warnings.append("No actions/action array found. Shape checks for training targets were skipped.")
    else:
        _validate_action_array(
            errors,
            warnings,
            np.asarray(sample[action_key]),
            name=action_key,
            expected_horizon=action_horizon,
            action_dim=action_dim,
        )
        action_dim_observed = (
            int(np.asarray(sample[action_key]).shape[-1]) if np.asarray(sample[action_key]).ndim else None
        )
        _extend_mask_warnings(warnings, masks, "action_mask", action_dim_observed)
        _extend_mask_warnings(warnings, masks, "delta_action_mask", action_dim_observed)

    if coarse_key is not None:
        _validate_action_array(
            errors,
            warnings,
            np.asarray(sample[coarse_key]),
            name=coarse_key,
            expected_horizon=coarse_action_horizon,
            action_dim=action_dim,
        )

    errors.extend(
        f"Missing expected camera '{camera}'. Accepted keys include: {', '.join(_camera_aliases(camera))}."
        for camera in expected_cameras
        if _first_present(sample, _camera_aliases(camera)) is None
    )
    image_keys_to_validate = () if expected_cameras else expected_image_keys
    errors.extend(
        f"Missing expected image key '{image_key}'. Accepted aliases include: {', '.join(_path_aliases(image_key))}."
        for image_key in image_keys_to_validate
        if _first_present(sample, _path_aliases(image_key)) is None
    )

    if requires_prompt and _first_present(sample, PROMPT_KEYS) is None:
        errors.append("Missing prompt/instruction. Add a prompt key unless this config injects one from task metadata.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "sample_keys": list(keys),
    }


def build_report(config_name: str, sample_npz: pathlib.Path | None = None) -> dict[str, Any]:
    """Build the adapter inspection report for a config name."""
    train_config = _load_train_config(config_name)
    model = train_config.model
    data_factory = train_config.data
    data_config, data_config_error = _safe_create_data_config(train_config)

    transform_groups = _collect_transform_groups(data_config)
    transforms = [transform for group in transform_groups.values() for transform in group]
    masks = _collect_masks(data_factory, transforms)
    prompt = _prompt_summary(data_factory, data_config, transforms)
    sample = _load_npz(sample_npz) if sample_npz else None

    report = {
        "config_name": config_name,
        "model": {
            "type": _stringify(getattr(model, "model_type", None)),
            "action_dim": _jsonable(getattr(model, "action_dim", None)),
            "action_horizon": _jsonable(getattr(model, "action_horizon", None)),
            "coarse_action_horizon": _jsonable(getattr(model, "coarse_action_horizon", None)),
            "explicit_action_reasoner": _jsonable(getattr(model, "adopt_explicit_action_reasoner", None)),
            "implicit_action_reasoner": _jsonable(getattr(model, "adopt_implicit_action_reasoner", None)),
        },
        "data": _data_summary(train_config, data_factory, data_config, data_config_error),
        "prompt": prompt,
        "transforms": {
            stage: [_describe_transform(transform) for transform in items] for stage, items in transform_groups.items()
        },
        "expected_cameras": _expected_cameras(transforms),
        "expected_image_keys": _expected_image_keys(transforms),
        "masks": masks,
        "notes": [
            "No model weights, checkpoints, GPUs, simulators, or robot hardware are required for this inspection.",
            GRIPPER_MASK_NOTE,
        ],
    }

    if sample is not None:
        report["sample_validation"] = validate_sample(
            sample,
            action_horizon=getattr(model, "action_horizon", None),
            action_dim=getattr(model, "action_dim", None),
            coarse_action_horizon=getattr(model, "coarse_action_horizon", None),
            expected_cameras=report["expected_cameras"],
            expected_image_keys=report["expected_image_keys"],
            requires_prompt=prompt["requires_prompt_in_sample"],
            masks=masks,
        )

    return report


def format_text_report(report: Mapping[str, Any]) -> str:
    """Format an inspection report for terminal output."""
    lines = [
        f"ACoT adapter inspection: {report['config_name']}",
        "",
        "Model",
        f"  type: {report['model']['type']}",
        f"  action_dim: {report['model']['action_dim']}",
        f"  action_horizon: {report['model']['action_horizon']}",
        f"  coarse_action_horizon: {report['model']['coarse_action_horizon']}",
        f"  explicit_action_reasoner: {report['model']['explicit_action_reasoner']}",
        f"  implicit_action_reasoner: {report['model']['implicit_action_reasoner']}",
        "",
        "Data and assets",
        f"  repo_id: {report['data']['repo_id']}",
        f"  asset_id: {report['data']['asset_id']}",
        f"  config_assets_dir: {report['data']['config_assets_dir']}",
        f"  norm_stats_loaded_from_config_assets: {report['data']['norm_stats_loaded_from_config_assets']}",
        "  policy_server_norm_stats: loaded from checkpoint assets when serving a trained policy",
        "",
        "Prompt",
        f"  source: {report['prompt']['source']}",
        f"  sample_requires_prompt: {report['prompt']['requires_prompt_in_sample']}",
        "",
        "Expected inputs",
        f"  cameras: {_format_list(report['expected_cameras'])}",
        f"  image keys: {_format_list(report['expected_image_keys'])}",
        "",
        "Masks",
    ]

    for name, summary in report["masks"].items():
        lines.extend(
            [
                f"  {name}: present={summary['present']} length={summary['length']} masked_count={summary['masked_count']}",
                f"    masked_indices: {_format_list(summary['masked_indices'])}",
                f"    meaning: {summary['explanation']}",
            ]
        )
        lines.extend(f"    warning: {warning}" for warning in summary["warnings"])

    lines.extend(["", "Transform chain"])
    for stage, transforms in report["transforms"].items():
        lines.append(f"  {stage}: {_format_list([item['name'] for item in transforms])}")

    if report["data"]["data_config_error"]:
        lines.extend(["", f"Data config warning: {report['data']['data_config_error']}"])

    if "sample_validation" in report:
        validation = report["sample_validation"]
        lines.extend(["", f"Sample validation: {'ok' if validation['ok'] else 'failed'}"])
        lines.extend(f"  error: {error}" for error in validation["errors"])
        lines.extend(f"  warning: {warning}" for warning in validation["warnings"])

    lines.extend(["", "Notes"])
    lines.extend(f"  - {note}" for note in report["notes"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config name to inspect.")
    parser.add_argument("--sample-npz", type=pathlib.Path, help="Optional small .npz sample to validate.")
    parser.add_argument("--format", choices=("text", "json"), default="text", dest="output_format")
    args = parser.parse_args(argv)

    report = build_report(args.config_name, args.sample_npz)
    if args.output_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text_report(report))
    return 0


def _load_train_config(config_name: str) -> Any:
    from openpi.training import config as _config  # noqa: PLC0415

    return _config.get_config(config_name)


def _safe_create_data_config(train_config: Any) -> tuple[Any | None, str | None]:
    try:
        return train_config.data.create(train_config.assets_dirs, train_config.model), None
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"

    data_factory = train_config.data
    original_load_norm_stats = getattr(data_factory, "_load_norm_stats", None)
    if original_load_norm_stats is None:
        return None, first_error

    def _skip_norm_stats(*_: Any, **__: Any) -> None:
        return None

    try:
        object.__setattr__(data_factory, "_load_norm_stats", _skip_norm_stats)
        data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    except Exception as exc:
        return None, f"{first_error}; fallback without norm stats also failed with {type(exc).__name__}: {exc}"
    finally:
        object.__setattr__(data_factory, "_load_norm_stats", original_load_norm_stats)

    return data_config, f"{first_error}; inspected transforms with norm-stat loading disabled."


def _collect_transform_groups(data_config: Any | None) -> dict[str, list[Any]]:
    if data_config is None:
        return {"repack_inputs": [], "data_inputs": [], "model_inputs": [], "model_outputs": [], "data_outputs": []}
    return {
        "repack_inputs": list(getattr(getattr(data_config, "repack_transforms", None), "inputs", ())),
        "data_inputs": list(getattr(getattr(data_config, "data_transforms", None), "inputs", ())),
        "model_inputs": list(getattr(getattr(data_config, "model_transforms", None), "inputs", ())),
        "model_outputs": list(getattr(getattr(data_config, "model_transforms", None), "outputs", ())),
        "data_outputs": list(getattr(getattr(data_config, "data_transforms", None), "outputs", ())),
    }


def _collect_masks(data_factory: Any, transforms: Sequence[Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, Any] = {}
    for name in MASK_EXPLANATIONS:
        if hasattr(data_factory, name):
            values[name] = getattr(data_factory, name)
    for transform in transforms:
        if hasattr(transform, "mask") and transform.__class__.__name__ in {"DeltaActions", "ACOTDeltaActions"}:
            values.setdefault("delta_action_mask", transform.mask)
        for name in ("state_mask", "action_mask"):
            if hasattr(transform, name):
                values.setdefault(name, getattr(transform, name))
    return {name: summarize_mask(name, values.get(name)) for name in MASK_EXPLANATIONS}


def _prompt_summary(data_factory: Any, data_config: Any | None, transforms: Sequence[Any]) -> dict[str, Any]:
    default_prompt = getattr(data_factory, "default_prompt", None)
    for transform in transforms:
        if transform.__class__.__name__ == "InjectDefaultPrompt" and getattr(transform, "prompt", None) is not None:
            default_prompt = transform.prompt
            break

    prompt_from_task = bool(getattr(data_config, "prompt_from_task", False)) if data_config is not None else False
    prompt_from_hl = (
        bool(getattr(data_config, "prompt_from_hl_instruction", False)) if data_config is not None else False
    )
    if prompt_from_hl:
        source = "high-level instruction segments from dataset metadata"
    elif prompt_from_task:
        source = "LeRobot task metadata"
    elif default_prompt is not None:
        source = "default prompt injected by config"
    else:
        source = "sample or inference request must provide prompt"
    return {
        "source": source,
        "default_prompt": _jsonable(default_prompt),
        "requires_prompt_in_sample": not (prompt_from_task or prompt_from_hl or default_prompt is not None),
    }


def _data_summary(train_config: Any, data_factory: Any, data_config: Any | None, error: str | None) -> dict[str, Any]:
    return {
        "factory": data_factory.__class__.__name__,
        "repo_id": _jsonable(getattr(data_config, "repo_id", getattr(data_factory, "repo_id", None))),
        "asset_id": _jsonable(
            getattr(data_config, "asset_id", getattr(getattr(data_factory, "assets", None), "asset_id", None))
        ),
        "action_sequence_keys": _jsonable(getattr(data_config, "action_sequence_keys", None)),
        "config_assets_dir": str(train_config.assets_dirs),
        "norm_stats_loaded_from_config_assets": bool(getattr(data_config, "norm_stats", None))
        if data_config is not None
        else False,
        "checkpoint_required_for_policy_server": True,
        "data_config_error": error,
    }


def _describe_transform(transform: Any) -> dict[str, Any]:
    fields = {}
    for name in _public_field_names(transform):
        value = getattr(transform, name)
        if name in {"tokenizer", "instruction_segments", "prompt_map_inject_to_training"}:
            fields[name] = value.__class__.__name__ if value is not None else None
        else:
            fields[name] = _jsonable(value)
    return {"name": transform.__class__.__name__, "fields": fields}


def _public_field_names(obj: Any) -> list[str]:
    if dataclasses.is_dataclass(obj):
        return [field.name for field in dataclasses.fields(obj) if not field.name.startswith("_")]
    return [name for name in vars(obj) if not name.startswith("_")]


def _expected_cameras(transforms: Sequence[Any]) -> list[str]:
    cameras: list[str] = []
    for transform in transforms:
        for camera in getattr(transform, "EXPECTED_CAMERAS", ()):
            if camera not in cameras:
                cameras.append(camera)
    return cameras


def _expected_image_keys(transforms: Sequence[Any]) -> list[str]:
    image_keys: list[str] = []
    for transform in transforms:
        structure = getattr(transform, "structure", None)
        if structure is None:
            continue
        for output_key, input_key in _flatten_mapping(structure).items():
            if "image" in output_key and isinstance(input_key, str) and input_key not in image_keys:
                image_keys.append(input_key)
    return image_keys


def _flatten_mapping(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, Mapping):
        result = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_mapping(value, child_prefix))
        return result
    return {prefix: tree}


def _load_npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validate_action_array(
    errors: list[str],
    warnings: list[str],
    value: np.ndarray,
    *,
    name: str,
    expected_horizon: int | None,
    action_dim: int | None,
) -> None:
    if value.ndim < 2:
        errors.append(f"{name} must have shape [horizon, dim] or [batch, horizon, dim]; got {value.shape}.")
        return

    horizon = int(value.shape[-2])
    dim = int(value.shape[-1])
    if expected_horizon is not None and horizon < expected_horizon:
        errors.append(f"{name} horizon {horizon} is shorter than expected horizon {expected_horizon}.")
    elif expected_horizon is not None and horizon > expected_horizon:
        warnings.append(
            f"{name} horizon {horizon} is longer than expected horizon {expected_horizon}; adapter slicing may apply."
        )

    if action_dim is not None and dim > action_dim:
        warnings.append(
            f"{name} dim {dim} is larger than model action_dim {action_dim}; adapter slicing/remapping may apply."
        )


def _extend_mask_warnings(
    warnings: list[str],
    masks: Mapping[str, Mapping[str, Any]] | None,
    mask_name: str,
    sample_dim: int | None,
) -> None:
    if masks is None or sample_dim is None:
        return
    mask = masks.get(mask_name)
    if not mask or not mask.get("present"):
        return
    length = int(mask["length"])
    masked_indices = [int(index) for index in mask["masked_indices"]]
    if length > sample_dim:
        warnings.append(f"{mask_name} length {length} is larger than sample dimension {sample_dim}.")
    if masked_indices and max(masked_indices) >= sample_dim:
        warnings.append(f"{mask_name} masks an index outside sample dimension {sample_dim}.")


def _first_present(sample: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in sample:
            return key
    return None


def _camera_aliases(camera: str) -> tuple[str, ...]:
    return (
        camera,
        f"images/{camera}",
        f"images.{camera}",
        f"observation.images.{camera}",
        f"observation/images/{camera}",
    )


def _path_aliases(path: str) -> tuple[str, ...]:
    return (path, path.replace("/", "."), path.replace(".", "/"))


def _format_list(value: Sequence[Any]) -> str:
    if not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, enum.Enum):
        return value.value
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
