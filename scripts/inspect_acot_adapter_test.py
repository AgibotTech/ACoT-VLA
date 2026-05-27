from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import ClassVar

import numpy as np

from scripts import inspect_acot_adapter as inspect


def test_summarize_mask_handles_none_and_empty():
    none_summary = inspect.summarize_mask("state_mask", None)
    assert none_summary["present"] is False
    assert none_summary["masked_indices"] == []

    empty_summary = inspect.summarize_mask("action_mask", [])
    assert empty_summary["present"] is True
    assert empty_summary["length"] == 0
    assert empty_summary["masked_count"] == 0


def test_summarize_mask_reports_masked_indices_and_oversize_warning():
    summary = inspect.summarize_mask("state_mask", [False, True, False, True], sample_dim=3)

    assert summary["present"] is True
    assert summary["length"] == 4
    assert summary["masked_indices"] == [1, 3]
    assert summary["warnings"]


def test_validate_sample_reports_missing_state_bad_action_horizon_camera_and_prompt():
    result = inspect.validate_sample(
        {"actions": np.zeros((3, 7), dtype=np.float32)},
        action_horizon=10,
        action_dim=32,
        expected_cameras=("top_head",),
        requires_prompt=True,
    )

    assert result["ok"] is False
    assert any("Missing state" in error for error in result["errors"])
    assert any("horizon 3 is shorter" in error for error in result["errors"])
    assert any("Missing expected camera" in error for error in result["errors"])
    assert any("Missing prompt" in error for error in result["errors"])


def test_validate_sample_reports_bad_action_rank():
    result = inspect.validate_sample(
        {
            "state": np.zeros((21,), dtype=np.float32),
            "actions": np.zeros((7,), dtype=np.float32),
            "prompt": np.array("pick"),
        },
        action_horizon=10,
        action_dim=32,
        requires_prompt=True,
    )

    assert result["ok"] is False
    assert any("must have shape" in error for error in result["errors"])


def test_validate_sample_accepts_camera_named_keys_when_repack_paths_exist():
    result = inspect.validate_sample(
        {
            "state": np.zeros((21,), dtype=np.float32),
            "actions": np.zeros((10, 21), dtype=np.float32),
            "prompt": np.array("pick"),
            "top_head": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        action_horizon=10,
        action_dim=32,
        expected_cameras=("top_head",),
        expected_image_keys=("observation.images.top_head",),
        requires_prompt=True,
    )

    assert result["ok"] is True


def test_build_report_with_fake_config_is_json_stable(monkeypatch):
    monkeypatch.setattr(inspect, "_load_train_config", lambda _: _FakeTrainConfig())

    report = inspect.build_report("fake_acot")
    decoded = json.loads(json.dumps(report))

    assert decoded["config_name"] == "fake_acot"
    assert set(decoded) >= {
        "config_name",
        "model",
        "data",
        "prompt",
        "transforms",
        "expected_cameras",
        "expected_image_keys",
        "masks",
        "notes",
    }
    assert decoded["model"]["action_horizon"] == 10
    assert decoded["model"]["coarse_action_horizon"] == 15
    assert decoded["expected_cameras"] == ["top_head"]
    assert decoded["masks"]["state_mask"]["masked_indices"] == [1]


@dataclasses.dataclass(frozen=True)
class _FakeModel:
    model_type: str = "ACOT_VLA_PI05"
    action_dim: int = 32
    action_horizon: int = 10
    coarse_action_horizon: int = 15
    adopt_explicit_action_reasoner: bool = True
    adopt_implicit_action_reasoner: bool = True


@dataclasses.dataclass(frozen=True)
class _FakeGroup:
    inputs: Sequence[object] = ()
    outputs: Sequence[object] = ()


@dataclasses.dataclass(frozen=True)
class _FakeRepackTransform:
    structure: dict[str, object]


@dataclasses.dataclass(frozen=True)
class _FakeRobotInputs:
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head",)

    action_dim: int = 32
    state_mask: tuple[bool, ...] = (False, True, False)
    action_mask: tuple[bool, ...] = (True, False, False)


@dataclasses.dataclass(frozen=True)
class ACOTDeltaActions:
    mask: tuple[bool, ...] = (True, True, False)
    use_delta_joint_actions: tuple[bool, ...] = (False, True)


@dataclasses.dataclass(frozen=True)
class _FakeDataConfig:
    repo_id: str = "fake/repo"
    asset_id: str = "fake_asset"
    norm_stats: None = None
    repack_transforms: _FakeGroup = dataclasses.field(
        default_factory=lambda: _FakeGroup(
            inputs=(_FakeRepackTransform({"images": {"top_head": "observation.images.top_head"}}),)
        )
    )
    data_transforms: _FakeGroup = dataclasses.field(
        default_factory=lambda: _FakeGroup(inputs=(_FakeRobotInputs(), ACOTDeltaActions()))
    )
    model_transforms: _FakeGroup = dataclasses.field(default_factory=_FakeGroup)
    action_sequence_keys: tuple[str, ...] = ("action",)
    prompt_from_task: bool = False
    prompt_from_hl_instruction: bool = False


@dataclasses.dataclass(frozen=True)
class _FakeDataFactory:
    repo_id: str = "fake/repo"
    default_prompt: str = "pick the object"

    def create(self, assets_dirs: pathlib.Path, model_config: _FakeModel) -> _FakeDataConfig:
        del assets_dirs, model_config
        return _FakeDataConfig()


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    model: _FakeModel = dataclasses.field(default_factory=_FakeModel)
    data: _FakeDataFactory = dataclasses.field(default_factory=_FakeDataFactory)
    assets_dirs: pathlib.Path = pathlib.Path("assets/fake_acot")
