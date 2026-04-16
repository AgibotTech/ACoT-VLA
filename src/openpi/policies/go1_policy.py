"""Policy transforms for the Go1 robot."""
import dataclasses
import json
import logging
import pathlib
from typing import ClassVar, Sequence
import numpy as np
import torch
import copy
import openpi.models.model as _model
import openpi.transforms as transforms


def load_field_indices(repo_id: str | list[str], feature_key: str, field_keys: Sequence[str]) -> list[int]:
    if isinstance(repo_id, list):
        repo_id = repo_id[0]
    info_path = pathlib.Path(repo_id) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})
    feature = features.get(feature_key)
    if feature is None:
        raise KeyError(f"Feature '{feature_key}' not found in info.json")

    field_descriptions = feature.get("field_descriptions", {})
    indices: list[int] = []
    for key in field_keys:
        if key not in field_descriptions:
            raise KeyError(f"Field '{key}' not found in field_descriptions of '{feature_key}'")
        indices.extend(field_descriptions[key]["indices"])
    logging.info("Loaded %d indices for %s from %s: fields=%s", len(indices), feature_key, info_path, field_keys)
    return indices

@dataclasses.dataclass(frozen=True)
class Go1Inputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.
    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [N] (variable dimension, remapped via state_indices)
    - actions: [action_horizon, M] (variable dimension, remapped via action_indices)
    """
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    state_indices: tuple[int, ...] | None = None
    action_indices: tuple[int, ...] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }

    def _remap_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_indices is not None:
            size = state.shape[-1]
            if size > max(self.state_indices):
                state = state[list(self.state_indices)]
        if self.state_mask is not None:
            if len(state) > len(self.state_mask):
                state = state[:len(self.state_mask)]
            state[self.state_mask[:len(state)]] = 0
        return state

    def _remap_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_indices is not None and actions.shape[1] > max(self.action_indices):
            actions = actions[:, list(self.action_indices)]
        if self.action_mask is not None:
            if actions.shape[1] > len(self.action_mask):
                actions = actions[:, :len(self.action_mask)]
            actions[:, self.action_mask[:actions.shape[1]]] = 0
        return actions

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = copy.deepcopy(state)
        state = self._remap_state(state)
        state = state.squeeze()

        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if "actions" in data:
            actions = data["actions"]
            actions = self._remap_actions(actions)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class Go1Outputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])}


@dataclasses.dataclass(frozen=True)
class Go1ACOTInputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [32]
    - actions: [action_horizon, 22]
    """

    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    acot_action_generation: Sequence[Sequence[int]] | None = None

    def __call__(self, data: dict) -> dict:
        if len(data["state"]) == 190:
            indices = list(range(54, 68)) + [0, 1]
            data["state"] = data["state"][indices]
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        state = copy.deepcopy(state)
        # state[14:]  = state[14:] * 120
        if self.state_mask is not None:
            state[self.state_mask] = 0

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}


        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }


        if self.acot_action_generation is not None and "actions" in data:
            action_horizons = self.acot_action_generation[0]
            joint_action_shifts = self.acot_action_generation[1]

            raw_data = data["actions"]
            keys = ["coarse_actions", "actions"]
            for idx, key in enumerate(keys):
                action_horizon = action_horizons[idx]
                joint_action_shift = joint_action_shifts[idx]
                required_length = (action_horizon - 1) * joint_action_shift + 1
                data[key] = copy.deepcopy(raw_data[:required_length:joint_action_shift])
                assert len(data[key]) == action_horizon
        
        for key in ['coarse_actions', 'actions']:
            if key in data:
                if data[key].shape[1] == 36:
                    data[key] = np.column_stack((data[key][:, 16:30], data[key][:, 0], data[key][:, 1]))
                if self.action_mask is not None:
                    data[key][:, self.action_mask[:data[key].shape[1]]] = 0
                data[key] = transforms.pad_to_dim(data[key], self.action_dim)
                inputs[key] = data[key]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go1ACOTOutputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""

    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        return {key: np.asarray(data[key][:, :16]) for key in keys if key in data}