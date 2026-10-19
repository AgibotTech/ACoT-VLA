"""Policy transforms for the ARX robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch
import copy
from collections.abc import Sequence
import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class ARXInputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [32]
    - actions: [action_horizon, 22]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)

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

        # Add actions if present
        if "actions" in data:
            actions = data["actions"]
            actions = transforms.pad_to_dim(actions, self.action_dim)
            if self.action_mask is not None:
                actions[:, self.action_mask] = 0
            if actions.ndim > 2 and actions.shape[0] == 1:
                actions = actions.squeeze()
            inputs["actions"] = actions

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ARXOutputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}  # HACK hardcode


@dataclasses.dataclass(frozen=True)
class ARXACOTInputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [32]
    - actions: [action_horizon, 22]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }
    acot_action_generation: Sequence[Sequence[int]] | None = None


    def __call__(self, data: dict) -> dict:
        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)

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

        # Add actions if present
        for key in ['coarse_actions', 'actions']:
            if key in data:
                actions = data[key]
                actions = transforms.pad_to_dim(actions, self.action_dim)
                if self.action_mask is not None:
                    actions[:, self.action_mask] = 0
                if actions.ndim > 2 and actions.shape[0] == 1:
                    actions = actions.squeeze()
                inputs[key] = actions

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ARXACOTOutputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""

    def __call__(self, data: dict) -> dict:
        keys = ["coarse_actions", "actions"]
        return {key: np.asarray(data[key][:, :14]) for key in keys if key in data}
