"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, Dict, Union
import os
import numpy as np
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.acot_vla as acot_vla
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.agilex_policy as agilex_policy
import openpi.policies.go1_policy as go1_policy
import openpi.policies.go2_policy as go2_policy
import openpi.policies.vlabench_policy as vlabench_policy
import openpi.policies.arx_policy as arx_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    prompt_from_hl_instruction: bool = False

    dataloader_sampler: str | None = ''

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # True: center-crop images to square, then resize to 224 (no black letterbox bars).
    center_crop_images: bool = False

    def _resize_images(self) -> _transforms.DataTransformFn:
        if self.center_crop_images:
            return _transforms.CenterCropResizeImages(224, 224)
        return _transforms.ResizeImages(224, 224)

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        self._resize_images(),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        self._resize_images(),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.ACOT_VLA_PI0:
                assert isinstance(model_config, acot_vla.ACOTConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        self._resize_images(),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.ACOTPadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.ACOT_VLA_PI05:
                assert isinstance(model_config, acot_vla.ACOTConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        self._resize_images(),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.ACOTPadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        self._resize_images(),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    state_mask = None
    action_mask = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        norm_stats_path = getattr(self, 'norm_stats_path', None)
        if isinstance(norm_stats_path, str):
            if os.path.exists(norm_stats_path):
                norm_stats = self._load_norm_stats_from_json(norm_stats_path)
            else:
                logging.warning("norm_stats_path=%s does not exist, falling back to assets.", norm_stats_path)
                norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        else:
            norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=norm_stats,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _aggregate_norm_stats(
        self, all_norm_stats: list[dict[str, _transforms.NormStats]]
    ) -> dict[str, _transforms.NormStats]:
        from openpi.shared.normalize import NormStats

        agg = {}
        for key in all_norm_stats[0].keys():
            agg[key] = NormStats(
                mean=np.mean([ns[key].mean for ns in all_norm_stats], axis=0),
                std=np.mean([ns[key].std for ns in all_norm_stats], axis=0),
                q01=np.mean([ns[key].q01 for ns in all_norm_stats], axis=0),
                q99=np.mean([ns[key].q99 for ns in all_norm_stats], axis=0),
            )
        return agg

    def _apply_norm_stats_mask(self, norm_stats: dict[str, _transforms.NormStats]) -> None:
        if self.state_mask is not None and "state" in norm_stats:
            state_mask = np.asarray(self.state_mask)
            state_len = norm_stats["state"].std.shape[0]
            dims = min(state_mask.shape[-1], state_len)
            if dims > 0:
                m = state_mask[:dims]
                norm_stats["state"].std[m] = 1e6
                norm_stats["state"].mean[m] = 0
                norm_stats["state"].q01[m] = 0
                norm_stats["state"].q99[m] = 0

        if self.action_mask is not None and "actions" in norm_stats:
            action_mask = np.asarray(self.action_mask)
            action_len = norm_stats["actions"].std.shape[0]
            dims = min(action_mask.shape[-1], action_len)
            if dims > 0:
                m = action_mask[:dims]
                norm_stats["actions"].std[m] = 1e6
                norm_stats["actions"].mean[m] = 0
                norm_stats["actions"].q01[m] = 0
                norm_stats["actions"].q99[m] = 0

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            if not isinstance(asset_id, list):
                asset_id = [asset_id]

            all_norm_stats = []
            for a_id in asset_id:
                data_assets_dir = str(assets_dir / a_id)
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
                all_norm_stats.append(norm_stats)

            norm_stats = self._aggregate_norm_stats(all_norm_stats)
            self._apply_norm_stats_mask(norm_stats)
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    def _load_norm_stats_from_json(self, json_path: str) -> dict[str, _transforms.NormStats] | None:
        if json_path is None:
            return None
        try:
            from openpi.shared.normalize import deserialize_json

            if not isinstance(json_path, list):
                json_path = [json_path]

            all_norm_stats = []
            for j_path in json_path:
                json_file_path = pathlib.Path(j_path)
                norm_stats = deserialize_json(json_file_path.read_text())
                logging.info(f"Loaded norm stats from {json_file_path}")
                all_norm_stats.append(norm_stats)

            norm_stats = self._aggregate_norm_stats(all_norm_stats)
            self._apply_norm_stats_mask(norm_stats)
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {json_file_path}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotVLABenchDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[vlabench_policy.VLABenchInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[vlabench_policy.VLABenchOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotACOTVLABenchDataConfig(DataConfigFactory):
    repo_id: Union[str, Sequence[str]] = "..."
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[vlabench_policy.VLABenchACOTInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[vlabench_policy.VLABenchACOTOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotACOTGo1DataConfig(DataConfigFactory):
    """
    Configuration for the Go1 robot dataset.
    This config handles the data transforms for the Go1 robot's multi-camera setup and state/action space.
    """
    prompt_from_hl_instruction: bool = False

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action"
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: list(_transforms.make_bool_mask(-14, 18))
    )
    action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: list(_transforms.make_bool_mask(-16, 16))
    )
    delta_action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(14, -18)
    )
    state_keys: Sequence[str] = ("state/joint/position", "state/left_effector/position", "state/right_effector/position")
    action_keys: Sequence[str] = ("action/joint/position", "action/left_effector/position", "action/right_effector/position")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Resolve dataset field indices from info.json field_descriptions so any layout
        # (legacy 190, v21 183, ...) is remapped dynamically instead of hardcoded.
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        state_indices = None
        action_indices = None
        if repo_id is not None:
            try:
                state_indices = tuple(go1_policy.load_field_indices(repo_id, "observation.state", tuple(self.state_keys)))
                action_indices = tuple(go1_policy.load_field_indices(repo_id, "action", tuple(self.action_keys)))
            except (FileNotFoundError, KeyError) as e:
                logging.warning("Could not load field indices from info.json, falling back to no remapping: %s", e)

        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[go1_policy.Go1ACOTInputs(
                action_dim=model_config.action_dim,
                state_mask = self.state_mask,
                action_mask = self.action_mask,
                state_indices=state_indices,
                action_indices=action_indices,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
            ],
            outputs=[go1_policy.Go1ACOTOutputs()],
        )

        # Apply delta action transform if enabled
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(self.delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(self.delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotACOTGo2DataConfig(DataConfigFactory):
    """
    Configuration for the Go2 robot dataset.
    This config handles the data transforms for the Go2 robot's multi-camera setup and state/action space.
    """
    prompt_from_hl_instruction: bool = False

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action"
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask: Sequence[int] = dataclasses.field(
        # mask two gripper state and useless four waist joint states.
        default_factory=lambda: _transforms.make_bool_mask(-14, 2, 4, -1, 11)
    )
    action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(-16, 4, -1, 11)
    )
    delta_action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(14, -18)
    )
    prompt_map_inject_to_training: Dict[str, Sequence[str]] = dataclasses.field(
        default_factory=lambda: {}
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[go2_policy.Go2ACOTInputs(
                action_dim=model_config.action_dim,
                state_mask = self.state_mask,
                action_mask = self.action_mask,
                prompt_map_inject_to_training = self.prompt_map_inject_to_training,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
            ],
            outputs=[go2_policy.Go2ACOTOutputs()],
        )

        # Apply delta action transform if enabled
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(self.delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(self.delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config

@dataclasses.dataclass(frozen=True)
class LeRobotACOTLiberoDataConfig(DataConfigFactory):

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoACOTInputs(
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[libero_policy.LiberoACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LeRobotACOTLiberoPlusDataConfig(DataConfigFactory):

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoACOTInputs(
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[libero_policy.LiberoACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LerobotAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                ),
                agilex_policy.AgilexInterleaveToGroup(),
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # mask state out (set to all zeros)
    mask_state: bool = False

    # if convert to eef position
    convert_to_eef_position: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                    convert_to_eef_position=self.convert_to_eef_position,
                )
            ],
            outputs=[agilex_policy.AgilexOutputs()],
        )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(12, -2)  # grouped: arm 0-11 delta, gripper 12,13 absolute
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class LerobotACOTAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                ),
                agilex_policy.AgilexInterleaveToGroup(),
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # if convert to eef position
    convert_to_eef_position: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexACOTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    convert_to_eef_position=self.convert_to_eef_position,
                    acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts)
                )
            ],
            outputs=[agilex_policy.AgilexACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(12, -2)  # grouped: arm 0-11 delta, gripper 12,13 absolute
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotARXDataConfig(DataConfigFactory):
    """
    Configuration for the Lerobot ARX dataset.
    This config handles the data transforms for the Lerobot ARX's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask = np.array(_transforms.make_bool_mask(-14, 18))
    action_mask = np.array(_transforms.make_bool_mask(-14, 18))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ARXInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=self.state_mask,
                    action_mask=self.action_mask,
                )
            ],
            outputs=[arx_policy.ARXOutputs()],
        )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(12, -2)  # grouped: arm 0-11 delta, gripper 12,13 absolute

        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class LerobotACOTARXDataConfig(DataConfigFactory):
    """
    Configuration for the Lerobot ARX dataset.
    This config handles the data transforms for the Lerobot ARX's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask = np.array(_transforms.make_bool_mask(-14, 18))
    action_mask = np.array(_transforms.make_bool_mask(-14, 18))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ARXACOTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=self.state_mask,
                    action_mask=self.action_mask,
                    acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts)
                )
            ],
            outputs=[arx_policy.ARXACOTOutputs()],
        )

        # Apply delta action transform if enabled
        delta_action_mask = _transforms.make_bool_mask(12, -2)  # grouped: arm 0-11 delta, gripper 12,13 absolute
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotGo1DataConfig(DataConfigFactory):
    # When True, waist dims 16-19 are left live instead of being zeroed. Only
    # valid with norm_stats whose q01/q99 for those dims are real -- with the
    # checkpoint's degenerate q01==q99==0 the quantile transform would map them
    # to ~1e6 rather than into [-1, 1].
    unmask_waist: bool = False
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    norm_stats_path: str | None = None

    state_keys: Sequence[str] = ("state/joint/position", "state/left_effector/position", "state/right_effector/position")
    action_keys: Sequence[str] = ("action/joint/position", "action/left_effector/position", "action/right_effector/position")

    # If True, append waist (5 dim) to state/action keys so dims 16..20 carry waist
    # joint position instead of zero padding. Requires state/waist/position and
    # action/waist/position fields in the dataset's meta/info.json.
    include_waist: bool = False

    # If True, images are center-cropped to square then resized to 224 instead of
    # letterboxed. Applies at training and serving (both use model_transforms).
    center_crop_images: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = ("action",)

    mask_gripper_state: bool = False
    output_dim: int = 22

    state_mask = np.array(_transforms.make_bool_mask(-16, 16))
    action_mask = np.array(_transforms.make_bool_mask(-16, 16))
    # If None, create() picks the default based on include_waist.
    delta_action_mask: Sequence[bool] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None

        state_keys = tuple(self.state_keys)
        action_keys = tuple(self.action_keys)
        if self.include_waist:
            state_keys = state_keys + ("state/waist/position",)
            action_keys = action_keys + ("action/waist/position",)

        state_indices = None
        action_indices = None
        if repo_id is not None:
            try:
                state_indices = tuple(go1_policy.load_field_indices(repo_id, "observation.state", state_keys))
                action_indices = tuple(go1_policy.load_field_indices(repo_id, "action", action_keys))
            except (FileNotFoundError, KeyError) as e:
                logging.warning("Could not load field indices from info.json, falling back to no remapping: %s", e)

        # include_waist and mask_gripper_state are independent controls:
        #   mask_gripper_state -> zero gripper proprioception (state dims 14,15)
        #   include_waist      -> keep only waist[4] (dim 20), zero waist[0..3] (dims 16-19)
        # Grippers in the action are always supervised; mask_gripper_state only affects state.
        action_dim = model_config.action_dim
        if self.include_waist:
            assert action_dim >= 21, f"action_dim ({action_dim}) must be >= 21 for include_waist=True"
        state_mask = np.zeros(action_dim, dtype=bool)
        action_mask = np.zeros(action_dim, dtype=bool)
        if self.mask_gripper_state:
            state_mask[14:16] = True
        if self.include_waist:
            if not self.unmask_waist:
                state_mask[16:20] = True   # waist[0..3] unused; keep waist[4] (dim 20)
                action_mask[16:20] = True
            state_mask[21:] = True
            action_mask[21:] = True
        else:
            state_mask[16:] = True     # no waist: drop everything past the grippers
            action_mask[16:] = True

        # Keep the masks used by _apply_norm_stats_mask (read from self) in sync with
        # the runtime mask used by Go1Inputs above. Without this, include_waist=True
        # leaves dim 20 (waist[4]) live at runtime but zeroed in norm_stats, so quantile
        # normalize divides by ~0 and produces values in the hundreds — blowing up loss.
        object.__setattr__(self, 'state_mask', state_mask)
        object.__setattr__(self, 'action_mask', action_mask)

        data_transforms = _transforms.Group(
            inputs=[
                go1_policy.Go1Inputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=state_mask,
                    action_mask=action_mask,
                    state_indices=state_indices,
                    action_indices=action_indices,
                )
            ],
            outputs=[go1_policy.Go1Outputs(output_dim=self.output_dim)],
        )

        if self.use_delta_joint_actions:
            if self.delta_action_mask is not None:
                delta_action_mask = self.delta_action_mask
            elif self.include_waist:
                # Match compare/openpi LerobotGo2DataConfig exactly:
                # delta on 14 joints + everything past the 2 grippers (incl. waist).
                # state_mask zeroes waist[0..3] so their delta is a no-op; only
                # waist[4] (dim 20) actually moves through delta space, which is
                # what the compare-repo checkpoint was trained on.
                delta_action_mask = _transforms.make_bool_mask(14, -2, 16)
            else:
                delta_action_mask = _transforms.make_bool_mask(14, -2, 6)

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt, center_crop_images=self.center_crop_images
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "ACOT-VLA"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if not os.getenv("DEBUG_MODE", default=False) == "true":
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
        else:
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / "debug").resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params"
        ),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params"
        ),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets-preview/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-preview/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # action cot related configs
    TrainConfig(
        name="acot_libero_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=False, coarse_action_expert_variant = "gemma_300m", action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LeRobotACOTLiberoDataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/libero_dataset/",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=51_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_dual_ae=[False, False]),
    ),
    # VLABench configs
    TrainConfig(
        name="acot_vlabench_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(action_dim = 7, coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=True, coarse_action_expert_variant = "gemma_300m", action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LeRobotACOTVLABenchDataConfig(
            repo_id=[
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/add_condiment",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/insert_flower",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_book",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_chemistry_tube",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_drink",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_fruit",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_mahjong",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_painting",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_poker",
                "/mnt/public/zhonglinqing/data/datasets/vlabench-dataset/primitive_lerobot/select_toy"
            ],
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=61_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_llm_embedder=False, freeze_dual_ae=[False, False]),
    ),
    # libero plus
    TrainConfig(
        name="acot_libero_plus_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=15, action_horizon=10, pi05=True, discrete_state_input=False, coarse_action_expert_variant = "gemma_300m", action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LeRobotACOTLiberoPlusDataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/libero_plus_dataset",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=101_000,
        save_interval=20000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_dual_ae=[False, False]),
    ),
    # go1 configs
    TrainConfig(
        name="acot_go1_openset_pick_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=False, action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo1DataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/lerobot_dataset/random_pick_3124_3136",
            default_prompt="pick up the item on the table use your arm.",
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1),
            repack_transforms =_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt"
                        }
                    )
                ]
            )
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=51_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_dual_ae=[False, False]),
    ),
    TrainConfig(
        name="acot_go1_wipe_stain_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=True, action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo1DataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/lerobot_dataset/Wipe_the_table_stains",
            default_prompt="Wipe the table stains",
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=51_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_llm_embedder=False, freeze_dual_ae=[False, False]),
    ),
    TrainConfig(
        name="acot_go1_pourwater_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=True, action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo1DataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/lerobot_dataset/pourwater_merged",
            default_prompt="Pour water",
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=240_000,
        save_interval=30000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 48,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 128,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_llm_embedder = False, freeze_dual_ae=[False, False]),
    ),
    # agilex configs
    TrainConfig(
        name="acot_agilex_openset_pick_action_cot_explicit_implicit_co_fusion",
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, pi05=True, discrete_state_input=False, action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTAgilexDataConfig(
            repo_id="/mnt/public/zhonglinqing/data/datasets/lerobot_dataset/agilex_random_pick/merge",
            default_prompt="pick up the item on the table use your arm.",
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
            extra_delta_transform=(False, False),
            joint_action_shifts=(2, 1),
            repack_transforms =_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=51_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 200,
        num_workers=48 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=128 if not os.getenv("DEBUG_MODE", default=False) == "true" else 16,
        freeze_filter=acot_vla.ACOTConfig().get_freeze_filter(freeze_vision = False, freeze_llm = True, freeze_dual_ae=[False, False]),
    ),
    # genie sim 3.0 baseline configs
    TrainConfig(
        name="acot_icra_simulation_challenge_reasoning_to_action",
        # For the ICRA sim challenge, we set both coarse and fine action horizons to 30 since the tasks are relatively long-horizon.
        # We also use both explicit and implicit action reasoners, and use the downsample-based implicit extractor.
        # You can modify these design choices based on the specific tasks and dataset. 
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, paligemma_variant="gemma_2b_lora", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo2DataConfig(
            default_prompt = "This is the icra simulation challenge baseline config. Please refer to the README for details.",
            # Fill in the 9 tasks for training. You can use all 9 tasks, or a subset of them based on your preference.
            repo_id = [
                "/mnt/public/E6/lerobot/7819/task_5833", # Pouring workpieces, single-arm task, uses right hand only
                "/mnt/public/E6/lerobot/7820/task_5832", # Opening a door, single-arm task, uses right hand only
                "/mnt/public/E6/lerobot/8153", # Scooping popcorn, single-arm task, uses right hand only
                "/mnt/public/E6/lerobot/7821/task_5829", # Carrying a pot, dual-arm task, uses both hands simultaneously

                "/mnt/public/E6/lerobot/7837/task_5441", # Grabbing toys, dual-arm task, uses left or right hand based on instruction
                "/mnt/public/E6/lerobot/7944/task_6100", # Supermarket item retrieval, dual-arm task, uses left or right hand based on instruction
                "/mnt/public/E6/lerobot/7818/task_5853", # Supermarket restocking, dual-arm task, uses left or right hand based on instruction
                "/mnt/public/E6/lerobot/8169/2026021101/gripper/task_6167", # packages sorting, grasps objects based on instruction, single-arm task but involves waist movement

                "/mnt/public/E6/lerobot/7878//task_5828", # Arranging the table, dual-arm task, uses both hands simultaneously
            ],
            # Set the asset dir to specify normalization stats calculated from the dataset
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/zhonglinqing/data/datasets/genie_sim_icra_datasets/nine_dataset_merge_assets",
            ),
            # this line defines a mapping from task name to (prompt, probability of replacement) for training. 
            # If the current episode's task name matches one of the keys in the mapping, then with the corresponding probability, 
            # the original prompt (loaded from the dataset based on the task name) will be replaced with the provided prompt. 
            # This allows for more diverse and potentially more informative prompts during training.
            prompt_map_inject_to_training = {
                # task name: (prompt to replace vanilla annotation, probability to replace)
                "Unload workpiece_icra_SIM": ("Pour the workpiece into the box", 0.5),
                "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
                "Make popcorn": ("Scoop the popcorn and pour it into the popcorn bucket", 0.5),
                "Carry the pot": ("Grasp the two handles of the pot and place it on the stove", 0.5),

                "Insert building block holes_2_SIM": (
                    "Pick up the yellow circular block from the table, "
                    "and place it into the round hole of the block box",
                    0.2
                ),
                "Remove misplaced beverages from shelves": (
                    "Pick up the incorrectly placed item from the shelf, "
                    "and place it into the shopping basket",
                    0.2
                ),
                "Stock supermarket shelves  \nStraighten products  \nAttend ICRA conference  \nOperate SIM card": (
                    "Pick up the wei-chuan orange juice in the shopping basket, "
                    "and place it on the shelf. "
                    "Then, straighten the toppled wei-chuan grape juice",
                    0.2
                ),
                "Sort packages": (
                    "Grab the <color> package on the table, "
                    "turn the waist right to face the barcode scanner, "
                    "place the package on the scanning table with the barcode facing up. "
                    "Then, grab the package, "
                    "rotate the waist and place the package in the blue bin. "
                    "Finally, return the waist back to face the initial table",
                    0.2
                ),

                "Clear the desktop": (
                    "Pick up the pen on the left side and place it into the pen holder, "
                    "close the laptop, "
                    "pick up the tissue on the table and place it into the trash bin on the right size. "
                    "Then, pick up the mouse and place it on the right side of the laptop. "
                    "Finally, straighten the colored pencil box",
                    0.5
                ),
            },
            repack_transforms =_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            # repack task name and episode id here for specific prompt replacement in training
                            "task": "task",
                            "episode_index": "episode_index"
                        }
                    )
                ]
            ),
            # this line allows using episode level annotation for training, essential for instruction following
            base_config = DataConfig(dataloader_sampler = "subtask", prompt_from_hl_instruction = True),
            # this line is important for action cot training, it shifts the action sequence by a certain number of steps 
            # to create the input for the coarse action reasoner and the final action head. 
            # You can tune these values based on the characteristics of your dataset. 
            # Generally, we find that having a small positive shift (e.g. 1 or 2) works well.
            joint_action_shifts = (2, 1),
            # if using delta action to train model is excepted, set True in extra_delta_transform
            extra_delta_transform = (True, True),
            # delta_action_mask controls which dimensions of the action are used for delta action transformation.
            delta_action_mask = _transforms.make_bool_mask(14, -18)
        ),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 10_000,
            peak_lr = 5e-5,
            decay_steps = 1_000_000,
            decay_lr = 5e-5,
        ),
        optimizer = _optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay = 0.999,
        weight_loader = weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps = 50_000,
        save_interval = 5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 200,
        num_workers = 24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size = 256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 16,
        # You can select to freeze certain parts of the model during training by setting the corresponding flags to True
        freeze_filter = acot_vla.ACOTConfig(paligemma_variant="gemma_2b_lora").get_freeze_filter(
            freeze_vision = False, freeze_llm = True, freeze_llm_embedder=True, freeze_dual_ae=[False, False]
        )
    ),
    # genie sim instruction and robust (pi05)
    TrainConfig(
        name="pi05_genie_sim_instruction_and_robust_20260526",
        model=pi0.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_color_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_number_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_shape_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_size_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_common_sense_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_object_type_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_specific_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/straighten_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_follow_logic_(or)_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_billards_color_500",
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_instruction_and_robust_20260526",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        resume=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/zhonglinqing/pkgs/pi05_model/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        save_interval=5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # genie sim instruction and robust (pi0)
    TrainConfig(
        name="pi0_genie_sim_instruction_and_robust_20260526",
        model=pi0.Pi0Config(pi05=False, action_horizon=50),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_color_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_number_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_shape_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_size_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_common_sense_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_object_type_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_specific_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/straighten_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_follow_logic_(or)_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_billards_color_500",
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_instruction_and_robust_20260526",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        resume=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/public_datasets/VLA_weigths/PI0/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        save_interval=5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # genie sim sim2real tasks(pi0.5)
    TrainConfig(
        name="pi05_genie_sim_s2r_20260615",
        model=pi0.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_4871",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_4873",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_4939",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_5125",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_6944",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_7070",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_7332",
                "/mnt/public/linyiren/data/geniesim_data/a2d_probe/task_7453",
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_s2r_20260615",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        resume=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/zhonglinqing/pkgs/pi05_model/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=40_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # genie sim spatial tasks (pi0.5)
    TrainConfig(
        name="pi05_genie_sim_spatial_20260528",
        model=pi0.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/pick_object_relative_position_absolute",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/pick_object_relative_position_relative",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/place_beverage_to_anothers_position",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/place_object_relative_position",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/sort_cubes_by_size",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/sort_number_from_small_to_big",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/stack_bowls",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/stack_three_building_blocks"
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_spatial_20260528",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        resume=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/zhonglinqing/pkgs/pi05_model/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # genie sim spatial tasks (pi0, reuse pi05 spatial 20260528 dataset)
    TrainConfig(
        name="pi0_genie_sim_spatial_20260518",
        model=pi0.Pi0Config(pi05=False, action_horizon=50),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/pick_object_relative_position_absolute",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/pick_object_relative_position_relative",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/place_beverage_to_anothers_position",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/place_object_relative_position",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/sort_cubes_by_size",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/sort_number_from_small_to_big",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/stack_bowls",
                "/mnt/public/linyiren/data/geniesim_data/spatial/v21/stack_three_building_blocks"
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_spatial_20260528",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        resume=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/public_datasets/VLA_weigths/PI0/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=40_000,
        save_interval=5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    TrainConfig(
        name="pi05_demo_stack_blocks",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/complex_task/simple_stack_building_blocks_by_demo",
                "/mnt/public/linyiren/data/geniesim_data/complex_task/stack_symmetric_tower_by_demo"
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_demo_stack_blocks",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=False,
            output_dim=16,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/zhonglinqing/pkgs/pi05_model/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        resume=True,
        save_interval=10000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    TrainConfig(
        name="pi05_genie_sim_manip_20260613",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/clean_the_desktop",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/hold_pot",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/open_door",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/place_block_into_box",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/pour_workpiece",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/scoop_popcorn",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/sorting_packages",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/stock_and_straighten_shelf",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21_lite/take_wrong_item_shelf"
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_manip_20260608_v30",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/zhonglinqing/pkgs/pi05_model/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        resume=True,
        save_interval=5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # G2 omnipicker fine-tune of the manipulation_pi05 checkpoint (local).
    # Differs from pi05_genie_sim_manip_20260613 only in: local repo_id paths,
    # weight_loader pointing at the downloaded manip checkpoint rather than the
    # pi05 base, and single-GPU batch/worker counts.
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_smoketest",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        data=LerobotGo1DataConfig(
            repo_id=["/home/datamentors/g2sim/data/lerobot_v21/folding_towels"],
            norm_stats_path="/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/assets/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            # prompt_from_hl_instruction reads info["instruction_segments"], a
            # Genie Sim field marking sub-task boundaries inside an episode.
            # Real G2 recordings carry one instruction for the whole episode and
            # have no such field, so take the prompt from the LeRobot task
            # instead (PromptFromLeRobotTask, keyed on task_index).
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=1e-5, decay_steps=10_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=4,
        batch_size=16,
        num_train_steps=30,
        resume=False,
        save_interval=1000,
    ),
    # Run A: action-expert-only fine-tune of manipulation_pi05.
    #
    # Identical to pi05_g2_manip_finetune_local except that the VLM is frozen:
    # get_freeze_filter(freeze_vision=True, freeze_llm=True) freezes ".*img.*"
    # (SigLIP) and ".*llm(?!.*_1|.*_2).*" (PaliGemma base), leaving only the
    # action expert ".*llm.*_1.*" trainable.
    #
    # Rationale: 30 episodes of a single task is very little data for a full
    # ~3B-parameter fine-tune. Training only the action expert preserves the
    # VLM's grounding, cuts optimizer state (so checkpoints are far smaller than
    # the 12G a full run writes), and is the conventional recipe at this scale.
    #
    # Starts from manipulation_pi05, NOT from the full fine-tune's output -- the
    # point is to freeze the ORIGINAL VLM, not whatever a full run moved it to.
    # Action-expert-only, WITH waist dims 16-19 unmasked.
    #
    # The checkpoint's own norm_stats have q01 == q99 == 0 for dims 16-19 --
    # Genie Sim never drove those joints. Under quantile normalisation
    #     (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    # that maps a real 0.3 rad waist motion to ~6e5 instead of into [-1, 1],
    # which is why those dims are masked by default.
    #
    # norm_stats_path here points at a spliced copy: dims 16-19 carry real
    # quantiles computed from all 107,785 G2 frames, every other dim is
    # byte-identical to the checkpoint's (verified). Arms and grippers are NOT
    # recomputed -- the model's weights were learned under those statistics and
    # the conventions already match (gripper range -0.785..0 in both).
    #
    # delta_action_mask keeps dim 20 in delta space as before but no longer
    # zeroes 16-19, so the four body joints carrying the real torso motion
    # (w3 is the largest span in this dataset) become trainable.
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_waist_unmasked",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_waist/pi05_g2_waist/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            # Do not zero dims 16-19: they now have real quantiles.
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        save_interval=500 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_fixed_masked: corrected ACTIONS norm_stats.
    #
    # use_delta_joint_actions=True means the actions block describes
    # action[t+k] - state[t] -- an offset from the robot's CURRENT pose across
    # the 30-step chunk, not an absolute joint target. The checkpoint's actions
    # quantiles come from Genie Sim, whose home pose differs from this G2's, so
    # they are centred elsewhere: e.g. dim 0 is [-0.298, +0.197] there but
    # [+0.284, +1.237] here. Unnormalising our model's output with the
    # checkpoint's centres injected a ~0.58 rad bias into every action, which is
    # what produced the violent motion on hardware.
    #
    # Only the ACTIONS block is recomputed. STATE stays byte-identical to the
    # checkpoint's: those are absolute joint positions and they already agree.
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_fixed_masked",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_fixed/pi05_g2_fixed/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_fixed_waist: corrected ACTIONS norm_stats.
    #
    # use_delta_joint_actions=True means the actions block describes
    # action[t+k] - state[t] -- an offset from the robot's CURRENT pose across
    # the 30-step chunk, not an absolute joint target. The checkpoint's actions
    # quantiles come from Genie Sim, whose home pose differs from this G2's, so
    # they are centred elsewhere: e.g. dim 0 is [-0.298, +0.197] there but
    # [+0.284, +1.237] here. Unnormalising our model's output with the
    # checkpoint's centres injected a ~0.58 rad bias into every action, which is
    # what produced the violent motion on hardware.
    #
    # Only the ACTIONS block is recomputed. STATE stays byte-identical to the
    # checkpoint's: those are absolute joint positions and they already agree.
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_fixed_waist",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_fixed/pi05_g2_fixed/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_motor_{masked,waist}: pi05_g2_fixed_{masked,waist} on corrected data.
    #
    # lerobot_v21_mp is reconverted from raw with observation.state taken from
    # /hal/joint_state.motor_position. The earlier datasets used .position, which
    # reads exactly HALF the true angle on arm joints 1-2 (L1, L2, R1, R2): the
    # HAL configures motor_encoder_resolution 20 vs encoder_resolution 19 there.
    # motor_position matches /hal/joint_cmd and the robot's /tf on every joint,
    # and it is what the G2 bridge feeds at inference. With .position the delta
    # actions on those joints were ~+/-0.84 rad biased (not a home-pose effect).
    #
    # Train split only (lerobot_v21_mp_train, 26 episodes, 94,639 frames). Held
    # out for validate_actions.py: lerobot_v21_mp_val = folding_towels ep7,
    # s002 ep8, s003 ep9. s003 ep5 (174-frame aborted recording) is dropped.
    # Mapping back to source episodes: <dataset>/meta/split_source.json.
    #
    # norm_stats: BOTH blocks recomputed from the 26 train episodes, dims 0-20.
    # State is no longer the checkpoint's Genie Sim quantiles: under those, L5
    # sat outside [-1, 1] in 93% of frames (R3/R5 ~44%), which pi05's 256-bin
    # state discretisation cannot represent. Masked dims are zeroed at load.
    TrainConfig(
        name="pi05_g2_motor_masked",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_motor_train/pi05_g2_motor_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    TrainConfig(
        name="pi05_g2_motor_waist",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_motor_train/pi05_g2_motor_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_mixed_waist_control ("MixedWaistControl"): continues pi05_g2_motor_waist
    # (motorwaist_20260925_1012/19999) on the old train split plus two new sessions:
    #   folding_towels_s004_s005 (episodes 3 and 20 dropped: 27 trailing NaN gripper rows)
    #   folding_towels_inference_pose
    # 111 train episodes / 348,579 frames. Held out in lerobot_v21_mp_val: the old 3 plus
    # s004_s005 eps 10, 32 and inference_pose eps 8, 24, 42 (see meta/split_source.json).
    # norm_stats recomputed on this train mix (the new sessions move the torso far more:
    # waist-yaw action range ~+/-0.2 rad vs ~+/-0.004 before). Same prompt in every dataset.
    TrainConfig(
        name="pi05_g2_mixed_waist_control",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s004_s005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_inference_pose",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_mixed_train/pi05_g2_mixed_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ACoT-VLA/checkpoints/pi05_g2_motor_waist/motorwaist_20260925_1012/19999/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_mixed_vision_action ("MixedWaistVision"): continues MixedWaistControl/95000
    # (best val) with the VISION encoder (SigLIP + img/head projector) unfrozen as well as the
    # action expert; the PaliGemma language model stays frozen. Same LR (1e-5) for all trainable
    # params. Data = the 5 MixedWaistControl train sets + two new sessions:
    #   folding_towels_infpose002 (ep 26 dropped: 27 NaN rows; val eps 3, 12, 20, 32)
    #   folding_towels_infpose003 (val ep 7)
    # 156 train episodes / 480,775 frames; val = the previous 8 episodes unchanged + those 5.
    # norm_stats recomputed on this 7-set train mix.
    TrainConfig(
        name="pi05_g2_mixed_vision_action",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_s004_s005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_train/folding_towels_infpose003",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_mixed2_train/pi05_g2_mixed2_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ACoT-VLA/checkpoints/pi05_g2_mixed_waist_control/MixedWaistControl/95000/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw_vision_action ("InfPoseWaistVision"): ONE embodiment only -- inference pose
    # + headset waist. Trained from the BASE manipulation_pi05 (not a G2 fine-tune).
    # Vision (SigLIP + img/head) and action expert trainable; PaliGemma LLM frozen; LR 1e-5.
    # Data (lerobot_v21_mp_ipw_*, fresh 90/10 split, see meta/split_source.json):
    #   folding_towels_inference_pose  train 45 / val 5 (4, 14, 24, 34, 44)
    #   folding_towels_infpose002      eps 0-9 only: train 9 / val 1 (5)
    #   folding_towels_infpose003      train 13 / val 1 (7)
    #   folding_towels_infpose004      train 37 / val 4 (5, 15, 25, 35)
    #   folding_towels_infpose005      train 42 / val 5 (4, 14, 24, 33, 42)
    # 146 train episodes / 404,396 frames; 16 val episodes / 43,723 frames. Stats on this train set.
    TrainConfig(
        name="pi05_g2_ipw_vision_action",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose005",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw_train/pi05_g2_ipw_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=1e-5, decay_steps=100_000, decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw_vision_action_wsd: identical to pi05_g2_ipw_vision_action except the LR
    # schedule is Warmup-Stable-Decay -- 1k warmup, flat 1e-5 (same peak as the cosine config),
    # then 1-sqrt decay to 1e-6 over the final 10k of 100k steps.
    TrainConfig(
        name="pi05_g2_ipw_vision_action_wsd",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose005",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw_train/pi05_g2_ipw_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw_vision_action_wsd_crop: pi05_g2_ipw_vision_action_wsd + square center crop
    # instead of letterbox (head 640x400 -> central 400x400, hands 1280x1056 -> central
    # 1056x1056, then 224x224). Checkpoints from this config must be served with it.
    TrainConfig(
        name="pi05_g2_ipw_vision_action_wsd_crop",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose005",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw_train/pi05_g2_ipw_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            center_crop_images=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw2_vision_action_wsd_crop: pi05_g2_ipw_vision_action_wsd_crop (fork commits 330cd2a + d0b7bef:
    # WSD 1e-5 -> 1e-6 over the last 10k + square center crop) on the ipw dataset v2 = the 5 ipw sessions +
    # folding_towels_infpose_headset_20261001_s01 (train 25 / val 3: eps 4, 14, 24) and _s02 (train 9 / val 1: ep 8)
    # from s3://ardia-ai-datasets-dev/processed/.../g2a0004-infpose-headset/2026-10-01. 180 train / 20 val eps.
    # Norm stats recomputed on this 7-set train mix. Base manipulation_pi05 weights.
    TrainConfig(
        name="pi05_g2_ipw2_vision_action_wsd_crop",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose_headset_20261001_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose_headset_20261001_s02",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw2_train/pi05_g2_ipw2_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            center_crop_images=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw3_vision_action_wsd_crop_ft99k: pi05_g2_ipw2_vision_action_wsd_crop (same model, freeze, LR/WSD, crop,
    # bs, steps) fine-tuned FROM InfPoseWaistVisionWsdCrop/99999 on the ipw3 dataset = ipw2 split unchanged + headset
    # sessions 20261002_s01/_s02, 20261006_s01, 20261007_s01, 20261008_s01 (val every 10th ep from 4; s01 eps 25-27
    # dropped: static/no-gripper takes) + 20261008_s02 (train only). 385 train / 44 val eps. Norm stats recomputed on the ipw3 train set. See g2sim/assets_ipw3_train/README.md.
    TrainConfig(
        name="pi05_g2_ipw3_vision_action_wsd_crop_ft99k",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261001_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261001_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261002_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261002_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261006_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261007_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261008_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw3_train/folding_towels_infpose_headset_20261008_s02",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw3_train/pi05_g2_ipw3_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            center_crop_images=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ACoT-VLA/checkpoints/pi05_g2_ipw2_vision_action_wsd_crop/InfPoseWaistVisionWsdCrop/99999/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw4_vision_action_wsd_crop_ft: pi05_g2_ipw3_vision_action_wsd_crop_ft99k (same model, freeze, LR/WSD, crop,
    # bs, steps) as a NEW run (not resume) FROM InfPoseWaistVisionWsdCropFt99k/99999 on ipw4 = ipw3 split unchanged +
    # folding_towels_infpose_headset_20261012_s01 (s3 .../2026-10-12/s01, 30 eps: train 27 / val 3: eps 4, 14, 24).
    # 412 train / 47 val eps. Norm stats recomputed on ipw4 train. See g2sim/assets_ipw4_train/README.md.
    TrainConfig(
        name="pi05_g2_ipw4_vision_action_wsd_crop_ft",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261001_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261001_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261002_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261002_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261006_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261007_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261008_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261008_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261012_s01",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw4_train/pi05_g2_ipw4_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            center_crop_images=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ACoT-VLA/checkpoints/pi05_g2_ipw3_vision_action_wsd_crop_ft99k/InfPoseWaistVisionWsdCropFt99k/99999/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=100_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # pi05_g2_ipw4_vision_action_wsd_crop_ft3: pi05_g2_ipw4_vision_action_wsd_crop_ft (same ipw4 data + stats, model,
    # freeze, crop, bs, peak/end LR) as a NEW 50k-step WSD cycle (1k warmup, flat 1e-5, sqrt decay to 1e-6 over the
    # last 5k) FROM InfPoseWaistVisionWsdCropFt2/98000 (best of Ft2, val mae 0.03054). No new data.
    TrainConfig(
        name="pi05_g2_ipw4_vision_action_wsd_crop_ft3",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=False, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_inference_pose",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose002",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose003",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose004",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose005",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261001_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261001_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261002_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261002_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261006_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261007_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261008_s01",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261008_s02",
                "/home/datamentors/g2sim/data/lerobot_v21_mp_ipw4_train/folding_towels_infpose_headset_20261012_s01",
            ],
            norm_stats_path="/home/datamentors/g2sim/assets_ipw4_train/pi05_g2_ipw4_train/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            unmask_waist=True,
            center_crop_images=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.WSDSchedule(
            warmup_steps=1_000, peak_lr=1e-5, total_steps=50_000, decay_steps=5_000, end_lr=1e-6, decay_type="sqrt",
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ACoT-VLA/checkpoints/pi05_g2_ipw4_vision_action_wsd_crop_ft/InfPoseWaistVisionWsdCropFt2/98000/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=50_000,
        log_interval=20,
        resume=True,
        save_interval=1000 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_actionexpert_only",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        freeze_filter=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220).get_freeze_filter(
            freeze_vision=True, freeze_llm=True
        ),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s003",
            ],
            norm_stats_path="/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/assets/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        log_interval=20,
        resume=True,
        # 500 rather than 2000: far more checkpoints to pick from, and each one
        # is much smaller here because only the action expert carries optimizer
        # state.
        save_interval=500 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # !!! DEPRECATED (2026-09-24) -- OLD, DO NOT USE FOR NEW WORK !!!
    # Trained on lerobot_v21, whose observation.state came from
    # /hal/joint_state.position: HALF the true angle on arm joints 1-2 (L1, L2,
    # R1, R2). Checkpoints from this config learned that bug and misbehave on
    # the real G2 and in sim. Kept only for reference. Use pi05_g2_motor_* .
    TrainConfig(
        name="pi05_g2_manip_finetune_local",
        model=pi0.Pi0Config(pi05=True, action_horizon=30, max_token_len=220),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s002",
                "/home/datamentors/g2sim/data/lerobot_v21/folding_towels_s003",
            ],
            # Reuse the checkpoint's own norm stats verbatim. norm_stats_path
            # takes the file directly; assets_dir/asset_id would join to
            # assets/<asset_id>/norm_stats.json, which does not exist here --
            # the checkpoint ships it at assets/norm_stats.json.
            norm_stats_path="/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/assets/norm_stats.json",
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            mask_gripper_state=True,
            output_dim=21,
            # prompt_from_hl_instruction reads info["instruction_segments"], a
            # Genie Sim field marking sub-task boundaries inside an episode.
            # Real G2 recordings carry one instruction for the whole episode and
            # have no such field, so take the prompt from the LeRobot task
            # instead (PromptFromLeRobotTask, keyed on task_index).
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/datamentors/g2sim/ckpt/checkpoints/manipulation_pi05/params"
        ),
        num_workers=8 if os.getenv("DEBUG_MODE") != "true" else 2,
        batch_size=16 if os.getenv("DEBUG_MODE") != "true" else 2,
        num_train_steps=20_000,
        # log often so TensorBoard shows curves within minutes of launch
        log_interval=20,
        resume=True,
        save_interval=500 if os.getenv("DEBUG_MODE") != "true" else 100,
    ),
    # acot genie sim instruction and robust
    TrainConfig(
        name="acot_genie_sim_instruction_and_robust_20260525",
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, pi05=True, paligemma_variant="gemma_2b", action_expert_variant = "gemma_300m", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, query_based_implicit_extractor=False, attention_pooling_implicit_extractor=False, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo1DataConfig(
            repo_id = [
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_color_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_number_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_shape_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_block_size_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_common_sense_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_object_type_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_specific_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/straighten_object_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_follow_logic_(or)_500",
                "/mnt/public/linyiren/data/geniesim_data/instruction_and_robust/v21/pick_billards_color_500"
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/acot_genie_sim_instruction_and_robust_20260525",
            ),
            default_prompt=None,
            state_mask = np.array(_transforms.make_bool_mask(-14, 18)).tolist(),
            action_mask = np.array(_transforms.make_bool_mask(-16, 16)).tolist(),
            extra_delta_transform=(True, True),
            joint_action_shifts=(2, 1),
            repack_transforms =_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt"
                        }
                    )
                ]
            ),
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
        ),
        num_train_steps=50_000,
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 16,
        save_interval=10000,
        freeze_filter=acot_vla.ACOTConfig(paligemma_variant="gemma_2b").get_freeze_filter(freeze_vision = False, freeze_llm = False, freeze_llm_embedder=False, freeze_dual_ae=[False, False])
    ),
    # genie sim manip (pi0, reuse pi05 manip 20260526 dataset)
    TrainConfig(
        name="pi0_genie_sim_manip_20260526",
        model=pi0.Pi0Config(pi05=False, action_horizon=30, max_token_len=220),
        data=LerobotGo1DataConfig(
            repo_id=[
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/pour_workpiece",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/open_door",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/scoop_popcorn",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/scoop_popcorn_part_2",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/hold_pot",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/place_block_into_box",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/take_wrong_item_shelf",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/stock_and_straighten_shelf",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/stock_and_straighten_shelf_part_2",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/sorting_packages_part_1",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/sorting_packages_part_2",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/sorting_packages_part_3",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/clean_the_desktop_part_1",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/clean_the_desktop_part_2",
                "/mnt/public/linyiren/data/geniesim_data/manipulation/v21/clean_the_desktop_part_3",
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/mnt/public/jincheng/train/lerobot/pi05_genie_sim_manip_20260526",
            ),
            default_prompt=None,
            use_delta_joint_actions=True,
            include_waist=True,
            output_dim=21,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/public/public_datasets/VLA_weigths/PI0/params"),
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        batch_size=256 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_train_steps=50_000,
        resume=True,
        save_interval=5000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1000,
    ),
    # genie sim 3.0 sim2real task config
    TrainConfig(
        name="s2r_select_color",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/root/openpi/data/select_color/",
            norm_stats_path="/root/openpi/checkpoints/select_color/29999/norm_stats.json",
            default_prompt="",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_select_color",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_size_recognize",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/root/openpi/data/recognize_size/",
            norm_stats_path="/root/openpi/checkpoints/recognize_size/29999/norm_stats.json",
            default_prompt="",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_size_recognize",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_grasp_targets",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/root/openpi/data/grasp_targets/",
            norm_stats_path="/root/openpi/checkpoints/grasp_targets/29999/norm_stats.json",
            default_prompt="",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_grasp_targets",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_organize_items",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/root/openpi/data/organize_items/",
            norm_stats_path="/root/openpi/checkpoints/organize_items/29999/norm_stats.json",
            default_prompt="",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_organize_items",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_pack_in_supermarket",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/E6/lerobot/8782/task_6944/",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_6944/norm_stats.json",
            default_prompt="collect and organize table tennis",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_pack_in_supermarket",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_place_block_into_drawer",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/jincheng/data/drawer/",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_7070/norm_stats.json",
            default_prompt="organize the block into the drawer",
            use_delta_joint_actions=True,
            mask_gripper_state=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_place_block_into_drawer",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_bimanual_chip_handover",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/jincheng/data/handover_chips",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_7332/norm_stats.json",
            default_prompt="Handover chips",
            use_delta_joint_actions=True,
            mask_gripper_state=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_bimanual_chip_handover",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="s2r_sort_fruit",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/jincheng/data/sort_fruit",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_7453/norm_stats.json",
            default_prompt="Sort the fruit",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="s2r_sort_fruit",
        num_train_steps=15_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
