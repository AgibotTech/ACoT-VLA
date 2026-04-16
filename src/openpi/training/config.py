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

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
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
                        _transforms.ResizeImages(224, 224),
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
                        _transforms.ResizeImages(224, 224),
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
                        _transforms.ResizeImages(224, 224),
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
                        _transforms.ResizeImages(224, 224),
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
            use_quantile_norm=False,
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

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[go1_policy.Go1ACOTInputs(
                action_dim=model_config.action_dim,
                state_mask = self.state_mask,
                action_mask = self.action_mask,
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
                    }
                )
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
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
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
                )
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

        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
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
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)

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
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
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
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    norm_stats_path: str | None = None

    state_keys: Sequence[str] = ("state/joint/position", "state/left_effector/position", "state/right_effector/position")
    action_keys: Sequence[str] = ("action/joint/position", "action/left_effector/position", "action/right_effector/position")

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

    state_mask = np.array(_transforms.make_bool_mask(-16, 16))
    action_mask = np.array(_transforms.make_bool_mask(-16, 16))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None

        state_indices = None
        action_indices = None
        if repo_id is not None:
            try:
                state_indices = tuple(go1_policy.load_field_indices(repo_id, "observation.state", self.state_keys))
                action_indices = tuple(go1_policy.load_field_indices(repo_id, "action", self.action_keys))
            except (FileNotFoundError, KeyError) as e:
                logging.warning("Could not load field indices from info.json, falling back to no remapping: %s", e)

        if self.mask_gripper_state:
            state_mask = np.array(_transforms.make_bool_mask(-14, 18))
        else:
            state_mask = np.array(_transforms.make_bool_mask(-16, 16))

        data_transforms = _transforms.Group(
            inputs=[
                go1_policy.Go1Inputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=state_mask,
                    action_mask=self.action_mask,
                    state_indices=state_indices,
                    action_indices=action_indices,
                )
            ],
            outputs=[go1_policy.Go1Outputs()],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(14, -2, 6)

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
        name="pi05_simi_5125_2200_eps_sim",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/E6/lerobot/5946/2025122901/gripper/task_5125/",
            norm_stats_path="/mnt/public/linyiren/code/openpi/norm_stats_5946/norm_stats.json",
            default_prompt="collect and organize table tennis",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="pi05_simi_5125_2200_eps_sim",
        num_train_steps=15_000,
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
        name="pi05_simi_4871_1500eps_sim",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/E6/lerobot/5638/2025121901/gripper/task_4871/",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_4871/norm_stats.json",
            default_prompt="collect and organize table tennis",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="pi05_simi_4871_1500eps_sim",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_simi_4873_1800eps_sim",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/E6/lerobot/5640/2025121901/gripper/task_4873/",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_4873/norm_stats.json",
            default_prompt="collect and organize table tennis",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="pi05_simi_4873_1800eps_sim",
        num_train_steps=30_000,
        num_workers=24,
        batch_size=32 * 8,
        save_interval=5000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_simi_4939_3000_eps_sim",
        model=pi0.Pi0Config(pi05=True),
        data=LerobotGo1DataConfig(
            repo_id="/mnt/public/E6/lerobot/5718/2025122002/gripper/task_4939/",
            norm_stats_path="/mnt/jincheng/train/lerobot/task_4939/norm_stats.json",
            default_prompt="collect and organize table tennis",
            use_delta_joint_actions=True,
            base_config=DataConfig(dataloader_sampler="subtask", prompt_from_hl_instruction=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_may21_280k_v1/params"),
        exp_name="pi05_simi_4939_3000_eps_sim",
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
