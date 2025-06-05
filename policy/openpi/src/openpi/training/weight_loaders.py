import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")

@dataclasses.dataclass(frozen=True)
class ActionDimAdaptiveWeightLoader(WeightLoader):
    """Loads weights from a checkpoint and adapts action dimension mismatches.
    
    This loader handles cases where the pretrained model has a different action_dim
    than the current model by appropriately resizing the action projection layers.
    """

    params_path: str
    # Strategy for handling dimension mismatch: 'truncate', 'pad_zero', 'pad_random'
    resize_strategy: str = "truncate"

    def load(self, params: at.Params) -> at.Params:
        # Load the original parameters
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        # Adapt action dimensions if needed
        adapted_params = self._adapt_action_dimensions(loaded_params, params)
        
        # Add all missing LoRA weights and other missing parameters
        return _merge_params(adapted_params, params, missing_regex=".*lora.*")
    
    def _adapt_action_dimensions(self, loaded_params: at.Params, target_params: at.Params) -> at.Params:
        """Adapt action dimension mismatches in projection layers."""
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_target = flax.traverse_util.flatten_dict(target_params, sep="/")
        
        # Keys that need action dimension adaptation
        action_dim_keys = [
            "action_in_proj/kernel",
            "action_out_proj/kernel", 
            "action_in_proj/bias",
            "action_out_proj/bias",
            "state_proj/kernel",
            "state_proj/bias"
        ]
        
        adapted_flat = flat_loaded.copy()
        
        for key in action_dim_keys:
            if key in flat_loaded and key in flat_target:
                loaded_weight = flat_loaded[key]
                target_shape = flat_target[key].shape
                
                if loaded_weight.shape != target_shape:
                    logger.info(f"Adapting {key} from shape {loaded_weight.shape} to {target_shape}")
                    adapted_weight = self._resize_weight(loaded_weight, target_shape, key)
                    adapted_flat[key] = adapted_weight
        
        return flax.traverse_util.unflatten_dict(adapted_flat, sep="/")
    
    def _resize_weight(self, weight: np.ndarray, target_shape: tuple, key: str) -> np.ndarray:
        """Resize a weight tensor to match target shape."""
        if "kernel" in key:
            if "action_in_proj" in key:
                # action_in_proj: (action_dim, hidden_dim)
                # Resize the first dimension (action_dim)
                if self.resize_strategy == "truncate":
                    return weight[:target_shape[0], :]
                elif self.resize_strategy == "pad_zero":
                    padded = np.zeros(target_shape, dtype=weight.dtype)
                    min_dim = min(weight.shape[0], target_shape[0])
                    padded[:min_dim, :] = weight[:min_dim, :]
                    return padded
                elif self.resize_strategy == "pad_random":
                    if weight.shape[0] > target_shape[0]:
                        return weight[:target_shape[0], :]
                    else:
                        padded = np.random.normal(0, 0.02, target_shape).astype(weight.dtype)
                        padded[:weight.shape[0], :] = weight
                        return padded
            elif "action_out_proj" in key:
                # action_out_proj: (hidden_dim, action_dim)
                # Resize the second dimension (action_dim)
                if self.resize_strategy == "truncate":
                    return weight[:, :target_shape[1]]
                elif self.resize_strategy == "pad_zero":
                    padded = np.zeros(target_shape, dtype=weight.dtype)
                    min_dim = min(weight.shape[1], target_shape[1])
                    padded[:, :min_dim] = weight[:, :min_dim]
                    return padded
                elif self.resize_strategy == "pad_random":
                    if weight.shape[1] > target_shape[1]:
                        return weight[:, :target_shape[1]]
                    else:
                        padded = np.random.normal(0, 0.02, target_shape).astype(weight.dtype)
                        padded[:, :weight.shape[1]] = weight
                        return padded
            elif "state_proj" in key:
                # state_proj: (action_dim, hidden_dim) - similar to action_in_proj
                if self.resize_strategy == "truncate":
                    return weight[:target_shape[0], :]
                elif self.resize_strategy == "pad_zero":
                    padded = np.zeros(target_shape, dtype=weight.dtype)
                    min_dim = min(weight.shape[0], target_shape[0])
                    padded[:min_dim, :] = weight[:min_dim, :]
                    return padded
                elif self.resize_strategy == "pad_random":
                    if weight.shape[0] > target_shape[0]:
                        return weight[:target_shape[0], :]
                    else:
                        padded = np.random.normal(0, 0.02, target_shape).astype(weight.dtype)
                        padded[:weight.shape[0], :] = weight
                        return padded
        elif "bias" in key:
            if "action_out_proj" in key:
                # action_out_proj bias: (action_dim,)
                if self.resize_strategy == "truncate":
                    return weight[:target_shape[0]]
                elif self.resize_strategy == "pad_zero":
                    padded = np.zeros(target_shape, dtype=weight.dtype)
                    min_dim = min(weight.shape[0], target_shape[0])
                    padded[:min_dim] = weight[:min_dim]
                    return padded
                elif self.resize_strategy == "pad_random":
                    if weight.shape[0] > target_shape[0]:
                        return weight[:target_shape[0]]
                    else:
                        padded = np.zeros(target_shape, dtype=weight.dtype)
                        padded[:weight.shape[0]] = weight
                        return padded
            else:
                # Other biases usually don't need action_dim adaptation
                return weight
        
        return weight

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
