from collections.abc import Sequence
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.rtc import RTCConfig
from openpi.policies.rtc import build_exp_prefix_weights
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


class Policy:
    """JAX policy with the transforms required by a trained checkpoint."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        rtc_sampler = getattr(model, "sample_actions_rtc", None)
        # Build the second JIT wrapper lazily so standard deployment does not
        # split model state or allocate RTC compilation state.
        self._rtc_sampler_method = rtc_sampler if callable(rtc_sampler) else None
        self._sample_actions_rtc = None
        self._action_horizon = int(model.action_horizon)
        self._action_dim = int(model.action_dim)
        self._rng = rng or jax.random.key(0)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = jax.tree.map(lambda value: value, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda value: np.asarray(value[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    def infer_rtc(
        self,
        obs: dict,
        *,
        previous_actions: np.ndarray,
        inference_delay: int,
        rtc_config: RTCConfig,
        noise: np.ndarray | None = None,
    ) -> dict:
        """Infer with guided RTC while preserving the ordinary ``infer`` path."""

        if self._rtc_sampler_method is None:
            raise RuntimeError("this policy model does not support guided RTC")
        if self._sample_actions_rtc is None:
            self._sample_actions_rtc = nnx_utils.module_jit(self._rtc_sampler_method)
        rtc_config.validate(self._action_horizon)

        previous_actions = np.asarray(previous_actions, dtype=np.float32)
        if previous_actions.ndim != 2 or previous_actions.shape[1] != self._action_dim:
            raise ValueError(
                f"RTC previous_actions must have shape [steps, {self._action_dim}], got {previous_actions.shape}"
            )
        if previous_actions.shape[0] <= 0:
            raise ValueError("RTC previous_actions must contain at least one step")

        # Input transforms mutate action arrays for delta conversion, so own a
        # private copy.  This re-anchors absolute robot-space actions against
        # the current observation and applies the checkpoint's normalization.
        raw_inputs = jax.tree.map(lambda value: value, obs)
        raw_inputs["actions"] = previous_actions[: self._action_horizon].copy()
        inputs = self._input_transform(raw_inputs)

        transformed_previous = np.asarray(inputs.pop("actions"), dtype=np.float32)
        available = min(transformed_previous.shape[0], self._action_horizon)
        padded_previous = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        padded_previous[:available] = transformed_previous[:available]
        prefix_weights = build_exp_prefix_weights(
            action_horizon=self._action_horizon,
            inference_delay=inference_delay,
            prefix_horizon=rtc_config.prefix_horizon,
            available_prefix_steps=available,
        )

        inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        previous_batch = jnp.asarray(padded_previous)[None, ...]
        weights_batch = jnp.asarray(prefix_weights)[None, ...]
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions_rtc(
            sample_rng,
            observation,
            previous_batch,
            weights_batch,
            max_guidance_weight=rtc_config.max_guidance_weight,
            **sample_kwargs,
        )
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda value: np.asarray(value[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "rtc_inference_delay_steps": int(inference_delay),
            "rtc_prefix_steps": int(available),
        }
        return outputs

    @property
    def supports_rtc(self) -> bool:
        return self._rtc_sampler_method is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
