import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import diffusion
from openpi.models.diffusion_config import DiffusionPolicyConfig
from openpi.shared import nnx_utils


def tiny_config(**kwargs):
    return dataclasses.replace(
        DiffusionPolicyConfig(
            action_dim=2,
            action_horizon=4,
            down_dims=(8, 16),
            kernel_size=3,
            diffusion_step_embed_dim=8,
            image_size=32,
            vision_width=8,
            vision_feature_dim=8,
            num_train_timesteps=8,
            num_inference_steps=2,
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_horizon": 3},
        {"down_dims": (7, 16)},
        {"n_obs_steps": 2},
        {"num_inference_steps": 101},
        {"sampler": "invalid"},
        {"kernel_size": 2},
        {"diffusion_step_embed_dim": 2},
        {"dtype": "bfloat16"},
        {"image_size": 16},
    ],
)
def test_reject_invalid_config(overrides):
    with pytest.raises(ValueError, match=r"must|supports|require|unsupported"):
        DiffusionPolicyConfig(**overrides)


@pytest.mark.parametrize("schedule", ["squaredcos_cap_v2", "linear"])
@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_scheduler_matches_diffusers(schedule, sampler, monkeypatch):
    # Official Diffusion Policy uses these schedulers. No diffusers dependency
    # is needed at runtime; this is an independent cross-framework oracle.
    diffusers = pytest.importorskip("diffusers")
    import torch  # noqa: PLC0415

    config = tiny_config(beta_schedule=schedule, num_train_timesteps=10, num_inference_steps=3)
    cls = diffusers.DDPMScheduler if sampler == "ddpm" else diffusers.DDIMScheduler
    scheduler = cls(num_train_timesteps=10, beta_schedule=schedule, prediction_type="epsilon", clip_sample=True)
    np.testing.assert_allclose(diffusion.diffusion_betas(config), scheduler.betas.numpy(), rtol=2e-6)
    scheduler.set_timesteps(3)
    np.testing.assert_array_equal(scheduler.timesteps.numpy(), np.array([6, 3, 0]))
    sample = np.random.default_rng(1).normal(size=(2, 4, 2)).astype(np.float32)
    epsilon = np.random.default_rng(2).normal(size=sample.shape).astype(np.float32)
    noise = np.random.default_rng(3).normal(size=sample.shape).astype(np.float32)
    if sampler == "ddpm":
        monkeypatch.setattr(
            "diffusers.schedulers.scheduling_ddpm.randn_tensor", lambda *args, **kwargs: torch.from_numpy(noise)
        )
    alphas = np.cumprod(1 - diffusion.diffusion_betas(config))
    for t in scheduler.timesteps.tolist():
        previous_alpha = alphas[t - 3] if t >= 3 else 1.0
        actual = diffusion.denoise_step(
            jnp.asarray(sample),
            jnp.asarray(epsilon),
            alphas[t],
            previous_alpha,
            jnp.asarray(noise),
            sampler=sampler,
            clip_sample=True,
        )
        expected = scheduler.step(torch.from_numpy(epsilon), t, torch.from_numpy(sample)).prev_sample.numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_jitted_model_loss_and_sampling(sampler):
    config = tiny_config(sampler=sampler)
    key = jax.random.key(0)
    model = config.create(key)
    obs = config.fake_obs(2)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, config.fake_act(2))
    assert loss.shape == (2, 4)
    assert np.all(np.isfinite(loss))
    sample = nnx_utils.module_jit(model.sample_actions)
    first = sample(key, obs)
    repeated = sample(key, obs)
    different = sample(jax.random.key(2), obs)
    assert first.shape == (2, 4, 2)
    assert first.dtype == jnp.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, different)
    with pytest.raises(ValueError, match="noise must have shape"):
        model.sample_actions(key, obs, noise=jnp.zeros((1, 2, 3)))


def test_training_gradients_reach_vision_and_unet():
    config = tiny_config()
    model = config.create(jax.random.key(0))

    @nnx.jit
    def gradient(model, obs, actions):
        return nnx.value_and_grad(lambda m: jnp.mean(m.compute_loss(jax.random.key(3), obs, actions, train=True)))(
            model
        )

    obs = config.fake_obs()
    # Non-constant images avoid normalization cancellation in small ResNets.
    obs = dataclasses.replace(
        obs,
        images={
            key: jax.random.uniform(jax.random.key(i), value.shape) for i, (key, value) in enumerate(obs.images.items())
        },
    )
    loss, grads = gradient(model, obs, config.fake_act())
    assert np.isfinite(loss)
    for subtree in (grads["encoders"], grads["unet"]):
        arrays = jax.tree.leaves(subtree)
        assert all(np.all(np.isfinite(value)) for value in arrays)
        assert sum(float(jnp.sum(jnp.abs(value))) for value in arrays) > 0
    for encoder in grads["encoders"].values():
        assert float(jnp.sum(jnp.abs(encoder["conv1"]["kernel"].value))) > 0
    jax.clear_caches()
