"""Source diagnostics must not change the objective or optimizer update."""
import dataclasses
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as model_lib
from openpi.training import config, utils
from scripts.train import train_step


class TinyModel(model_lib.BaseModel):
    def __init__(self):
        self.action_dim, self.action_horizon, self.max_token_len = 56, 16, 1
        self.kernel = nnx.Param(jnp.zeros((1, 56)))

    def compute_loss(self, rng, observation, actions, *, train=False):
        return jnp.mean((actions - self.kernel.value) ** 2, axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.broadcast_to(self.kernel.value, (4, 16, 56))


class SourceMetricsTest(unittest.TestCase):
    def test_source_means_and_unchanged_update_without_source_metadata(self):
        plain = dataclasses.replace(config._CONFIGS_DICT['debug'], freeze_strategy='none', freeze_filter=nnx.Nothing())
        mixed_data = config.DataConfig(multi_dataset_mode='coverage', lerobot_datasets=(
            config.LeRobotDataset('ego', .5, source_id='ego', mask_wrist_images=True),
            config.LeRobotDataset('real', .5, source_id='real', mask_wrist_images=False),
        ))
        mixed = dataclasses.replace(plain, data=dataclasses.replace(plain.data, base_config=mixed_data))
        model = TinyModel()
        graph, params = nnx.split(model)
        tx = optax.sgd(.1)
        state = utils.TrainState(step=jnp.array(0), params=params, model_def=graph,
                                opt_state=tx.init(params), tx=tx, ema_decay=None, ema_params=None)
        obs = model_lib.Observation(images={'left_wrist_0_rgb':jnp.zeros((4,8,8,3))},
                                   image_masks={'left_wrist_0_rgb':jnp.array([False,False,True,True])},
                                   state=jnp.zeros((4,56)))
        actions = jnp.broadcast_to(jnp.arange(4,dtype=jnp.float32)[:,None,None], (4,16,56))
        mixed_state, metrics = train_step(mixed,jax.random.key(0),state,(obs,actions))
        plain_state, plain_metrics = train_step(plain,jax.random.key(0),state,(obs,actions))
        self.assertAlmostEqual(float(metrics['loss']),3.5)
        self.assertAlmostEqual(float(metrics['loss_ego']),.5)
        self.assertAlmostEqual(float(metrics['loss_real']),6.5)
        self.assertNotIn('loss_ego',plain_metrics)
        for a,b in zip(jax.tree.leaves(mixed_state.params),jax.tree.leaves(plain_state.params),strict=True):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(mixed_state.params['kernel'].value, .3/56, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
