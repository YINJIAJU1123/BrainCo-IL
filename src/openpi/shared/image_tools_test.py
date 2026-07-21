import jax.numpy as jnp
import numpy as np

from openpi.shared import image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 4: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)


def test_resize_with_pad_numpy_preserves_batch_and_padding():
    images = np.full((2, 10, 20, 3), 255, dtype=np.uint8)

    resized = image_tools.resize_with_pad_numpy(images, 20, 20)

    assert resized.shape == (2, 20, 20, 3)
    assert resized.dtype == np.uint8
    assert np.all(resized[:, :5] == 0)
    assert np.all(resized[:, 5:15] == 255)
    assert np.all(resized[:, 15:] == 0)
