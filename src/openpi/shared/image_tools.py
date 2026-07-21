import functools

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import openpi.shared.array_typing as at


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
@at.typecheck
def resize_with_pad(
    images: at.UInt8[at.Array, "*b h w c"] | at.Float[at.Array, "*b h w c"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.UInt8[at.Array, "*b {height} {width} c"] | at.Float[at.Array, "*b {height} {width} c"]:
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), method=method
    )
    if images.dtype == jnp.uint8:
        # round from float back to uint8
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0 if images.dtype == jnp.uint8 else -1.0,
    )

    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


def resize_with_pad_numpy(
    images: np.ndarray,
    height: int,
    width: int,
    method: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Resize uint8 HWC images without initializing JAX in data-loader workers."""
    if images.shape[-3:-1] == (height, width):
        return images
    if images.dtype != np.uint8 or images.shape[-1] != 3:
        raise ValueError(f"Expected RGB uint8 HWC images, got shape={images.shape} dtype={images.dtype}")

    batch_shape = images.shape[:-3]
    flat_images = images.reshape((-1, *images.shape[-3:]))
    resized = np.stack([_resize_with_pad_pil(image, height, width, method) for image in flat_images])
    return resized.reshape((*batch_shape, height, width, 3))


def _resize_with_pad_pil(
    image: np.ndarray,
    height: int,
    width: int,
    method: Image.Resampling,
) -> np.ndarray:
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)

    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=method)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(canvas)
