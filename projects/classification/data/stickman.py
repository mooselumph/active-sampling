from functools import partial

import jax
import jax.numpy as jnp

import glob
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import os
import pickle

from utils import plot_image_grid, make_absolute

from omegaconf import OmegaConf, open_dict

from jax.tree_util import tree_flatten, tree_unflatten


def map_fn(image_path, num_bits=5, size=256, num_channels=1, training=True):
    """Read image file, quantize and map to [-0.5, 0.5] range.
    If num_bits = 8, there is no quantization effect."""
    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
    # Resize input image
    image = tf.cast(image, tf.float32)[:,:,:num_channels]
    image = tf.image.resize(image, (size, size))
    image = tf.clip_by_value(image, 0., 255.)
    # Discretize to the given number of bits
    if num_bits < 8:
        image = tf.floor(image / 2 ** (8 - num_bits))
    # Send to [-1, 1]
    num_bins = 2 ** num_bits
    image = image / num_bins - 0.5
    if training:
        image = image + tf.random.uniform(tf.shape(image), 0, 1. / num_bins)
    
    def get_params(filename):
        param_path, _ = os.path.splitext(filename)
        param_path = param_path.decode("utf-8") + '.pickle'
        with open(param_path,'rb') as f:
            params = pickle.load(f)

        z = np.zeros(size**2)
        leaves, _ = tree_flatten(params)
        z[:len(leaves)] = leaves
        return z

    leaves = tf.numpy_function(get_params,[image_path],Tout=tf.double)

    return image, leaves


@jax.jit
def postprocess(x, num_bits):
    """Map [-0.5, 0.5] quantized images to uint space"""
    num_bins = 2 ** num_bits
    x = jnp.floor((x + 0.5) * num_bins)
    x *= 256. / num_bins
    return jnp.clip(x, 0, 255).astype(jnp.uint8)



def get_train_dataset(image_path, image_size, num_channels,  num_bits, batch_size, ext='jpg', skip=None, **kwargs):
    del kwargs
    train_ds = tf.data.Dataset.list_files(f"{image_path}/*.{ext}")
    if skip is not None:
        train_ds = train_ds.skip(skip)
    train_ds = train_ds.shuffle(buffer_size=20000)
    train_ds = train_ds.map(partial(map_fn, size=image_size, num_channels=num_channels, num_bits=num_bits, training=True))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat()
    return iter(tfds.as_numpy(train_ds))


def get_val_dataset(image_path, image_size, num_bits, batch_size, ext='jpg',
                    take=None, repeat=False, **kwargs):
    del kwargs
    val_ds = tf.data.Dataset.list_files(f"{image_path}/*.{ext}")
    if take is not None:
        val_ds = val_ds.take(take)
    val_ds = val_ds.map(partial(map_fn, size=image_size, num_bits=num_bits, training=False))
    val_ds = val_ds.batch(batch_size)
    if repeat:
        val_ds = val_ds.repeat()
    return iter(tfds.as_numpy(val_ds))


def setup_data(config: OmegaConf,show_grid=False):

    with open_dict(config):
        config.data.image_path = make_absolute(config.data.image_path)

    num_images = len(glob.glob(f"{config.data.image_path}/*.{config.data.ext}"))

    with open_dict(config):
        config.train.steps_per_epoch = num_images // config.train.batch_size

    train_split = int(config.data.train_split * num_images)

    print(f"{num_images} training images")
    print(f"{config.train.steps_per_epoch} training steps per epoch")

    #Train data
    train_ds = get_train_dataset(**config.data, batch_size=config.train.batch_size, skip=train_split)

    # Val data
    # During training we'll only evaluate on one batch of validation 
    # to save on computations
    val_ds = get_val_dataset(**config.data, batch_size=config.train.batch_size, take=config.train.batch_size, repeat=True)

    # Sample
    if show_grid:
        plot_image_grid(postprocess(next(val_ds), num_bits=config.data.num_bits)[:25], 
                        title="Input data sample")

    return train_ds, val_ds