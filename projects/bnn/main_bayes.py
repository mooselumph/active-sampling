import logging

import hydra
from omegaconf import OmegaConf

import tensorflow as tf
import jax

from train_bayes import train
from data import setup_data

from models import load_model
from hypermodels import DiagonalGaussian

"""Simple training loop.
    Args:
        train_ds: Training dataset iterator (e.g. tensorflow dataset)
        val_ds: Validation dataset (optional)
        num_samples: Number of samples to generate at each epoch
        image_size: Input image size
        num_channels: Number of channels in input images
        num_bits: Number of bits for discretization
        init_lr: Initial learning rate (Adam)
        num_epochs: Numer of training epochs
        num_sample_epochs: Visualize sample at this interval
        num_warmup_epochs: Linear warmup of the learning rate to init_lr
        num_save_epochs: save mode at this interval
        steps_per_epochs: Number of steps per epochs
        K: Number of flow iterations in the GLOW model
        L: number of scales in the GLOW model
        nn_width: Layer width in the Affine Coupling Layer
        sampling_temperature: Smoothing temperature for sampling from the 
            Gaussian priors (1 = no effect)
        learn_top_prior: Whether to learn the prior for highest latent variable zL.
            Otherwise, assumes standard unit Gaussian prior
        key: Random seed
    """

@hydra.main(config_path='configs',config_name='bnn')
def main(config: OmegaConf):

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())


    key = jax.random.PRNGKey(0)

    train_ds, val_ds = setup_data(config,show_grid=False)

    model, params = load_model(config,train_ds,key)

    hypermodel = DiagonalGaussian(model,params)

    x, _ = next(train_ds)
    params = jax.jit(hypermodel.init)(key, x)


    params = train(config, hypermodel, params, train_ds, val_ds, key)


if __name__ == '__main__': 
    main()
