import logging

import hydra
from omegaconf import OmegaConf

import tensorflow as tf
import jax

from train import train
from data import setup_data

from models import load_model


@hydra.main(config_path='configs',config_name='default')
def main(config: OmegaConf):

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())


    key = jax.random.PRNGKey(0)

    train_ds, val_ds = setup_data(config,show_grid=False)

    model, params = load_model(config,train_ds,key)

    model, params = train(config, model, params, train_ds, val_ds, key)


if __name__ == '__main__': 
    main()
