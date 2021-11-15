import logging

import hydra
from omegaconf import OmegaConf

import tensorflow as tf
import jax

import models
from data import stickman

from train import train_and_evaluate
from dataloader import get_train_test_dataloader


@hydra.main(config_path='configs',config_name='default')
def main(config: OmegaConf):

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train_ds, test_ds = stickman.setup_data(config)
    trainloader, testloader = get_train_test_dataloader(train_ds, test_ds, config)
    model = models.CNN(**config.model)

    train_and_evaluate(model, trainloader, testloader, config, workdir='../')


if __name__ == '__main__': 
    main()
