import flax
import flax.linen as nn
import jax.flatten_util
import optax
from data import *
from models import MLP, HyperModel, FullHyperModel
from flax.core import freeze, unfreeze
from train import *
import train_hypermodel
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import sys


@hydra.main(config_path='configs', config_name='default')
def main(config):
    run_hypermodel(config)


def run_hypermodel(config):
    key = jax.random.PRNGKey(config.PRNGSeed)

    if config.data.generate:
        key, new_key = jax.random.split(key)
        x_train, y_train, x_draw, y_draw = gaussian_process(new_key,
                                                            config.data.num_draw_points, config.data.num_train_points,
                                                            config.data.xlims, sftf(config.data.length_scale))
        if config.data.save_file:
            save_data('datasets', config.data.save_file, x_train, y_train, x_draw, y_draw)

    else:
        x_train, y_train, x_draw, y_draw = get_function_data(config.data.data_dir, config.data.file)

    # Generate fourier features
    encoding_fun = jax.vmap(fourier_positional_encoding, (0, None, None, None), 0)
    fourier = config.data.fourier
    x_train_encoded = encoding_fun(x_train, fourier.max_freq, fourier.num_bands, fourier.base)

    MLP.features = config.model.features
    base_model = MLP()
    hypermodel = FullHyperModel()
    key, new_key = jax.random.split(key)
    base_params = base_model.init(new_key, x_train_encoded)
    flattened_base_params, MLP_unflattener = jax.flatten_util.ravel_pytree(base_params)
    trainloader, testloader = create_train_test_loaders(x_train_encoded, y_train,
                                                        train_split=config.train.train_split,
                                                        batch_size=config.train.batch_size)

    key, new_key = jax.random.split(key)
    final_hypermodel_state = train_hypermodel.train_and_evaluate(new_key, config, hypermodel,
                                                                 trainloader, testloader, MLP_unflattener)

    final_base_params = hypermodel.apply(final_hypermodel_state.params, x_train_encoded[:64]).flatten()
    final_base_params = MLP_unflattener(final_base_params)

    plt.plot(x_draw, y_draw, label='True function')
    x_in = encoding_fun(x_draw, fourier.max_freq, fourier.num_bands, fourier.base)
    y_preds1 = MLP().apply(final_base_params, x_in).flatten()
    plt.plot(x_draw, y_preds1, label='MLP 1', color='orange')
    plt.scatter(x_train[:64], y_train[:64], color='orange')
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig('figure.png')

def run(config):
    # construct function based on RBF Kernel
    key = jax.random.PRNGKey(config.PRNGSeed)

    if config.data.generate:
        key, new_key = jax.random.split(key)
        x_train, y_train, x_draw, y_draw = gaussian_process(new_key,
                                                            config.data.num_draw_points, config.data.num_train_points,
                                                            config.data.xlims, sftf(config.data.length_scale))
        if config.data.save_file:
            save_data('datasets', config.data.save_file, x_train, y_train, x_draw, y_draw)

    else:
        x_train, y_train, x_draw, y_draw = get_function_data(config.data.data_dir, config.data.file)

    # Generate fourier features
    encoding_fun = jax.vmap(fourier_positional_encoding, (0, None, None, None), 0)
    fourier = config.data.fourier
    x_train_encoded = encoding_fun(x_train, fourier.max_freq, fourier.num_bands, fourier.base)

    # Train MLP
    num_train_points = int(len(x_train) / 2)
    trainloader, testloader = create_train_test_loaders(x_train_encoded[:num_train_points], y_train[:num_train_points],
                                                        train_split=config.train.train_split,
                                                        batch_size=config.train.batch_size)
    MLP.features = config.model.features
    key, new_key = jax.random.split(key)
    final_state1 = train_and_evaluate(new_key, config, MLP(), trainloader, testloader)
    trainloader, testloader = create_train_test_loaders(x_train_encoded[num_train_points:], y_train[num_train_points:],
                                                        train_split=config.train.train_split,
                                                        batch_size=config.train.batch_size)
    key, new_key = jax.random.split(key)
    final_state2 = train_and_evaluate(new_key, config, MLP(), trainloader, testloader)

    # Draw the "true" function
    plt.plot(x_draw, y_draw, label='True function')
    x_in = encoding_fun(x_draw, fourier.max_freq, fourier.num_bands, fourier.base)
    y_preds1 = MLP().apply(final_state1.params, x_in).flatten()
    y_preds2 = MLP().apply(final_state2.params, x_in).flatten()
    plt.plot(x_draw, y_preds1, label='MLP 1', color='orange')
    plt.plot(x_draw, y_preds2, label='MLP 2', color='green')
    plt.scatter(x_train[:num_train_points], y_train[:num_train_points], color='orange')
    plt.scatter(x_train[num_train_points:], y_train[num_train_points:], color='green')
    plt.legend()
    plt.savefig('figure.png')


def save_data(save_dir, save_filename, x_train, y_train, x_draw, y_draw):
    save_dir = get_original_cwd() + '/{}'.format(save_dir)
    file_path = save_dir + '/{}'.format(save_filename)

    with open(file_path, 'wb') as f:
        jnp.save(f, x_train)
        jnp.save(f, y_train)
        jnp.save(f, x_draw)
        jnp.save(f, y_draw)


def get_function_data(save_dir, save_filename):
    save_dir = get_original_cwd() + '/{}'.format(save_dir)
    file_path = save_dir + '/{}'.format(save_filename)

    with open(file_path, 'rb') as f:
        x_train = jnp.load(f)
        y_train = jnp.load(f)
        x_draw = jnp.load(f)
        y_draw = jnp.load(f)

    return x_train, y_train, x_draw, y_draw


if __name__ == '__main__':
    main()
