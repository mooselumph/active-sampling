import jax
import flax
import jax.numpy as jnp

import os
import time

from functools import partial

import tensorflow as tf

import numpy as np

from utils import plot_image_grid
from models import GLOW
from data import postprocess

@jax.vmap
def get_logpz(z, priors):
    logpz = 0
    for zi, priori in zip(z, priors):
        if priori is None:
            mu = jnp.zeros(zi.shape)
            logsigma = jnp.zeros(zi.shape)
        else:
            mu, logsigma = jnp.split(priori, 2, axis=-1)
        logpz += jnp.sum(- logsigma - 0.5 * jnp.log(2 * jnp.pi) 
                         - 0.5 * (zi - mu) ** 2 / jnp.exp(2 * logsigma))
    return logpz


def sample(model, 
           params, 
           eps=None, 
           shape=None, 
           sampling_temperature=1.0, 
           key=jax.random.PRNGKey(0),
           postprocess_fn=None, 
           save_path=None,
           display=True):
    """Sampling only requires a call to the reverse pass of the model"""
    if eps is None:
        zL = jax.random.normal(key, shape) 
    else: 
        zL = eps[-1]
    y, *_ = model.apply(params, zL, eps=eps, sampling_temperature=sampling_temperature, reverse=True)
    if postprocess_fn is not None:
        y = postprocess_fn(y)
    plot_image_grid(y, save_path=save_path, display=display,
                    title=None if save_path is None else save_path.rsplit('.', 1)[0].rsplit('/', 1)[1])
    return y




random_key = jax.random.PRNGKey(0)

def train_glow(
        config,
        model,
        params,
        train_ds,
        val_ds=None,
        key=jax.random.PRNGKey(0)
        ):

    # Init model

    opt = flax.optim.Adam(learning_rate=config.optim.init_lr).create(params)
    
    def lr_warmup(step):
        return config.optim.init_lr * jnp.minimum(1., step / (config.train.num_warmup_epochs * config.train.steps_per_epoch + 1e-8))
    
    # Helper functions for training
    bits_per_dims_norm = np.log(2.) * config.data.num_channels * config.data.image_size**2
    @jax.jit
    def get_logpx(z, logdets, priors):
        logpz = get_logpz(z, priors)
        logpz = jnp.mean(logpz) / bits_per_dims_norm        # bits per dimension normalization
        logdets = jnp.mean(logdets) / bits_per_dims_norm
        logpx = logpz + logdets - config.data.num_bits                  # num_bits: dequantization factor
        return logpx, logpz, logdets
        
    @jax.jit
    def train_step(opt, batch):
        def loss_fn(params):
            _, z, logdets, priors = model.apply(params, batch, reverse=False)
            logpx, logpz, logdets = get_logpx(z, logdets, priors)
            return - logpx, (logpz, logdets)
        logs, grad = jax.value_and_grad(loss_fn, has_aux=True)(opt.target)
        opt = opt.apply_gradient(grad, learning_rate=lr_warmup(opt.state.step))
        return logs, opt
    
    # Helper functions for evaluation 
    @jax.jit
    def eval_step(params, batch):
        _, z, logdets, priors = model.apply(params, batch, reverse=False)
        return - get_logpx(z, logdets, priors)[0]
    
    # Helper function for sampling from random latent fixed during training for comparison
    eps = []
    if not os.path.exists("samples"): os.makedirs("samples")
    if not os.path.exists("weights"): os.makedirs("weights")
    for i in range(config.model.params.L):
        expected_h = config.data.image_size // 2**(i + 1)
        expected_c = config.data.num_channels * 2**(i + 1)
        if i == config.model.params.L - 1: expected_c *= 2
        eps.append(jax.random.normal(key, (config.train.num_samples, expected_h, expected_h, expected_c)))
    sample_fn = partial(sample, eps=eps, key=key, display=False,
                        sampling_temperature=config.model.sampling_temperature,
                        postprocess_fn=partial(postprocess, num_bits=config.data.num_bits))
    
    # Train
    print("Start training...")
    print("Available jax devices:", jax.devices())
    print()
    bits = 0.
    start = time.time()
    try:
        for epoch in range(config.train.num_epochs):
            # train
            for i in range(config.train.steps_per_epoch):
                batch = next(train_ds)
                loss, opt = train_step(opt, batch)
                print(f"\r\033[92m[Epoch {epoch + 1}/{config.train.num_epochs}]\033[0m"
                      f"\033[93m[Batch {i + 1}/{config.train.steps_per_epoch}]\033[0m"
                      f" loss = {loss[0]:.5f},"
                      f" (log(p(z)) = {loss[1][0]:.5f},"
                      f" logdet = {loss[1][1]:.5f})", end='')
                if np.isnan(loss[0]):
                    print("\nModel diverged - NaN loss")
                    return None, None
                
                step = epoch * config.train.steps_per_epoch + i + 1
                if step % int(config.train.num_sample_epochs * config.train.steps_per_epoch) == 0:
                    sample_fn(model, opt.target, 
                              save_path=f"samples/step_{step:05d}.png")

            # eval on one batch of validation samples 
            # + generate random sample
            t = time.time() - start
            if val_ds is not None:
                bits = eval_step(opt.target, next(val_ds))
            print(f"\r\033[92m[Epoch {epoch + 1}/{config.train.num_epochs}]\033[0m"
                  f"[{int(t // 3600):02d}h {int((t % 3600) // 60):02d}mn]"
                  f" train_bits/dims = {loss[0]:.3f},"
                  f" val_bits/dims = {bits:.3f}" + " " * 50)
            
            # Save parameters
            if (epoch + 1) % config.train.num_save_epochs == 0 or epoch == config.train.num_epochs - 1:
                with open(f'weights/model_epoch={epoch + 1:03d}.weights', 'wb') as f:
                    f.write(flax.serialization.to_bytes(opt.target))
    except KeyboardInterrupt:
        print(f"\nInterrupted by user at epoch {epoch + 1}")
        
    # returns final model and parameters
    return model, opt.target