import jax
import flax
import jax.numpy as jnp

import os
import time

from functools import partial

import numpy as np

from utils import plot_image_grid
from data import postprocess

def train(
    config,

    model,
    params,
    train_ds,
    val_ds=None,
    key=jax.random.PRNGKey(0)
    ):

    if not os.path.exists("samples"): os.makedirs("samples")
    if not os.path.exists("weights"): os.makedirs("weights")

    opt = flax.optim.Adam(learning_rate=config.optim.init_lr).create(params)
    
    def lr_warmup(step):
        return config.optim.init_lr * jnp.minimum(1., step / (config.train.num_warmup_epochs * config.train.steps_per_epoch + 1e-8))

        
    @jax.jit
    def train_step(opt, x ,y, key):
        def loss_fn(params):
            (xpred, _), logqw, logpw = model.apply(params, y, reverse=True, key=key)

            sigma_output = config.train.sigma_output

            logpdw = jnp.log(1/jnp.sqrt(2*jnp.pi)) - jnp.prod(jnp.array(x.shape[1:]))*jnp.log(sigma_output) - jnp.sum((x - xpred)**2)/(2*sigma_output**2)

            return logqw - logpw - logpdw, (logqw, logpw, logpdw)

        (loss,aux), grad = jax.value_and_grad(loss_fn,has_aux=True)(opt.target)
        opt = opt.apply_gradient(grad, learning_rate=lr_warmup(opt.state.step))
        return loss, opt, aux
    
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

                key, subkey = jax.random.split(key,2)

                x, y = next(train_ds)
                x, y = jnp.array(x), jnp.array(y)
                loss, opt, (logqw,logpw,logpdw) = train_step(opt, x, y, subkey)
                
                print(f"\r\033[92m[Epoch {epoch + 1}/{config.train.num_epochs}]\033[0m"
                      f"\033[93m[Batch {i + 1}/{config.train.steps_per_epoch}]\033[0m"
                      f" loss = {loss:.5f},"
                      f" logqw = {logqw:.5f},"
                      f" logpq = {logpw:.5f},"
                      f" logpdw = {logpdw:.5f},"
                      , 
                      end='')
                
                if np.isnan(loss):
                    print("\nModel diverged - NaN loss")
                    return None, None
                
                step = epoch * config.train.steps_per_epoch + i + 1

            # eval on one batch of validation samples 
            # + generate random sample
            t = time.time() - start
            
            print(f"\r\033[92m[Epoch {epoch + 1}/{config.train.num_epochs}]\033[0m"
                  f"[{int(t // 3600):02d}h {int((t % 3600) // 60):02d}mn]"
                  f" train_loss = {loss:.3f},"
                  )
            
            # Save parameters
            if (epoch + 1) % config.train.num_save_epochs == 0 or epoch == config.train.num_epochs - 1:
                with open(f'weights/model_epoch={epoch + 1:03d}.weights', 'wb') as f:
                    f.write(flax.serialization.to_bytes(opt.target))

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at epoch {epoch + 1}")
        
    # returns final model and parameters
    return opt.target