import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from typing import Tuple

def get_mu_sig(model,params,train_ds,num_batches=10):

    latents = []

    for i in range(num_batches):
        batch = next(train_ds)
        z,*_ = model.apply(params,batch,reverse=False)
        latents.append(z)

    z = jnp.concatenate(latents,axis=0)
    mu = jnp.mean(z,axis=0)
    sig = jnp.std(z,axis=0)
    
    return mu, sig


class Normal(nn.Module):

    shape: Tuple[int]
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
    temperature: float = 0.7
    empirical_vars = False

    @nn.compact
    def __call__(self, z=None, reverse=False, mu=None, sigma=None, key=None, num_samples=1):

        if not self.empirical_vars:
            mu = self.param('mu',jax.nn.initializers.zeros,  self.shape)
            logsigma = self.param('logsigma',jax.nn.initializers.zeros, self.shape )
            sigma = jnp.exp(logsigma)

        if reverse: 

            if self.empirical_vars:
                assert(mu is not None and sigma is not None)

            key = key if key is not None else self.key
            z = mu + jax.random.normal(key, (num_samples,) + self.shape)*sigma*self.temperature
            return z

        else: 
            assert (z is not None)

            if self.empirical_vars:
                @jax.vmap
                def get_logpz(z):
                    return - jnp.sum(jnp.log(jnp.std(z)))
            else:
                @jax.vmap
                def get_logpz(z):
                    return jnp.sum(- logsigma - 0.5 * jnp.log(2 * jnp.pi) 
                                        - 0.5 * (z - mu) ** 2 / jnp.exp(2 * logsigma))

            logpz = get_logpz(z)
            return logpz

            