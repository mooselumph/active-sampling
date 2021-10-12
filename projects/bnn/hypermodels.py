

from numpy import log
import jax
import flax
import jax.numpy as jnp
import flax.linen as nn


import jax.nn.initializers as init

from typing import List, Any

from functools import partial

from utils import make_absolute

from omegaconf import OmegaConf

from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.flatten_util import ravel_pytree


class DiagonalGaussian(nn.Module):

    model: nn.Module
    params: Any

    sigma_prior: float = 1

    def setup(self):

        flat, self.unravel = ravel_pytree(self.params)
 
        self.mu = self.param('mean',lambda _: flat)
        self.logsigma = self.param('logsima', lambda _: -5*jnp.ones_like(flat))


    @nn.compact
    def __call__(self,input,key=jax.random.PRNGKey(0),**kwargs):

        key, subkey = jax.random.split(key,2)

        # Sample from posterior
        eps = jax.random.normal(subkey, self.mu.shape)

        w = self.mu + jnp.log(1 + jnp.exp(self.logsigma)) * eps

        logqw =  jnp.log(1/jnp.sqrt(2*jnp.pi)) - jnp.sum(self.logsigma) - jnp.sum(eps**2)/2
        logpw = jnp.log(1/jnp.sqrt(2*jnp.pi)) - len(w)*jnp.log(self.sigma_prior) - jnp.sum(w**2)/2

        params = self.unravel(w)

        output = self.model.apply(params, input, **kwargs)

        return output, logqw, logpw