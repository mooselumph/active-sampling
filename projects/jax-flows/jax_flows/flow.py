import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from typing import Tuple

class Flow(nn.Module):

    prior: nn.Module
    bijection: nn.Module

    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    @nn.compact
    def __call__(self, x=None, z=None, reverse=False,**kwargs):
        

        if reverse: 

            if z is None:
                z = self.prior(reverse=True,**kwargs)
            
            x = self.bijection(z,reverse=True)

            return x

        else: 
            assert(x is not None)

            z, logdets = self.bijection(x)
            logpz = self.prior(z,**kwargs)

            logpz = jnp.mean(logpz)
            logdets = jnp.mean(logdets) 
            logpx = logpz + logdets 

            return z, logdets, logpz, logpx