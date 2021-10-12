import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from typing import Callable

import operator
from functools import reduce

# from einops import rearrange, reduce, repeat



class ActNormOrig(nn.Module):
    scale: float = 1.
    eps: float = 1e-8

    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Data dependent initialization. Will use the values of the batch
        # given during model.init

        axes = tuple(i for i in range(len(inputs.shape) - 1))
        def dd_mean_initializer(key, shape):
            """Data-dependant init for mu"""
            nonlocal inputs
            x_mean = jnp.mean(inputs, axis=axes, keepdims=True)
            return - x_mean
        
        def dd_stddev_initializer(key, shape):
            """Data-dependant init for sigma"""
            nonlocal inputs
            x_var = jnp.mean(inputs**2, axis=axes, keepdims=True)
            var = self.scale / (jnp.sqrt(x_var) + self.eps)
            return var
        
        # Forward
        shape = (1,) * len(axes) + (inputs.shape[-1],)
        mu = self.param('actnorm_mean', dd_mean_initializer, shape)
        sigma = self.param('actnorm_sigma', dd_stddev_initializer, shape)
        
        logsigma = jnp.log(jnp.abs(sigma))
        logdet_factor = reduce(
            operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)

        if not reverse:
            y = sigma * (inputs + mu)
            logdet += logdet_factor * jnp.sum(logsigma)
        else:
            y = inputs / (sigma + self.eps) - mu
            logdet -= logdet_factor * jnp.sum(logsigma)
        
        # Logdet and return
        return y, logdet


class ActNorm(nn.Module):
    scale: float = 1.
    eps: float = 1e-8
    clamp: int = 2
    clamp_type: str = 'atan'

    def setup(self):

        if self.clamp_type == 'atan':
            self.clamp_fun = lambda u: self.clamp * (0.636 * jnp.arctan(u))
        elif self.clamp_type == 'glow':
            self.clamp_fun = lambda u: jnp.log(jax.nn.sigmoid(u + self.clamp))
        else:
            raise NotImplementedError


    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Data dependent initialization. Will use the values of the batch
        # given during model.init

        axes = tuple(i for i in range(len(inputs.shape) - 1))
        def dd_mean_initializer(key, shape):
            """Data-dependant init for mu"""
            nonlocal inputs
            x_mean = jnp.mean(inputs, axis=axes, keepdims=True)
            return - x_mean
        
        def dd_stddev_initializer(key, shape):
            """Data-dependant init for sigma"""
            nonlocal inputs
            x_var = jnp.mean(inputs**2, axis=axes, keepdims=True)
            var = self.scale / (jnp.sqrt(x_var) + self.eps)
            return jnp.log(var)
        
        # Forward
        shape = (1,) * len(axes) + (inputs.shape[-1],)
        mu = self.param('actnorm_mean', dd_mean_initializer, shape)

        preclamp = self.param('actnorm_logsigma_preclamp', dd_stddev_initializer, shape)
        logsigma = self.clamp_fun(preclamp)
        
        logdet_factor = reduce(
            operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)

        if not reverse:
            y = jnp.exp(logsigma) * inputs + mu
            logdet += logdet_factor * jnp.sum(logsigma)
        else:
            y = (inputs - mu) * jnp.exp(-logsigma)
            logdet -= logdet_factor * jnp.sum(logsigma)
        
        # Logdet and return
        return y, logdet

class AffineCoupling(nn.Module):

    out_dims: int
    subnet: nn.Module
    incompressible: bool = True

    eps: float = 1e-8
    identity_init: bool = True

    clamp: int = 2
    clamp_type: str = 'atan'

    def setup(self):

        if self.clamp_type == 'atan':
            self.clamp_fun = lambda u: self.clamp * (0.636 * jnp.arctan(u))
        elif self.clamp_type == 'glow':
            self.clamp_fun = lambda u: jnp.log(jax.nn.sigmoid(u + self.clamp))
        else:
            raise NotImplementedError
    
    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Split
        xa, xb = jnp.array_split(inputs, 2, axis=-1)

        net = self.subnet(xb)
        
        mu, preclamp = jnp.array_split(net, 2, axis=-1)
        logsigma = self.clamp_fun(preclamp)

        if self.incompressible:
            logsigma -= jnp.mean(logsigma,keepdims=True)
        
        sum_dims = tuple(range(1,logsigma.ndim))

        # Merge
        if not reverse:
            ya = jnp.exp(logsigma) * xa + mu
            logdet += 0 if self.incompressible else jnp.sum(logsigma, axis=sum_dims) 
        else:
            ya = (xa - mu) * jnp.exp(-logsigma)
            logdet -= 0 if self.incompressible else jnp.sum(logsigma, axis=sum_dims) 
            
        y = jnp.concatenate((ya, xb), axis=-1)
        return y, logdet



class AffineCouplingOrig(nn.Module):

    out_dims: int
    subnet: nn.Module 

    eps: float = 1e-8
    identity_init: bool = True

    clamp: int = 2
    clamp_type: str = 'atan'

    def setup(self):

        if self.clamp_type == 'atan':
            self.clamp_fun = lambda u: self.clamp * (0.636 * jnp.arctan(u))
        elif self.clamp_type == 'glow':
            self.clamp_fun = lambda u: jnp.log(jax.nn.sigmoid(u + self.clamp))
        else:
            raise NotImplementedError
    
    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Split
        xa, xb = jnp.split(inputs, 2, axis=-1)

        net = self.subnet(xb)
        
        mu, logsigma = jnp.split(net, 2, axis=-1)

        # See https://github.com/openai/glow/blob/master/model.py#L376
        # sigma = jnp.exp(logsigma)
        sigma = jax.nn.sigmoid(logsigma + 2.)

        
        sum_dims = tuple(range(1,sigma.ndim))

        # Merge
        if not reverse:
            ya = sigma * xa + mu
            logdet += jnp.sum(jnp.log(sigma), axis=sum_dims)

        else:
            ya = (xa - mu) / (sigma + self.eps)
            logdet -= jnp.sum(jnp.log(logsigma), axis=sum_dims)
            
        y = jnp.concatenate((ya, xb), axis=-1)
        return y, logdet