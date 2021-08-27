import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Callable, List

import operator
from functools import partial, reduce

# from einops import rearrange, reduce, repeat

import numpy as np

from .utils import ConvZeros

# Flow Blocks

class ActNorm(nn.Module):
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


class InvertibleConv1x1(nn.Module):
    channels: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        c = self.channels
        # Sample random rotation matrix
        q, _ = jnp.linalg.qr(jax.random.normal(self.key, (c, c)), mode='complete')
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = jnp.triu(u, k=1)
        s = jnp.diag(u)
        self.sign_s = jnp.sign(s)
        S_log_init = jnp.log(jnp.abs(s))
        self.l_mask = jnp.tril(jnp.ones((c, c)), k=-1)
        self.u_mask = jnp.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (c, c))
        self.U = self.param("U", lambda k, sh: U_init, (c, c))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (c,))
        
    def __call__(self, inputs, logdet=0, reverse=False):
        c = self.channels
        assert c == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + jnp.eye(c)
        U = self.U * self.u_mask + jnp.diag(self.sign_s * jnp.exp(self.log_s))
        logdet_factor = inputs.shape[1] * inputs.shape[2]
        
        # forward
        if not reverse:
            # TODO: Switch to einops notation
            # lax.conv uses weird ordering: NCHW and OIHW
            W = jnp.matmul(self.P, jnp.matmul(L, U))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)), 
                             W[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet += jnp.sum(self.log_s) * logdet_factor
        # inverse
        else:
            W_inv = jnp.matmul(jax.scipy.linalg.inv(U), jnp.matmul(
                jax.scipy.linalg.inv(L), self.P_inv))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)),
                             W_inv[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet -= jnp.sum(self.log_s) * logdet_factor
            
        return y, logdet


class AffineCoupling(nn.Module):

    subnet: Callable[[int],nn.Module] 
    out_dims: int

    eps: float = 1e-8
    identity_init: bool = True
    
    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Split
        xa, xb = jnp.split(inputs, 2, axis=-1)

        net = self.subnet(out_dims=self.out_dims)(xb)
        
        mu, logsigma = jnp.split(net, 2, axis=-1)

        # See https://github.com/openai/glow/blob/master/model.py#L376
        # sigma = jnp.exp(logsigma)
        sigma = jax.nn.sigmoid(logsigma + 2.)
        
        # Merge
        if not reverse:
            ya = sigma * xa + mu
            logdet += jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
        else:
            ya = (xa - mu) / (sigma + self.eps)
            logdet -= jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
            
        y = jnp.concatenate((ya, xb), axis=-1)
        return y, logdet


# Blocks

class FlowStep(nn.Module):

    subnet: Callable[[int],nn.Module] 
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
        
    @nn.compact
    def __call__(self, x, logdet=0, reverse=False):
        out_dims = x.shape[-1]
        if not reverse:
            x, logdet = ActNorm()(x, logdet=logdet, reverse=False)
            x, logdet = InvertibleConv1x1(out_dims, self.key)(x, logdet=logdet, reverse=False)
            x, logdet = AffineCoupling(out_dims, self.subnet)(x, logdet=logdet, reverse=False)
        else:
            x, logdet = AffineCoupling(out_dims, self.subnet)(x, logdet=logdet, reverse=True)
            x, logdet = InvertibleConv1x1(out_dims, self.key)(x, logdet=logdet, reverse=True)
            x, logdet = ActNorm()(x, logdet=logdet, reverse=True)
        return x, logdet


class Sequential(nn.Module):

    modules: List[nn.Module]

    @nn.compact
    def __call__(self,x,reverse=False,logdet=0):

        for module in self.modules[::(-1 if reverse else 1)]:
            x, logdet = module(x,logdet=logdet, reverse=reverse)

        return x, logdet