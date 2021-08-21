
import jax
import jax.numpy as jnp
import flax.linen as nn


import numpy as np

from .utils import ConvZeros

# Channel reshaping

# TODO: Update to use jax.lax.conv. Compare with einops equiv for speed.
class Squeeze(nn.Module):

    channels: int

    def setup(self):

        self.downsample_kernel = jnp.zeros(1, 4, 2, 2)

        self.downsample_kernel[0, 0, 0, 0] = 1
        self.downsample_kernel[0, 1, 0, 1] = 1
        self.downsample_kernel[0, 2, 1, 0] = 1
        self.downsample_kernel[0, 3, 1, 1] = 1

        self.downsample_kernel = jnp.concatenate([self.downsample_kernel] * self.channels, 0)

        self.conv_layer = nn.Conv(features= 4,kernel_size=[2,2], strides=2,feature_group_count=self.channels,use_bias=False)
        self.conv_tpose_layer = nn.ConvTranspose(features= 4,kernel_size=[2,2], strides=2,feature_group_count=self.channels,use_bias=False)

    def forward(self,x):

        params = {'kernel': self.downsample_kernel}
        x = self.conv_layer.apply(params,x)
        return x

    def reverse(self,x):
        params = {'kernel': self.downsample_kernel}
        x = self.conv_layer.apply(params,x)
        x = self.conv_tpose_layer.apply(params,x)


def squeeze(x):
    x = jnp.reshape(x, (x.shape[0], 
                        x.shape[1] // 2, 2, 
                        x.shape[2] // 2, 2,
                        x.shape[-1]))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(x, x.shape[:3] + (4 * x.shape[-1],))
    return x

def unsqueeze(x):
    x = jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 
                        2, 2, x.shape[-1] // 4))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(x, (x.shape[0], 
                        2 * x.shape[1],
                        2 * x.shape[3],
                        x.shape[5]))
    return x


# Multiscale

class Split(nn.Module):
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)


    @nn.compact
    def __call__(self, x, reverse=False, z=None, eps=None, temperature=1.0):
        """Args (reverse = True):
            * z: If given, it is used instead of sampling (= deterministic mode).
                This is only used to test the reversibility of the model.
            * eps: If z is None and eps is given, then eps is assumed to be a 
                sample from N(0, 1) and rescaled by the mean and variance of 
                the prior. This is used during training to observe how sampling
                from fixed latents evolve. 
               
        If both are None, the model samples z from scratch
        """

        if not reverse:
            del z, eps, temperature
            z, x = jnp.split(x, 2, axis=-1)
            
        # Learn the prior parameters for z
        prior = ConvZeros(x.shape[-1] * 2, name="conv_prior")(x)
        
        # Reverse mode: Only return the output
        if reverse:
            # sample from N(0, 1) prior (inference)

            if z is None:
                if eps is None:
                    eps = jax.random.normal(self.key, x.shape) 

                eps *= temperature
                mu, logsigma = jnp.split(prior, 2, axis=-1)
                z = eps * jnp.exp(logsigma) + mu

            return jnp.concatenate([z, x], axis=-1)
        # Forward mode: Also return the prior as it is used to compute the loss
        else:
            return z, x, prior
