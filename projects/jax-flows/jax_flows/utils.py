import jax
import jax.numpy as jnp
import flax.linen as nn


# from einops import rearrange, reduce, repeat

import numpy as np
# 
#  Utils

class ConvZeros(nn.Module):
    features: int
        
    @nn.compact
    def __call__(self, x):
        """A simple convolutional layers initializer to all zeros"""
        x = nn.Conv(self.features, kernel_size=(3, 3),
                    strides=(1, 1), padding='same',
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros)(x)
        return x