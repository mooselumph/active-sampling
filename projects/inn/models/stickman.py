import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

from typing import List

from functools import partial

# from einops import rearrange, reduce, repeat

from flows.reshaping import Squeeze, Flatten
from flows.bijections import FlowStep, Sequential
from flows.subnets import ConvSubnet, MlpSubnet


class Stickman(nn.Module):

    conv_widths: List[int]  # Width of convolutional subnetworks
    mlp_width: int # Width of mlp subnetworks
    unflatten_shape: tuple

    def setup(self):
        self.modules = []
        for width in self.conv_widths:
            self.modules.append(Squeeze)
            subnet = partial(ConvSubnet,width=width)
            self.modules.append(FlowStep(subnet))

        self.modules.append(Flatten(shape=self.unflatten_shape))
        subnet = partial(MlpSubnet,width=self.mlp_width)
        self.modules.append(subnet)
        
    @nn.compact
    def __call__(self, x, reverse=False, z=None, eps=None, sampling_temperature=1.0):
        """Args:
            * x: Input to the model
            * reverse: Whether to apply the model or its inverse
        """
        
        x = Sequential(self.modules)(x, reverse)