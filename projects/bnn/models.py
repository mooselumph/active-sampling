import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

from typing import List

from functools import partial

from utils import make_absolute

from omegaconf import OmegaConf

# from einops import rearrange, reduce, repeat

from jax_flows.reshaping import Squeeze, Flatten
from jax_flows.blocks import FlowStep, Sequential
from jax_flows.subnets import ConvSubnet, MlpSubnet
from jax_flows.permutations import FixedPermutation

from jax_flows.priors import Normal
from jax_flows.flow import Flow


def load_model(config: OmegaConf,train_ds,key):

    n = len(config.bijection.params.conv_widths)
    output_hw = config.data.image_size // 2**n
    output_c = config.data.num_channels * 4**n
    model = vanilla(unflatten_shape = (output_hw,output_hw,output_c),**config.bijection.params)

    @jax.jit
    def init(d):
        return model.init(key, d)

    x, _ = next(train_ds)
    params = init(x)

    if config.model.checkpoint_path:

        path = make_absolute(config.model.checkpoint_path)
        with open(path, 'rb') as f:
            params = flax.serialization.from_bytes(params, f.read())

    return model, params


class vanilla(nn.Module):

    conv_widths: List[int]  # Width of convolutional subnetworks
    num_conv_layers_per_scale: int
    mlp_width: int # Width of mlp subnetworks
    num_mlp_layers: int
    unflatten_shape: tuple

    def setup(self):
        modules = []
        for width in self.conv_widths:
            modules.append(Squeeze())
            for _ in range(self.num_conv_layers_per_scale):
                subnet = partial(ConvSubnet,width=width)
                modules.append(FlowStep(subnet))

        modules.append(Flatten(shape=self.unflatten_shape))

        for _ in range(self.num_mlp_layers):
            subnet = partial(MlpSubnet,width=self.mlp_width)
            modules.append(FlowStep(subnet,permutation=FixedPermutation))

        self.modules = modules
        
    @nn.compact
    def __call__(self, x, reverse=False, num_modules=None):
        """Args:
            * x: Input to the model
            * reverse: Whether to apply the model or its inverse
        """

        x, logdets = Sequential(self.modules)(x, reverse=reverse, num_modules=num_modules)
        return x, logdets

            

