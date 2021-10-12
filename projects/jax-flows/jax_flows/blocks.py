import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from typing import Callable, List, Optional

from .affine import ActNorm, AffineCoupling
from .permutations import InvertibleConv1x1

class FlowStep(nn.Module):

    subnet: Callable[[int],nn.Module] 
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    norm: Optional[nn.Module] = ActNorm
    permutation: nn.Module = InvertibleConv1x1
    coupling: nn.Module = AffineCoupling

    @nn.compact
    def __call__(self, x, logdet=0, reverse=False):
        out_dims = x.shape[-1]
        subnet = self.subnet(out_dims)
        if not reverse:
            if self.norm:
                x, logdet = self.norm()(x, logdet=logdet, reverse=False)
            x, logdet = self.permutation(out_dims, self.key)(x, logdet=logdet, reverse=False)
            x, logdet = self.coupling(out_dims, subnet)(x, logdet=logdet, reverse=False)
        else:
            x, logdet = self.coupling(out_dims, subnet)(x, logdet=logdet, reverse=True)
            x, logdet = self.permutation(out_dims, self.key)(x, logdet=logdet, reverse=True)
            if self.norm:
                x, logdet = self.norm()(x, logdet=logdet, reverse=True)
        return x, logdet


class Sequential(nn.Module):

    modules: List[nn.Module]

    @nn.compact
    def __call__(self,x,logdet=0,reverse=False,num_modules=None):

        if num_modules:
            modules = [self.modules[-ind-1 if reverse else ind] for ind in range(num_modules)]
        else:
            modules = self.modules[::(-1 if reverse else 1)]

        for module in modules:
            x, logdet = module(x, logdet=logdet, reverse=reverse)

        # for module in self.modules[::(-1 if reverse else 1)]:
        #     x, logdet = module(x, logdet=logdet, reverse=reverse)

        return x, logdet