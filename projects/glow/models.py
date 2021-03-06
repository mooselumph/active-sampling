import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial

from utils import make_absolute

# from einops import rearrange, reduce, repeat

from jax_flows.utils import ConvZeros
from jax_flows.reshaping import squeeze, unsqueeze, Split
from jax_flows.blocks import FlowStep
from jax_flows.subnets import ConvSubnet


def load_model(config,train_ds,key):

    model = GLOW(**config.model.params)
    params = model.init(key, next(train_ds))

    if config.model.checkpoint_path:

        path = make_absolute(config.model.checkpoint_path)
        with open(path, 'rb') as f:
            params = flax.serialization.from_bytes(params, f.read())

    return model, params


class GLOW(nn.Module):
    K: int = 32                                       # Number of flow steps
    L: int = 3                                        # Number of scales
    nn_width: int = 512                               # NN width in Affine Coupling Layer
    learn_top_prior: bool = False                     # If true, learn prior N(mu, sigma) for zL
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
        
    def flows(self, x, logdet=0, reverse=False, name=""):
        """K subsequent flows. Called at each scale."""
        for k in range(self.K):
            it = k + 1 if not reverse else self.K - k
            subnet = partial(ConvSubnet,width=self.nn_width)
            x, logdet = FlowStep(subnet, self.key, name=f"{name}/step_{it}")(
                x, logdet=logdet, reverse=reverse)
        return x, logdet
        
    
    @nn.compact
    def __call__(self, x, reverse=False, z=None, eps=None, sampling_temperature=1.0):
        """Args:
            * x: Input to the model
            * reverse: Whether to apply the model or its inverse
            * z (reverse = True): If given, use these as intermediate latents (deterministic)
            * eps (reverse = True, z!=None): If given, use these as Gaussian samples which are later 
                rescaled by the mean and variance of the appropriate prior.
            * sampling_temperature (reverse = True, z!=None): Sampling temperature
        """
        
        ## Inputs
        # Forward pass: Save priors for computing loss
        # Optionally save zs (only used for sanity check of reversibility)
        priors = []
        if not reverse:
            del z, eps, sampling_temperature
            z = []
        # In reverse mode, either use the given latent z (deterministic)
        # or sample them. For the first one, uses the top prior.
        # The intermediate latents are sampled in the `Split(reverse=True)` calls
        else:
            if z is not None:
                assert len(z) == self.L
            else:
                x *= sampling_temperature
                if self.learn_top_prior:
                    # Assumes input x is a sample from N(0, 1)
                    # Note: the inputs to learn the top prior is zeros (unconditioned)
                    # or some conditioning e.g. class information.
                    # If not learnable, the model just uses the input x directly
                    # see https://github.com/openai/glow/blob/master/model.py#L109
                    prior = ConvZeros(x.shape[-1] * 2, name="prior_top")(jnp.zeros(x.shape))
                    mu, logsigma = jnp.split(prior, 2, axis=-1)
                    x = x * jnp.exp(logsigma) + mu
                
        ## Multi-scale model
        logdet = 0
        for l in range(self.L):
            # Forward
            if not reverse:
                x = squeeze(x)
                x, logdet = self.flows(x, logdet=logdet,
                                       reverse=False,
                                       name=f"flow_scale_{l + 1}/")
                if l < self.L - 1:
                    zl, x, prior = Split(
                        key=self.key, name=f"flow_scale_{l + 1}/")(x, reverse=False)
                else:
                    zl, prior = x, None
                    if self.learn_top_prior:
                        prior = ConvZeros(zl.shape[-1] * 2, name="prior_top")(jnp.zeros(zl.shape))
                z.append(zl)
                priors.append(prior)
                    
            # Reverse
            else:
                if l > 0:
                    x = Split(key=self.key, name=f"flow_scale_{self.L - l}/")(
                        x, reverse=True, 
                        z=z[-l - 1] if z is not None else None,
                        eps=eps[-l - 1] if eps is not None else None,
                        temperature=sampling_temperature)
                x, logdet = self.flows(x, logdet=logdet, reverse=True,
                                       name=f"flow_scale_{self.L - l}/")
                x = unsqueeze(x)
                
        ## Return
        return x, z, logdet, priors