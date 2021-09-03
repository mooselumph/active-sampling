import flax.linen as nn
import jax

from .utils import ConvZeros

class ConvSubnet(nn.Module):

    out_dims: int
    width: int = 512
    identity_init: bool = True

    def setup(self):
        if self.identity_init:
            self.final_layer_init = dict(
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            self.final_layer_init = dict()
        

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(features=self.width, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', name="ACL_conv_1")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.width, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', name="ACL_conv_2")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.out_dims, kernel_size=(3, 3),strides=(1, 1),
                        padding='same', name="ACL_conv_out", **self.final_layer_init)(x)
        
        return x


class MlpSubnet(nn.Module):

    out_dims: int
    width: int = 392
    identity_init: bool = True

    def setup(self):
        if self.identity_init:
            self.final_layer_init = dict(
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            self.final_layer_init = dict()

    @nn.compact
    def __call__(self, x):

        x = nn.Dense(features=self.width, name="ACL_dense_1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.width, name="ACL_dense_2")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dims, name="ACL_dense_out", **self.final_layer_init)(x)

        return x