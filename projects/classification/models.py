
import jax
import jax.numpy as jnp

from flax import linen as nn
import optax

class CNN(nn.Module):

  num_classes: int = 10

  def setup(self):
    self.features = Features()
    self.classifier = Classifier(self.num_classes)

  def __call__(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

class Features(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    return x

class Classifier(nn.Module):
  num_classes: int = 10

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    x = nn.log_softmax(x)
    return x


@jax.jit
def features(params, x):
  return CNN().apply({"params": params}, x,
    method=lambda module, x: module.features(x))


