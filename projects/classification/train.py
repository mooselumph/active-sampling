import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

from omegaconf import OmegaConf
import optax

from functools import partial


@partial(jax.jit, static_argnums=(0,1,))
def apply_model(config, model, state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = model.apply({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, config.model.num_classes)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def create_train_state(config, model, trainloader, rng):

# ASK ROBERT: The actual datapoint doesn't matter here, right?
  x, _ = next(iter(trainloader))
  """Creates initial `TrainState`."""
  params = model.init(rng, x)['params']
  tx = optax.sgd(config.train.learning_rate, config.train.momentum)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.vmap
def get_class(z):
  y = 1*(z[0]+z[1]-2*z[2] > 0)
  return y

def train_epoch(config, model, state, trainloader, rng):
  """Train for a single epoch."""

  epoch_loss = []
  epoch_accuracy = []

  for x, z in trainloader:
    y = get_class(z)

    grads, loss, accuracy = apply_model(config, model, state, x, y)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)

  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy



def train_and_evaluate(
                      model: nn.Module,
                      trainloader,
                      testloader,
                      config,
                      workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(OmegaConf.to_container(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(config, model, trainloader, init_rng)

  testloader_iterator = iter(testloader)

  for epoch in range(1, config.train.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(config, model, state, trainloader,
                                                    input_rng)


  
    try:
      x,z = next(testloader_iterator)
    except StopIteration:
      testloader_iterator = iter(testloader)
      x,z = next(testloader_iterator)

    y = get_class(z)
    _, test_loss, test_accuracy = apply_model(config, model, state, x, y)


    print(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state