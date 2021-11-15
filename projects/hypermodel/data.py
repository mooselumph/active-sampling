import jax.numpy as jnp
import jax
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from utils import *

import sklearn.gaussian_process.kernels
from sklearn.gaussian_process.kernels import RBF

def polynomial(coeffs):
    def f(x):
        return jnp.polyval(jnp.array(coeffs), x)
    return f


def generate_points(poly_fun, rng_key, xlim=[-1, 1], num_points=100):
    minval, maxval = xlim
    x_points = jax.random.uniform(rng_key, shape=(num_points,), minval=minval, maxval=maxval)
    y_points = jax.vmap(poly_fun, 0, 0)(x_points)
    return {'x': x_points, 'y': y_points}



# Copied from Robert's Perceiver repo
# Reference:
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

def fourier_positional_encoding(x, max_freq, num_bands, base):
    x = jnp.expand_dims(x,-1)
    dtype, orig_x = x.dtype, x

    scales = jnp.logspace(0., jnp.log(max_freq / 2) / jnp.log(base), num_bands, base = base, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * jnp.pi
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis =-1)
    x = jnp.concatenate((x, orig_x), axis = -1)
    return x



class PolyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def create_train_test_loaders(x, y, train_split=0.8, batch_size=64):
    num_train = int(train_split*len(x))

    train_x = x[:num_train]
    train_y = y[:num_train]
    train_ds = PolyDataset(train_x, train_y)

    test_x = x[num_train:]
    test_y = y[num_train:]
    test_ds = PolyDataset(test_x, test_y)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def get_poly_data(config):
    coeff = omegaconf_list_to_array(config.data.polynomial.coeff)
    coefficients = jnp.array(coeff)*config.data.polynomial.scale
    f = polynomial(coefficients)
    key = jax.random.PRNGKey(0)
    points = generate_points(f, key, xlim=config.data.xlim, num_points=config.data.num_points)
    encode = jax.vmap(fourier_positional_encoding, in_axes=(0, None, None, None), out_axes=0)

    params = config.data.fourier
    encoded_points = encode(points['x'], params.max_freq, params.num_bands, params.base)

    return create_train_test_loaders(encoded_points, points['y'], 
                train_split=config.data.train_split, batch_size=config.train.batch_size)




def gaussian_process(key, num_draw_points, num_train_points, xlims, l):
    minval, maxval = xlims
    num_total_points = num_draw_points + num_train_points
    draw_points = jnp.linspace(minval, maxval, num_draw_points)
    x_points = jax.random.uniform(key, shape=(num_train_points,), minval=minval, maxval=maxval)

    all_points = jnp.concatenate((draw_points, x_points)).flatten()
    all_points = jnp.expand_dims(all_points, axis=1)
    all_points = np.array(all_points)
    K = RBF(length_scale=l)(all_points)
    y_vals = np.random.multivariate_normal(np.zeros(num_total_points), K)

    y_vals = jnp.array(y_vals)
    all_points = jnp.ravel(jnp.array(all_points))
    x_train = all_points[num_draw_points:]
    y_train = y_vals[num_draw_points:]
    x_draw = all_points[:num_draw_points]
    y_draw = y_vals[:num_draw_points]
    return x_train, y_train, x_draw, y_draw 


