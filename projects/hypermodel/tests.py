from data import *
import jax
import jax.numpy as jnp



def test_poly():
    x = 2.0
    answer = x**2 + x + 1.5
    fun = polynomial([1, 1, 1.5])
    assert fun(x) == answer


def test_generate_points():
    coefficients = [0, 1, 0]
    f = polynomial(coefficients)
    key = jax.random.PRNGKey(0)
    points = generate_points(f, key, num_points=3)
    print(points)

def test_fourier_encoding():
    x = 1.5
    return fourier_positional_encoding(x, max_freq=6)


if __name__ == '__main__':
    res = test_fourier_encoding()
    print(res)