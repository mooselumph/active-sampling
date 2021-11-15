import flax
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    features = [10, 1]
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)  
        return x
    



class DeepSet(nn.Module):
    phi_features = [512, 20]
    rho_features = [512, 20]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.phi_features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.phi_features) - 1:
                x = nn.relu(x)  

        x = jnp.sum(x, axis=0, keepdims=True)

        for i, feat in enumerate(self.rho_features):
            x = nn.Dense(feat, name=f'layers_{i + len(self.phi_features)}')(x)
            if i != len(self.rho_features) - 1:
                x = nn.relu(x)  
        return x


# Test for permutation invariance
def test_deepset():
    key = jax.random.PRNGKey(0)
    key, new_key = jax.random.split(key)
    x = jax.random.uniform(new_key, shape=(5, 20))

    model = DeepSet()
    key, new_key = jax.random.split(key)
    params = model.init(new_key, x)
    output = model.apply(params, x)

    key, new_key = jax.random.split(key)
    permuted_x = jax.random.permutation(new_key, x)
    permuted_output = model.apply(params, permuted_x)


    assert not jnp.allclose(x, permuted_x)
    assert jnp.allclose(output, permuted_output)
    



if __name__ == '__main__':
    test_deepset()