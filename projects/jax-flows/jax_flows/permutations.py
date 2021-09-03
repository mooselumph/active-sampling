import jax
import jax.numpy as jnp

from jax.ops import index, index_update

import flax.linen as nn

import numpy as np


# from einops import rearrange, reduce, repeat


class InvertibleConv1x1(nn.Module):
    channels: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        c = self.channels
        # Sample random rotation matrix
        q, _ = jnp.linalg.qr(jax.random.normal(self.key, (c, c)), mode='complete')
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = jnp.triu(u, k=1)
        s = jnp.diag(u)
        self.sign_s = jnp.sign(s)
        S_log_init = jnp.log(jnp.abs(s))
        self.l_mask = jnp.tril(jnp.ones((c, c)), k=-1)
        self.u_mask = jnp.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (c, c))
        self.U = self.param("U", lambda k, sh: U_init, (c, c))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (c,))
        
    def __call__(self, inputs, logdet=0, reverse=False):
        c = self.channels
        assert c == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + jnp.eye(c)
        U = self.U * self.u_mask + jnp.diag(self.sign_s * jnp.exp(self.log_s))
        logdet_factor = inputs.shape[1] * inputs.shape[2]
        
        # forward
        if not reverse:
            # TODO: Switch to einops notation
            # lax.conv uses weird ordering: NCHW and OIHW
            W = jnp.matmul(self.P, jnp.matmul(L, U))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)), 
                             W[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet += jnp.sum(self.log_s) * logdet_factor
        # inverse
        else:
            W_inv = jnp.matmul(jax.scipy.linalg.inv(U), jnp.matmul(
                jax.scipy.linalg.inv(L), self.P_inv))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)),
                             W_inv[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet -= jnp.sum(self.log_s) * logdet_factor
            
        return y, logdet


class FixedPermutation(nn.Module):
    width: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    def setup(self):

        self.perm = jax.random.permutation(self.key,self.width)
       
        perm_inv = jnp.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            perm_inv = index_update(perm_inv, index[p], i)

        self.perm_inv = perm_inv



    def __call__(self,x,logdet=0.,reverse=False):

        if not reverse:
            return x[:, self.perm], logdet
        else:
            return x[:, self.perm_inv], logdet



class InvertibleLinear(nn.Module):
    width: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        n = self.width
        # Sample random rotation matrix
        q, _ = jnp.linalg.qr(jax.random.normal(self.key, (n, n)), mode='complete')
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = jnp.triu(u, k=1)
        s = jnp.diag(u)
        self.sign_s = jnp.sign(s)
        S_log_init = jnp.log(jnp.abs(s))
        self.l_mask = jnp.tril(jnp.ones((n, n)), k=-1)
        self.u_mask = jnp.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (n, n))
        self.U = self.param("U", lambda k, sh: U_init, (n, n))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (n,))
        
    def __call__(self, inputs, logdet=0, reverse=False):
        n = self.width
        assert n == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + jnp.eye(n)
        U = self.U * self.u_mask + jnp.diag(self.sign_s * jnp.exp(self.log_s))
        
        # forward
        if not reverse:
            # lax.conv uses weird ordering: NCHW and OIHW
            W = jnp.matmul(self.P, jnp.matmul(L, U))
            y = jnp.matmul(inputs,W)
            logdet += jnp.sum(self.log_s) 
        # inverse
        else:
            W_inv = jnp.matmul(jax.scipy.linalg.inv(U), jnp.matmul(
                jax.scipy.linalg.inv(L), self.P_inv))
            y = jnp.matmul(inputs,W_inv)
            logdet -= jnp.sum(self.log_s) 
            
        return y, logdet
