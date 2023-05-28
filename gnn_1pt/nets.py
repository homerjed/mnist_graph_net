from typing import Sequence, Callable
import jax
import haiku as hk
import jraph
import jax.numpy as jnp


ACTIVATION = jax.nn.swish

layernorm_kwargs = dict(axis=1, create_scale=True, create_offset=True)

def layernorm(x):
    return hk.LayerNorm(**layernorm_kwargs)(x)


def LayerNormLinear(x, l, use_layernorm=False):
  h = hk.LayerNorm(**layernorm_kwargs)(x) if use_layernorm else x
  return hk.Linear(l, with_bias=False)(h)


class LinearResNet(hk.Module):
  def __init__(
    self, 
    hidden_sizes: Sequence[int],
    activation: Callable = ACTIVATION,
    activate_final: bool = False,
    dropout_rate: float = 0.1):
    super().__init__()
    self.hidden_sizes = hidden_sizes
    self.activation = activation
    self.activate_final = activate_final
    self.dropout_rate = dropout_rate
    self.use_layernorm = False

  def __call__(self, x, inference=False):
    z = x
    for f in self.hidden_sizes:
    #   h = self.activation(z)
    #   h = hk.LayerNorm(**layernorm_kwargs)(h)
      h = layernorm(z) if self.use_layernorm else z
      h = hk.Linear(f)(h)
      if not inference:
        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
      h = self.activation(h)

      #h = hk.LayerNorm(**layernorm_kwargs)(h)
      h = layernorm(h) if self.use_layernorm else h
      h = hk.Linear(f, w_init=jnp.zeros)(h)
      if not inference:
        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
      h = self.activation(z)

      z = z + h 
    return self.activation(z) if self.activate_final else z

