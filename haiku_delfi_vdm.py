import time
import os 
from functools import partial
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as jr 
from jax.scipy.stats import norm, multivariate_normal 
import optax
from tqdm.notebook import trange, tqdm
from sklearn.datasets import make_moons

import haiku as hk
import distrax as dx

Array = jnp.ndarray

def scale(x):
  return 2. * ((x - x.min()) / (x.max() - x.min())) - 1.

def get_moons_data(N, rng):
  x, y = make_moons(
    n_samples=N, noise=0.05, random_state=int(rng.sum()))
  x = jnp.asarray(x)
  y = jnp.asarray(y)[:, None]
  return scale(x), y 

def get_test_data(N, rng):
  return (
    scale(
      jr.multivariate_normal(
        rng, 
        mean=jnp.ones(3), 
        cov=jnp.eye(3), 
        shape=(batch_size, ))),
    jnp.concatenate(
      [jnp.ones((N // 2, 1)), jnp.zeros((N // 2, 1))], axis=0))

class DenseMonotone(hk.Linear):
  """Strictly increasing Dense layer."""

  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = hk.get_parameter(
      'kernel',
      init=self.kernel_init,
      shape=(inputs.shape[-1], self.features))
    kernel = abs(jnp.asarray(kernel, self.dtype))
    y = jax.lax.dot_general(
      inputs, 
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision)
    if self.use_bias:
      bias = hk.get_parameter(
        'bias', init=self.bias_init, shape=(self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y

class NoiseScheduleNet(hk.Module):
  n_features: int = 8
  nonlinear: bool = True
  init_gamma_0: float = -13.3
  init_gamma_1: float = 5.
  init_bias = init_gamma_0
  init_scale = init_gamma_1 - init_gamma_0
  n_out = 1

  def __call__(self, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    self.l1 = DenseMonotone(
      self.n_out,
      kernel_init=hk.initializers.Constant(self.init_scale),
      bias_init=hk.initializers.Constant(self.init_bias))
    if self.nonlinear:
      self.l2 = DenseMonotone(
        self.n_features, 
        kernel_init=hk.initializers.VarianceScaling())
      self.l3 = DenseMonotone(
        self.n_out, 
        kernel_init=hk.initializers.VarianceScaling(),
        use_bias=False)

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))

    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (jax.nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h

    return h.squeeze(axis=-1)

class NoiseSchedule(hk.Module):
  """
    Simple scalar noise schedule, i.e. gamma(t) in the paper:
    gamma(t) = abs(w) * t + b
  """
  init_gamma_0: float = -13.3
  init_gamma_1: float = 5.
  init_bias = init_gamma_0
  init_scale = init_gamma_1 - init_gamma_0

  def __call__(self, t): # y
    self.w = hk.get_parameter(
      "w", 
      init=hk.initializers.Constant(self.init_scale),
      shape=(1,))
    self.b = hk.get_parameter(
      "b", 
      init=hk.initializers.Constant(self.init_bias),
      shape=(1,))
    gamma = abs(self.w) * t + self.b
    return gamma 

class Base2FourierFeatures(hk.Module):
  # Create Base 2 Fourier features
  def __call__(self, inputs):
    # Creating base 2 Fourier features, is 8 assuming vocabsize = 256?
    # Is using latter numbers here => higher freqs?
    freqs = jnp.asarray(range(8), dtype=inputs.dtype) #[0, 1, ..., 7]
    w = 2. ** freqs * 2. * jnp.pi
    w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
    h = jnp.repeat(inputs, len(freqs), axis=-1)
    h *= w
    h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
    return h

class ScoreNetwork(hk.Module):
  data_dim: int = 3 
  hidden_units: int = 32
  init_gamma_0: float = -13.3
  init_gamma_1: float = 5.
  use_ff: bool = True 

  def __call__(self, z, gamma_t, y=None):

    if self.use_ff:
      self.ff = Base2FourierFeatures() 

    # Normalize gamma_t to [-1,+1] for Fourier features
    lb = self.init_gamma_0
    ub = self.init_gamma_1
    gamma_t_norm = 2. * ((gamma_t - lb) / (ub - lb)) - 1.  

    # Concatenate normalized gamma_t as extra feature
    h = jnp.concatenate([z, gamma_t_norm[:, None]], axis=1)

    # Append Fourier features and conditioning label
    if self.use_ff:
      h_ff = self.ff(h)
      h = jnp.concatenate([h, h_ff, y], axis=1)
    else:
      h = jnp.concatenate([h, y], axis=-1)
      
    h = hk.nets.MLP(
      3 * [self.hidden_units] + [data_dim],
      activation=jax.nn.swish)(h)

    return h

@hk.without_apply_rng
@hk.transform
def gamma_(t):
  return NoiseSchedule()(t)

@hk.without_apply_rng
@hk.transform
def score(x, t, y):
  return ScoreNetwork()(x, t, y)

def data_encode(x):
  return scale(x)

def data_decode(z_0_rescaled, gamma_0):
  return scale(z_0_rescaled)

def data_logprob(x, z_0_rescaled, gamma_0):
  """ Should this be gaussian log prob for gaussian decoding likelihood p(x|z_0)? """
  logprobs = data_decode(z_0_rescaled, gamma_0)
  # Equation 8 in paper (NOTE THIS GIVES LOG PROBS OF ALL DIMS?)
  var_0 = jax.nn.sigmoid(gamma_0) # try using sttdev_0 instead of var0?
  logprob = dx.MultivariateNormalDiag(
    loc=z_0_rescaled.reshape(-1, x.shape[-1]),
    scale_diag=jnp.ones(x.shape[-1]) * var_0).log_prob(x)
  return logprob

def data_generate_x(z_0, gamma_0, rng, scale_outputs=True):
  """ Sampling from p(x|z_0)? This looks like decoding/re-scaling here """
  var_0 = jax.nn.sigmoid(gamma_0)
  z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
  samples = data_decode(z_0_rescaled, gamma_0)
  return samples

@jax.jit
def vdm_vlb(params, x, y=None, rng=None):
  #gamma = lambda t: model.apply(params, t, y, method=Model.gamma)
  gamma = lambda t: gamma_.apply(params, t)

  # Begin/end time points for gammas
  gamma_0, gamma_1 = gamma(0.), gamma(1.)
  # Variance preserving: gamma constrains mean and variance of chain's conditonal Gaussians
  var_0, var_1 = jax.nn.sigmoid(gamma_0), jax.nn.sigmoid(gamma_1)

  n_batch, data_dim = x.shape

  # Encode: simply scale to [-1, 1] in continuous data case
  f = data_encode(x)

  # 1. RECONSTRUCTION LOSS
  # Add noise and reconstruct
  rng, rng1 = jr.split(rng)
  # Reparameterization trick: auxiliary noise variable
  eps_0 = jr.normal(rng1, shape=f.shape)
  # Add noise: z_0 = alpha_0 * x_encoded + sigma_0 * eps (see below equation 13)
  alpha_0 = jnp.sqrt(1. - var_0) 
  z_0 = alpha_0 * f + jnp.sqrt(var_0) * eps_0
  # Reconstructed image (last latent), should be [-1, 1]? eps_0/sqrt(exp(gamma_0))
  z_0_rescaled = f + jnp.exp(0.5 * gamma_0) * eps_0  # = z_0/sqrt(1-var)
  # Reconstruction loss q(x|z_0) assuming equation 8 for discrete data
  stdev0 = jnp.exp(0.5 * gamma_0[..., None])
  # Assume Gaussian likelihood p(x|z) of decoded => MSE loss 
  # 1D Gaussian: maximize logL of data given gaussian continuous decoding parameters, all points are sampled from this dist, so logL is sum of this?
  loss_recon = -dx.MultivariateNormalDiag(
    loc=z_0_rescaled.reshape(-1, data_dim),
    scale_diag=jnp.ones((data_dim,)) * var_0).log_prob(x)

  # E_q(z_0|x)[-log p(x|z_0)]
  loss_recon = -multivariate_normal.logpdf(
    x, 
    mean=z_0_rescaled.reshape(-1, 2),
    cov=jnp.eye(2) * var_0)

  # 2. LATENT LOSS
  # KL z1 with N(0,1) prior (both gaussians, z1 approx of course)
  mean1_sqr = (1. - var_1) * jnp.square(f) # (alpha_1 * f) ** 2
  # KL divergence between two Gaussians: N(0, 1) and N(alpha_1, sigma_1)
  loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=1)

  # 3. DIFFUSION LOSS
  # Sample time steps (Monte Carlo unbiased estimate diffusion loss)
  rng, rng1 = jr.split(rng)
  t = jr.uniform(rng1, shape=(n_batch,))

  # Sample z_t
  gamma_t = gamma(t)
  var_t = jax.nn.sigmoid(gamma_t)[:, None]
  rng, rng1 = jr.split(rng)
  # For reparameterisation trick
  eps = jr.normal(rng1, shape=f.shape)
  # z_t = alpha_t * x + sigma * eps
  z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps

  # Compute predicted noise from prior latent z_t and gamma_t
  #eps_hat = model.apply(params, z_t, gamma_t, y=y, method=Model.score)
  eps_hat = score.apply(params, z_t, gamma_t, y=y)

  print(eps_hat.shape, eps_hat)

  # Compute MSE of predicted noise and noise added to image in forward process
  loss_diff_mse = jnp.square(jnp.subtract(eps, eps_hat)).sum(axis=1)

  # Loss for infinite depth T, i.e. continuous time, equation 15
  _, g_t_grad = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
  loss_diff = .5 * g_t_grad * loss_diff_mse # Equation 17

  # End of diffusion loss L_T computation
  return loss_klz, loss_recon, loss_diff

def loss_fn(params, x, y=None, rng=None):
  # Calculate VLB(x) = prior loss + reconstruction loss + diffusion loss 
  (
    loss_klz, 
    loss_recon, 
    loss_diff) = vdm_vlb(params, x, y, rng)
  loss = loss_klz + loss_recon + loss_diff
  metrics = [
    loss_klz.mean(),
    loss_recon.mean(),
    loss_diff.mean()]
  return loss.mean(), metrics

# @partial(
#     jax.jit,
#     static_argnames=[
#         "nde_args", "nde_type"])
@hk.without_apply_rng
@hk.transform
def log_prob(
  nde_args: Dict, #key,
  nde_type: str, #params,
  simulations: Array,
  parameters: Array,
  pdfs: Array = None):
  return vdm_vlb(
    params, x=simulations, y=parameters, rng=key)

@jax.jit
def train_step(rng, optim_state, params, x, y=None):
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  rng, rng1 = jr.split(rng)
  (loss, metrics), grads = grad_fn(params, x, y, rng=rng1)
  updates, optim_state = optimizer.update(grads, optim_state, params)
  params = optax.apply_updates(params, updates)
  return rng, optim_state, params, loss, metrics

@jax.jit
def sample_step(i, T_sample, z_t, rng, y=None):
  """ Sample VDM """

  eps = jr.normal(rng, z_t.shape)

  # Timesteps / latent indices in chain, 0 < s < t < 1
  t = (T_sample - i) / T_sample
  s = (T_sample - i - 1) / T_sample

  # Get SNR ratios at various times and broadcast them
  gamma_s = gamma_.apply(params, s, )#y)
  gamma_t = gamma_.apply(params, t, )#y)
  gamma_s *= jnp.ones((z_t.shape[0],), gamma_t.dtype)
  gamma_t *= jnp.ones((z_t.shape[0],), gamma_t.dtype)
    
  # Noise prediction from z_t
  eps_hat = score.apply(params, z_t, gamma_t, y)

  a = jax.nn.sigmoid(-gamma_s)[:, None]
  b = jax.nn.sigmoid(-gamma_t)[:, None]
  c = -jnp.expm1(gamma_s - gamma_t)[:, None]
  sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))[:, None]

  # Lower time latent z_s ~ p(z_s|z_t, x^), x^ = 
  z_s = jnp.sqrt(a / b) \
      * (z_t - sigma_t * c * eps_hat) \
      + jnp.sqrt((1. - a) * c) * eps

  alpha_t = jnp.sqrt(1. - b)
  # Predicted 
  x_pred = (z_t - sigma_t * eps_hat) / alpha_t

  return z_s, x_pred

def sample_fn(rng, params, N_sample, T_sample, data_dim, y=None):
  """ Sampler (T_sample is how many steps into depth) """
  # Sample z_0 from the diffusion model
  rng, rng1 = jr.split(rng)
  # Prior sample z_t=1
  z = [jr.normal(rng1, (N_sample, data_dim))] # ndims
  # Predicted steps in chain starting from x_pred|z_prior
  x_pred = []

  for i in trange(T_sample):
    rng, rng1 = jr.split(rng)
    _z, _x_pred = sample_step(i, T_sample, z[-1], rng1, y)
    z.append(_z)
    x_pred.append(_x_pred)

  # Gamma value at t=0: sample from data space
  gamma_0 = gamma_.apply(params, 0.) # variance at t=0?
  x_sample = data_generate_x(z[-1], gamma_0, rng, scale_outputs=True)

  return z, x_pred, x_sample


if __name__ == "__main__":
  rng = jr.PRNGKey(0)

  # Training args
  data_dim = 2
  learning_rate = 1e-3
  n_train_steps = 10_000 # VDM: loss better with T not train steps
  batch_size = 1024

  # Sampling args 
  T_sample = 1_000
  n_sample = 2_048 
  n_labels = n_sample // 2 # 2 classes

  t0 = time.time()

  rng, rng_gamma, rng_score = jr.split(rng, 3)
  noise_schedule_params = gamma_.init(
    rng_gamma, 
    jnp.zeros((1,))) # timestep t
  score_network_params = score.init(
    rng_score, 
    128 * jnp.ones((1, data_dim)), # data
    jnp.zeros((1,)),
    128 * jnp.ones((1, 1)))        # condition

  def tree_shape(xs):
    return jax.tree_util.tree_map(lambda x: x.shape, xs)
 
  print(tree_shape(noise_schedule_params))
  print(tree_shape(score_network_params))

  params = hk.data_structures.merge(
    noise_schedule_params, 
    score_network_params)
  print(tree_shape(params))

  # Initialize optimizer
  optimizer = optax.adamw(learning_rate)
  optim_state = optimizer.init(params)

  # Training loop (should take ~20 mins)
  losses, losses_recon, losses_latent, losses_diff = [], [], [], []

  for i in trange(n_train_steps):

    rng, rng_data = jr.split(rng)
  
    x, y = get_test_data(batch_size, rng_data)  
    x, y = get_moons_data(batch_size, rng_data)

    (
      rng, optim_state, params, loss, _metrics
    ) = train_step(
      rng, optim_state, params, x, y)

    bpd_latent, bpd_recon, bpd_diff = _metrics
    print((f"\r {i + 1:07d} / {n_train_steps} "
           f"L={loss:.4f} bpd_latent={bpd_latent:.4f} "
           f"bpd_recon={bpd_recon:.4f} bpd_diff={bpd_diff:.4f} " 
           f"t={(time.time() - t0) / 60.:.2f} mins"), end="")
    losses.append(loss)
    losses_recon.append(bpd_recon)
    losses_latent.append(bpd_latent)
    losses_diff.append(bpd_diff)

  print(f"\nDone. Time={(time.time() - t0) / 60.:.2f} mins.")

  """
    Plot training losses
  """

  f_plot = 10 
  fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.8))
  steps = range(len(losses_recon))[::f_plot]

  ax = axs[0]
  ax.plot(steps, losses[::f_plot], color="firebrick")

  ax = axs[1]
  ax.plot(steps, losses_recon[::f_plot], color="rebeccapurple", alpha=0.7, label="recon")
  ax.plot(steps, losses_latent[::f_plot], color="forestgreen", alpha=0.7, label="latent")
  ax.plot(steps, losses_diff[::f_plot], color="goldenrod", alpha=0.7, label="diffusion")
  ax.legend()

  # plt.setp(axs[0], ylim=axs[1].get_ylim(), xlim=axs[1].get_xlim())

  t_ = jnp.linspace(0., 1., 1_000)
  gammas = gamma_.apply(params, t_)
  ax = axs[2]
  ax.plot(t_, jax.nn.sigmoid(-gammas), label=r"$\sigma_t$")
  ax.plot(t_, jnp.sqrt(1. - jax.nn.sigmoid(-gammas)), label=r"$\alpha_t$")
  ax.legend()
  plt.tight_layout()
  plt.show()

  # Plot the learned noise schedule, gamma = gamma(t \in [0., 1.]) 
  print('gamma_0', gamma_.apply(params, t=0.))
  print('gamma_1', gamma_.apply(params, t=1.))
  print(min(losses), min(losses_recon))

  """
    Plot generated / training data
  """
  rng, rng1 = jr.split(rng)

  #x, y = get_test_data(n_sample, rng)
  x, y = get_moons_data(n_sample, rng)

  z, x_pred, _ = sample_fn(
    rng1, params, N_sample=n_sample, T_sample=T_sample, data_dim=data_dim, y=y)

  x_generated = x_pred[-1]
  x_generated = scale(x_generated) # is this automatically done?

  fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
  ax = axs[0]
  ax.set_title("generated")
  ax.scatter(*x_generated[:,:2].T, c=y, s=1.5, cmap="PiYG")
  ax = axs[1]
  ax.set_title("true")
  ax.scatter(*x[:,:2].T, c=y, s=1.5, cmap="PiYG")
  # Match axis limits of generated axes to true axes
  plt.setp(axs[0], ylim=axs[1].get_ylim(), xlim=axs[1].get_xlim())
  plt.tight_layout()
  plt.show()

  # Check VDM is producing data correctly
  print(f"x dim: {x[:,0].min()} {x[:,0].max()} - {x_generated[:,0].min()} {x_generated[:,0].max()}")
  print(f"y dim: {x[:,1].min()} {x[:,1].max()} - {x_generated[:,1].min()} {x_generated[:,1].max()}")

  print(f"\nDone. Time={(time.time() - t0) / 60.:.2f} mins.")
