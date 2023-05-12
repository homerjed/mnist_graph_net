from functools import partial
from typing import (
  Tuple, List, Dict, Any, Sequence, Callable)

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr 
import jraph
import optax
import matplotlib.pyplot as plt

from get_mnist import load_mnist
from mnist_to_graphs import (
  get_mnist_graphs, 
  get_rotated_mnist_graphs,
  pad_graph_to_nearest_power_of_two, 
  pad_graph_to_value,
  print_graph)

Array = jnp.ndarray
Graph = jraph.GraphsTuple

N_GRAPHS = 10
N_STEPS = 50_000
PAD_VALUE = 2048 #512 #256
R_LINK = 0.15 #2. #1.5
F = 128
E = 32

"""
    try normalising x,y data before getting edge attrs
"""

class LinearResNet(hk.Module):
  def __init__(
    self, 
    hidden_sizes: Sequence[int],
    activation: Callable = jax.nn.leaky_relu,
    activate_final: bool = False):
    super().__init__()
    self.hidden_sizes = hidden_sizes
    self.activation = activation
    self.activate_final = activate_final

  def __call__(self, x):
    # assert x.shape[-1] == self.hidden_size, (
    #   "Input must be hidden size.")
    z = x
    for f in self.hidden_sizes:
      h = self.activation(z)
      h = hk.Linear(f)(h)
      h = self.activation(h)
      h = hk.Linear(f, w_init=jnp.zeros)(h)
      z = z + h 
    return self.activation(z) if self.activate_final else z

@jraph.concatenated_args
def edge_update_fn(feats: Array) -> Array:
  """ Edge update function for graph net. """
  net = hk.Sequential([
    hk.Linear(F),
    LinearResNet([F, F, F])])
  return net(feats)

@jraph.concatenated_args 
def node_update_fn(feats: Array) -> Array:
  """ Node update function for graph net. """
  net = hk.Sequential([
    hk.Linear(F),
    LinearResNet([F, F, F])])
  return net(feats)

@jraph.concatenated_args
def update_global_fn(feats: Array) -> Array:
  """ Global update function for graph net. """
  net = hk.Sequential([
    hk.Linear(F),
    LinearResNet([F, F, F], activate_final=True),
    hk.Linear(10, with_bias=False)]) # output MNIST classes
  return net(feats)

def net_fn(graph: Graph) -> Graph:
  embedder = jraph.GraphMapFeatures(
    embed_edge_fn=hk.Linear(E, with_bias=False), 
    embed_node_fn=hk.Linear(E, with_bias=False), 
    embed_global_fn=hk.Linear(E, with_bias=False))
  net = jraph.GraphNetwork(
    update_node_fn=node_update_fn,
    update_edge_fn=edge_update_fn,
    update_global_fn=update_global_fn)
  return net(embedder(graph)) 

def compute_loss(
  params: hk.Params, 
  graph: Graph, 
  label: Array,
  net: Graph) -> Tuple[Array, Array]:
  """Computes loss and accuracy."""

  pred_graph = net.apply(params, graph)

  # Output of GNN and target: one hot encoded MNIST labels
  # preds = pred_graph 
  preds = jax.nn.log_softmax(pred_graph.globals)
  targets = jax.nn.one_hot(label, 10)

  mask = jraph.get_graph_padding_mask(pred_graph)

  # Cross entropy loss.
  loss = -(preds * targets * mask[:, None]).mean()

  # Accuracy taking into account the mask.
  accuracy = jnp.sum(
    (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
  return loss, accuracy
 

def train(
  dataset: List[Dict[str, Any]], 
  num_train_steps: int) -> Tuple[hk.Params, List, List]:
  """Training loop."""
  key = jr.PRNGKey(0)

  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
  graph = pad_graph_to_value(graph, PAD_VALUE)

  # Initialize the network.
  params = net.init(key, graph)
  print(f"n_params = {sum(x.size for x in jax.tree_leaves(params))}")

  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-3)
  opt_state = opt_init(params)

  compute_loss_fn = partial(compute_loss, net=net)
  compute_loss_fn = jax.jit(jax.value_and_grad(
    compute_loss_fn, has_aux=True))

  losses, accs = [], []
  for _ in range(num_train_steps):
    key, key_idx = jr.split(key)
    idx = int(jr.randint(key_idx, shape=(1,), minval=0, maxval=N_GRAPHS))
    graph = dataset[idx]['input_graph']
    label = dataset[idx]['target']

    """ Should remove graph label, for model to generate itself. """
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
    graph = pad_graph_to_value(graph, PAD_VALUE)

    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
    """ This label gets masked out. """
    label = jnp.concatenate([label, jnp.array([0])])

    (loss, acc), grad = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    if _ % 100 == 0:
      print(f'step: {_:06d}, loss: {loss:.4f}, acc: {100. * acc:.4f}')
    losses.append(loss)
    accs.append(acc)

  print('Training finished')
  return params, losses, accs


def evaluate(
  dataset: List[Dict[str, Any]],
  params: hk.Params) -> Tuple[Array, Array]:
  """Evaluation Script."""
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))

  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']

  accumulated_loss = 0.
  accumulated_accuracy = 0.

  compute_loss_fn = jax.jit(partial(compute_loss, net=net))
  for idx in range(len(dataset)):
    graph = dataset[idx]['input_graph']
    label = dataset[idx]['target']

    graph = pad_graph_to_value(graph, PAD_VALUE)
    label = jnp.concatenate([label, jnp.array([0])]) # add padding label to graphtuple

    loss, acc = compute_loss_fn(params, graph, label)

    accumulated_accuracy += acc
    accumulated_loss += loss

    if idx % 100 == 0:
      print(f'Evaluated {idx + 1} graphs')

  print('Completed evaluation.')
  loss = accumulated_loss / idx
  accuracy = accumulated_accuracy / idx
  print(f'Eval loss: {loss:.4f}, accuracy {accuracy:.4f}')
  return loss, accuracy


if __name__ == "__main__":

  X, Y = load_mnist()
  dataset = get_mnist_graphs(
    X, Y, n_graphs=N_GRAPHS, plot=True, r_link=R_LINK)

  params, losses, accs = train(dataset, num_train_steps=N_STEPS)

  evaluate(dataset, params)

  fig, axs = plt.subplots(1, 2, dpi=200, figsize=(8.,4.))
  ax = axs[0]
  ax.semilogy(losses[::10])
  ax = axs[1]
  ax.plot(accs[::10])
  plt.show()

  n_plot = 8

  # test it 
  net = hk.without_apply_rng(hk.transform(net_fn))

  c, n = 0, 0
  total_accuracy = 0.
  for _ in range(len(dataset)):

    x = dataset[_]["input_graph"]
    label = dataset[_]["target"]
    #x = pad_graph_to_nearest_power_of_two(x)
    x = pad_graph_to_value(x, PAD_VALUE)
    label = jnp.concatenate([label, jnp.array([0])])

    y_ = net.apply(params, x)
    mask = jraph.get_graph_padding_mask(y_)

    accuracy = jnp.sum(     
      (jnp.argmax(y_.globals, axis=1) == label) * mask) / jnp.sum(mask)
    print(f"{jnp.argmax(y_.globals, axis=1)[0]} {int(dataset[_]['target'])}")

    total_accuracy += accuracy

  total_accuracy /= _
  print(f"ACCURACY = {total_accuracy:.2f}%")

  # Try rotating the graph to test invariance of classification
  print("TESTING ROTATION INVARIANCE.")
  key = jr.PRNGKey(1294)
  n_test = 10
  ix = jr.randint(key, shape=(n_test,), minval=0, maxval=n_test)
  dataset = get_rotated_mnist_graphs(
    X[ix], Y[ix], n_graphs=n_test, r_link=R_LINK, plot=True)
  
  for i in range(n_test):
    x, y = dataset[i]["input_graph"], dataset[i]["target"]
    x = pad_graph_to_value(x, PAD_VALUE)
    label = jnp.concatenate([y, jnp.array([0])])

    y_ = net.apply(params, x)
    mask = jraph.get_graph_padding_mask(y_) 

    accuracy = jnp.sum(     
      (jnp.argmax(y_.globals, axis=1) == y) * mask) / jnp.sum(mask)
    print(f"{jnp.argmax(y_.globals, axis=1)[0]} {int(dataset[i]['target'])}")