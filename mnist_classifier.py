from functools import partial
from typing import (
  Tuple, List, Dict, Any)

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
import matplotlib.pyplot as plt

from get_mnist import load_mnist
from mnist_to_graphs import (
  get_mnist_graphs, 
  pad_graph_to_nearest_power_of_two, 
  pad_graph_to_value,
  print_graph)

Array = jnp.ndarray
Graph = jraph.GraphsTuple

PAD_VALUE = 1024
F = 32
E = 8


@jraph.concatenated_args
def edge_update_fn(feats: Array) -> Array:
  """Edge update function for graph net."""
#   net = hk.Sequential([hk.Linear(F), jax.nn.leaky_relu, hk.Linear(F, with_bias=False)])
  net = hk.nets.MLP(
    [F, F // 2, F // 4], 
    activation=jax.nn.leaky_relu)
  return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: Array) -> Array:
  """Node update function for graph net."""
#   net = hk.Sequential([hk.Linear(F), jax.nn.leaky_relu, hk.Linear(F, with_bias=False)])
  net = hk.nets.MLP(
    [F, F // 2, F // 4],
    activation=jax.nn.leaky_relu)
  return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: Array) -> Array:
  """ Global update function for graph net. """
  net = hk.Sequential([
    hk.Linear(F), jax.nn.leaky_relu, 
    hk.Linear(F // 2), jax.nn.leaky_relu, 
    hk.Linear(10, with_bias=False)]) # output MNIST classes
  return net(feats)

class Identity(hk.Module):
  def __call__(self, x):
    return x

def net_fn(graph: Graph) -> Graph:
  # Add a global paramater for graph classification. (GRAPH ALREADY HAS GLOBAL PARAMETER)
  #graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
#   print_graph(graph)
  embedder = jraph.GraphMapFeatures(
    Identity(), Identity(), Identity())
    # embed_edge_fn=hk.Linear(E, with_bias=False), 
    # embed_node_fn=hk.Linear(E, with_bias=False), 
    # embed_global_fn=hk.Linear(E, with_bias=False))
  net = jraph.GraphNetwork(
    update_node_fn=node_update_fn,
    update_edge_fn=edge_update_fn,
    update_global_fn=update_global_fn)
  # Final MLP to classify based on graph global features
#   classifier = hk.nets.MLP(
#     [F, F // 2, F // 4, 10],
#     activation=jax.nn.leaky_relu)
  #return jax.nn.softmax(classifier(net(embedder(graph)).globals))
  x = net(embedder(graph)) 
#   x = embedder(graph)
#   for _ in range(3):
#     x = net(x)
  #print("x: \n", x)
  #return jax.nn.softmax(classifier(x.globals))
  return x


def compute_loss(
    params: hk.Params, 
    graph: Graph, 
    label: Array,
    net: Graph) -> Tuple[Array, Array]:
  """Computes loss and accuracy."""

  pred_graph = net.apply(params, graph)

  n_batch = graph.nodes.shape
  print("SAUSAGE", n_batch)

  # Output of GNN and target: one hot encoded MNIST labels
 # preds = pred_graph 
  preds = jax.nn.log_softmax(pred_graph.globals)
  targets = jax.nn.one_hot(label, 10)
  print("BACON", targets)

  #print_graph(graph)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  # (JAX maintains a cache of compiled programs, the compilation cost is amortized.)
  # Remove padding during loss calculation so it is not 'counted' in loss
  mask = jraph.get_graph_padding_mask(pred_graph)
  #print("EGG", pred_graph)

  # Cross entropy loss.
  loss = -(preds * targets * mask[:, None]).mean()
 # loss = -(preds * targets).mean()
  #print(preds.shape)
  #loss = -(preds[:-1] * targets).mean()
  #print(preds.shape, targets.shape)
  #loss = jnp.mean(jnp.square(jnp.subtract(preds, targets) * mask[:, None]))

  # Accuracy taking into account the mask.
  #mask = jraph.get_graph_padding_mask(pred_graph)
  accuracy = jnp.sum(
    (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
#   accuracy = jnp.sum((jnp.argmax(preds[:-1], axis=1) == label)[:-1])
  return loss, accuracy
 

def train(
  dataset: List[Dict[str, Any]], 
  num_train_steps: int) -> Tuple[hk.Params, List, List]:
  """Training loop."""

  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
#   g = graph
#   print(f'Number of nodes: {g.n_node[0]}')
#   print(f'Number of edges: {g.n_edge[0]}')
#   print(f'Node features shape: {g.nodes.shape}')
#   print(f'Edge features shape: {g.edges.shape}')
  #graph = pad_graph_to_nearest_power_of_two(graph)
  graph = pad_graph_to_value(graph, PAD_VALUE)
  print("INITIAL GRAPH: \n")
  print_graph(graph)

  # Initialize the network.
  params = net.init(jax.random.PRNGKey(0), graph)
  print("PARAMETERS INITIALISED.")

  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-3)
  opt_state = opt_init(params)
  print("OPTIMISER INITIALISED.")

  compute_loss_fn = partial(compute_loss, net=net)
  compute_loss_fn = jax.jit(jax.value_and_grad(
    compute_loss_fn, has_aux=True))

  losses, accs = [], []
  for idx in range(num_train_steps):
    graph = dataset[idx % len(dataset)]['input_graph']
    label = dataset[idx % len(dataset)]['target']
    # Jax will re-jit your graphnet every time a new graph shape is encountered.
    # In the limit, this means a new compilation every training step, which
    # will result in *extremely* slow training. To prevent this, pad each
    # batch of graphs to the nearest power of two. Since jax maintains a cache
    # of compiled programs, the compilation cost is amortized.
    #graph = pad_graph_to_nearest_power_of_two(graph)
    graph = pad_graph_to_value(graph, PAD_VALUE)
    # graph = graph._replace(edges=jnp.zeros(graph.edges.shape))

    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
    label = jnp.concatenate([label, jnp.array([0])])

    (loss, acc), grad = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    if idx % 50 == 0:
      print(f'step: {idx:06d}, loss: {loss:.4f}, acc: {acc:.4f}')
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

    #graph = pad_graph_to_nearest_power_of_two(graph)
    graph = pad_graph_to_value(graph, PAD_VALUE)
    label = jnp.concatenate([label, jnp.array([0])]) # add padding label to graphtuple
    #print_graph(graph)

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
  dataset = get_mnist_graphs(X, Y, n_graphs=100, plot=True)

  params, losses, accs = train(dataset, num_train_steps=30_000)

  evaluate(dataset, params)

  fig, axs = plt.subplots(1, 2, dpi=200, figsize=(8.,4.))
  ax = axs[0]
  ax.plot(losses[::10])
  ax = axs[1]
  ax.plot(accs[::10])
  plt.show()

  n_plot = 8

  # test it 
  net = hk.without_apply_rng(hk.transform(net_fn))
  fig, axs = plt.subplots(n_plot, n_plot, figsize=(6.,6.), dpi=200)
  c, n = 0, 0
  total_accuracy = 0.
  for _, ax in enumerate(axs.reshape(-1)):

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
  test_graph = dataset[0]["input_graph"]
  test_label = dataset[0]["target"]
  #test_graph.nodes


  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.

  # Cross entropy loss.
#   loss = -jnp.mean(preds * targets * mask[:, None])

#   # Accuracy taking into account the mask.
#   accuracy = jnp.sum(
#     (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)