import logging
import functools
from typing import Tuple, Union, List, Dict, Any

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
import numpy as np 
import matplotlib.pyplot as plt

from get_mnist import load_mnist
from mnist_to_graphs import get_mnist_graphs, pad_graph_to_nearest_power_of_two, print_graph


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential([
    hk.Linear(128), 
    jax.nn.relu,
    hk.Linear(128)])
  return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential([
    hk.Linear(128), 
    jax.nn.relu,
    hk.Linear(128)])
  return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # MUTAG is a binary classification task, so output pos neg logits.
  net = hk.Sequential([
    hk.Linear(128), 
    jax.nn.relu, 
    hk.Linear(10)]) # output MNIST classes
  return net(feats)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  # Add a global paramater for graph classification. (GRAPH ALREADY HAS GLOBAL PARAMETER)
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(128), 
      hk.Linear(128), 
      hk.Linear(128))
  net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn,)
#   embed = embedder(graph)
#   print_graph(embed)
#   out = net(embed)
#   print_graph(out)
#   return out
  return net(embedder(graph))


def compute_loss(
    params: hk.Params, 
    graph: jraph.GraphsTuple, 
    label: jnp.ndarray,
    net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
  pred_graph = net.apply(params, graph)
  preds = jax.nn.log_softmax(pred_graph.globals)
  targets = jax.nn.one_hot(label, 10)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  mask = jraph.get_graph_padding_mask(pred_graph)

  # Cross entropy loss.
  loss = -jnp.mean(preds * targets * mask[:, None])

  # Accuracy taking into account the mask.
  accuracy = jnp.sum(
    (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
  return loss, accuracy
 

def train(dataset: List[Dict[str, Any]], num_train_steps: int) -> hk.Params:
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
  graph = pad_graph_to_nearest_power_of_two(graph)
  print("INITIAL GRAPH: \n")
  print_graph(graph)

  # Initialize the network.
  params = net.init(jax.random.PRNGKey(42), graph)
  print("PARAMETERS INITIALISED.")

  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-4)
  opt_state = opt_init(params)
  print("OPTIMISER INITIALISED.")

  compute_loss_fn = functools.partial(compute_loss, net=net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
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
    graph = pad_graph_to_nearest_power_of_two(graph)
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
    params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluation Script."""
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))

  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']

  accumulated_loss = 0.
  accumulated_accuracy = 0.

  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for idx in range(len(dataset)):
    graph = dataset[idx]['input_graph']
    label = dataset[idx]['target']

    graph = pad_graph_to_nearest_power_of_two(graph)
    label = jnp.concatenate([label, jnp.array([0])])

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
    dataset = get_mnist_graphs(X, Y)

    params, losses, accs = train(dataset, num_train_steps=10_000)

    evaluate(dataset, params)

    fig, axs = plt.subplots(1, 2, dpi=200, figsize=(8.,4.))
    ax = axs[0]
    ax.plot(losses)
    ax = axs[1]
    ax.plot(accs)
    plt.show()

    # test it 
    net = hk.without_apply_rng(hk.transform(net_fn))
    for _ in range(80):
        y_= net.apply(params, dataset[_]["input_graph"])
        preds = jax.nn.log_softmax(y_.globals)

        # mask = jraph.get_graph_padding_mask(pred_graph)
        targets = jax.nn.one_hot(Y[_], 10)

        # print(preds, targets)
        print(jnp.argmax(preds), jnp.argmax(targets), "\n")
        # print(y_, Y[_])



  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.

  # Cross entropy loss.
#   loss = -jnp.mean(preds * targets * mask[:, None])

#   # Accuracy taking into account the mask.
#   accuracy = jnp.sum(
#     (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)