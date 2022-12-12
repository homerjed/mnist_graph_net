from cmath import isinf
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import jraph
import haiku as hk 

from get_mnist import load_mnist
from mnist_to_graphs import get_mnist_graphs, print_graph

def vgae_encoder(
    graph: jraph.GraphsTuple,
    hidden_dim: int,
    latent_dim: int) -> Tuple[jraph.GraphsTuple, jraph.GraphsTuple]:

  """VGAE network definition."""
  # graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  
  @jraph.concatenated_args
  def hidden_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for hidden layer."""
    net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.relu])
    return net(feats)

  @jraph.concatenated_args
  def latent_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for latent layer."""
    return hk.Linear(latent_dim)(feats)

  net_hidden = jraph.GraphConvolution(
    update_node_fn=hidden_node_update_fn,
    add_self_edges=True)
  h = net_hidden(graph)
  
  net_mean = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn,
    add_self_edges=True)
  net_log_std = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn,
    add_self_edges=True)

  # this is ouputting a graph with latent-dim dimensional nodes
  mean, log_std = net_mean(h), net_log_std(h)

  # try ouputting mean, logstd as outputs of linear layers: not graphs
  # encoded_graph_objs = jnp.concatenate([graph.nodes, graph.edges], axis=-1)
  # edges_ = hk.Linear(latent_dim, with_bias=False)(graph.edges.reshape(1, -1))

  mean = hk.Linear(latent_dim, with_bias=False)(graph.nodes.reshape(1, -1))
  log_std = hk.Linear(latent_dim, with_bias=False)(graph.nodes.reshape(1, -1))
  return mean, log_std


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    X, Y = load_mnist()
    dataset = get_mnist_graphs(X, Y, n_graphs=16, plot=False)
    x = dataset[0]["input_graph"]

    encoder = partial(vgae_encoder, hidden_dim=128, latent_dim=4)
    net = hk.without_apply_rng(hk.transform(encoder))
    params = net.init(key, x)
    m, s = net.apply(params, x)
    print("x:\n") 
    print_graph(x)
    print("\n")
    print("m:\n") 
    if isinstance(m, jraph.GraphsTuple):
        print_graph(m)
    else:
        print(m.shape)
    print("\n")
    print("s:\n")
    if isinstance(m, jraph.GraphsTuple):
        print_graph(s)
    else:
        print(s.shape)
