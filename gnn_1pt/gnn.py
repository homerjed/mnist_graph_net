from functools import partial
import jax
import jax.numpy as jnp 
import jraph 
import haiku as hk

from nets import LinearResNet, LayerNormLinear

Array = jnp.ndarray
Graph = jraph.GraphsTuple


def graph_conv_net_fn(
  graph: Graph, 
  inference: bool = False, 
  n_layers: int = 1, 
  E: int = 8,
  use_layernorm: bool = True,
  residual: bool = False) -> Graph:

  global_embed_dim = 5 #len(alpha) if residual else E

  embedder = jraph.GraphMapFeatures(
    embed_edge_fn=None,
    embed_node_fn=partial(
        LayerNormLinear, l=E, use_layernorm=use_layernorm), 
    embed_global_fn=partial(
        LayerNormLinear, l=5, use_layernorm=use_layernorm))

  graph = embedder(graph)

  for _ in range(n_layers):
    net = jraph.GraphConvolution(
      update_node_fn=get_node_update_fn(inference=inference),
      update_global_fn=get_global_update_fn(inference=inference),
      aggregate_nodes_fn=jraph.segment_mean,
      aggregate_edges_for_globals_fn=jraph.segment_mean)

      # attention_logit_fn=attention_logit_fn,
      # attention_reduce_fn=attention_reduce_fn)#jraph.segment_sum)
    if residual:
       graph = add_graphs_tuples(net(graph), graph)
  return graph


def graph_network_fn(
  graph: Graph, 
  inference: bool = False, 
  n_layers: int = 2, 
  E: int = 8,
  use_layernorm: bool = False,
  residual: bool = False) -> Graph:

  global_embed_dim = 5 #len(alpha) if residual else E

  embedder = jraph.GraphMapFeatures(
    embed_edge_fn=None,
    embed_node_fn=partial(
        LayerNormLinear, l=E, use_layernorm=use_layernorm), 
    embed_global_fn=partial(
        LayerNormLinear, l=5, use_layernorm=use_layernorm))

  graph = embedder(graph)

  for _ in range(n_layers):
    net = jraph.GraphNetwork(
      update_edge_fn=None,
      update_node_fn=get_node_update_fn(inference=inference),
      update_global_fn=get_global_update_fn(inference=inference),
      aggregate_nodes_for_globals_fn=jraph.segment_mean,
      aggregate_edges_for_globals_fn=jraph.segment_mean)

      # attention_logit_fn=attention_logit_fn,
      # attention_reduce_fn=attention_reduce_fn)#jraph.segment_sum)
    if residual:
       graph = add_graphs_tuples(net(graph), graph)
  return graph


def node_net_fn(
  x: Array, 
  F: int = 32, 
  inference: bool = False):
  """ Mapping (v_i, agg(E_i), u) -> v'_i"""
#   h = LinearResNet([F, F, F])(x, inference=inference)
  h = hk.nets.MLP([F, F, F], activation=jax.nn.leaky_relu)(
    x, dropout_rate=None if inference else 0.1, rng=hk.next_rng_key())
  return h 


def global_net_fn(
  x: Array, 
  parameter_dim: int = 5, 
  F: int = 32, 
  inference: bool = False):
  """ Mapping initial global property u0=(N_e, N_v) -> u """
  # This needs to take into account conditioning information e.g. z?!
  # h = LinearResNet([F, F, F])(x, inference=inference)
  h = hk.nets.MLP([F, F, F], activation=jax.nn.leaky_relu)(
    x, dropout_rate=None if inference else 0.1, rng=hk.next_rng_key())
  h = hk.Linear(parameter_dim, with_bias=False)(h)
  return h


def get_node_update_fn(inference: bool):
    net_fn_ = partial(node_net_fn, inference=inference)
    return jraph.concatenated_args(net_fn_)


def get_global_update_fn(inference: bool):
    net_fn_ = partial(global_net_fn, inference=inference)
    return jraph.concatenated_args(net_fn_)


def add_graphs_tuples(graphs: Graph, other_graphs: Graph) -> Graph:
  graphs = graphs._replace(
    nodes=graphs.nodes + other_graphs.nodes)
  graphs = graphs._replace(
    globals=graphs.globals + other_graphs.globals)
  graphs = graphs._replace(
    senders=graphs.senders.astype(bool),
    receivers=graphs.receivers.astype(bool))
  return graphs


def attention_logit_fn(
  edges: Array, 
  sender_attr: Array, 
  receiver_attr: Array, 
  global_edge_attributes: Array) -> Array:
  del edges
  x = jnp.concatenate([sender_attr, receiver_attr], axis=1)
  x = hk.Linear(1)(layernorm(x))
  return x


def attention_reduce_fn(edge_features, weights):
  return edge_features[0] * weights