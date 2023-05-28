import jraph
import jax.numpy as jnp
import numpy as np 

Graph = jraph.GraphsTuple


def pad_graph_to_value(
  graphs_tuple: Graph, value: int) -> Graph:
  """ Pad graph to a fixed value instead of various powers of 2. """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = value + 1
  pad_edges_to = value
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  padded_graphs = jraph.pad_with_graphs(
    graphs_tuple, 
    pad_nodes_to, 
    pad_edges_to,
    pad_graphs_to)
  return padded_graphs


def make_neighbour_graph(data):
    """
        Build graph with edges connecting neighbouring 
        nodes (in pdf) only
    """
    # jnp.tile(jnp.array([-1, 0, 1]), n_node_per_graph)
    n_node_per_graph = np.prod(data.shape)
    node_features = data
    # No self connections
    x = jnp.eye(n_node_per_graph, k=-1) + jnp.eye(n_node_per_graph, k=+1)
    senders, receivers = jnp.argwhere(x > 0.).T
    n_edge_per_graph = max(senders)

    g = jraph.GraphsTuple(
        nodes=node_features.flatten(),
        edges=None,
        n_node=jnp.array([n_node_per_graph]),
        n_edge=jnp.array([n_edge_per_graph]),
        senders=senders,
        receivers=receivers,
        globals=jnp.array([n_node_per_graph, n_edge_per_graph], dtype=jnp.float32)[None, :])
    return g


def make_graph(data, redshift, Rs):
    """ 
        Make graph of data for one redshift
        and all smoothing scales.
    """
    n_node_per_graph = np.prod(data.shape)
    n_edge_per_graph = n_node_per_graph ** 2.
    globals = jnp.array([n_node_per_graph, n_edge_per_graph])
    globals = globals.astype(jnp.float32)[jnp.newaxis, :]

    n_z, n_R, n_m = data.shape
    _Rs = jnp.concatenate([Rs] * n_m)
    _zs = jnp.array([redshift] * n_R * n_m)
    node_feature_datas = [data.reshape(-1), _Rs, _zs]
    node_features = jnp.stack(node_feature_datas, axis=1)
    print("node_features:", node_features.shape)
    g = jraph.get_fully_connected_graph(
        node_features=node_features, 
        n_node_per_graph=n_node_per_graph,
        n_graph=1,
        global_features=globals,
        add_self_edges=False)
    return g


def make_subgraphs(moments, redshifts, Rs):
    # Vmapping, so this is z-axis
    all_g_z = []

    # Add scale information to nodes?
    # z_R_information = jnp.stack([R_values_float])

    for z, mz in zip(redshifts, moments):
        n_R, n_m = mz.shape
        # Rs = jnp.array(
        #     [float(R) for R in R_values] * n_m)
        _Rs = jnp.concatenate([Rs] * n_m)
        _zs = jnp.array([z] * n_R * n_m)
        # print(mz.shape, np.prod(mz.shape))
        # Only add redshifts to node features if using multi-redshifts
        n_node_per_graph = np.prod(mz.shape)
        node_features = jnp.stack([
            mz.flatten(), _Rs, _zs], axis=1)
        g_z = jraph.get_fully_connected_graph(
            # Concatenate redshift information to nodes
            node_features=node_features, 
            n_node_per_graph=n_node_per_graph,
            n_graph=1,
            #global_features=parameters, # maybe just parameters for main_graph not subgraphs?
            add_self_edges=True)
        all_g_z.append(g_z)
    return all_g_z 


def make_graph_from_subgraphs(subgraphs, parameters):
    """ 
        Make ONE graph from fully connected redshift-subgraphs
        - NO connections between subgraphs of different redshift
        - Initialise global property as total cardinalities of subgraphs over all z
            -> try cardinality vector with cardinalities of subgraphs
    """
    # For each redshift, shift its sender/receiver indices by the number of nodes in the 
    # redshift subgraph, so that the senders/receivers are still unique in the main graph
    # for all subgraphs
    receivers = jnp.concatenate([
        g_z.n_node * n + g_z.receivers 
        for n, g_z in enumerate(subgraphs)])
    senders = jnp.concatenate([
        g_z.n_node * n + g_z.senders 
        for n, g_z in enumerate(subgraphs)])
    nodes = jnp.concatenate([g_z.nodes for g_z in subgraphs])[:, None]
    n_node = sum([g_z.n_node for g_z in subgraphs])
    n_edge = sum([g_z.n_edge for g_z in subgraphs])
    cardinalities = jnp.array([n_node, n_edge], dtype=jnp.float32).T#[None, :]
    # cardinalities = jnp.array([len(nodes), len(senders)], dtype=jnp.float32).T#[None, :]
    # cardinalities = jnp.log10(cardinalities)
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=None, 
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=cardinalities)