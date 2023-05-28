import numpy as np 


def get_R_z_string(R_values, z_values, n_moments_calculate=3):
    all_redshifts = [0., 0.5, 1., 2., 3.]
    all_R_values = ["5.0", "10.0", "15.0", "20.0", "25.0", "30.0"]
    resolution = 1024
    if isinstance(R_values, list):
        if len(R_values) == len(all_R_values):
            R_string = "all"
        else:
            R_string = "_".join([str(float(R)) for R in R_values])
    else:
        R_string = str(R_values)
    if isinstance(z_values, list):
        if len(z_values) == len(all_redshifts):
            z_string = "all"
        else:
            z_string = "_".join([str(float(z)) for z in z_values])
    else:
        z_string = str(z_values)

    return f"x={resolution}_R={R_string}_z={z_string}_nm={n_moments_calculate}"


def adjacency_matrix(sender_indices, receiver_indices, n_nodes=None):
    # Determine the number of nodes in the graph if not provided
    if n_nodes is None:
        n_nodes = max(max(sender_indices), max(receiver_indices)) + 1

    # Create an empty adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))

    # Set entries in the adjacency matrix for each edge
    for sender, receiver in zip(sender_indices, receiver_indices):
        adj_matrix[sender, receiver] = adj_matrix[receiver, sender] = 1 

    return adj_matrix


def print_graph_attributes(g):
    attrs = ["nodes", "edges", "senders", "receivers", "globals", "n_node", "n_edge"]

    for a in attrs:
        val = getattr(g, a)
        print(a, val.shape if val is not None else None)