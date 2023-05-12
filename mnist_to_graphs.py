import jraph
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import KDTree

Graph = jraph.GraphsTuple

def image_to_point_cloud(
  img, 
  cloud_size=28.0,      # length of image in (x,y) space
  Npix=28,              # size of image 
  Nextra=0,             # number of extra points to add 
  noise_scale=0.05,     # perturbation to pixels, without this classification is easier
  subsample_factor=2,   # number of cloud points to subsample
  plot=True):
  """ 
  pixel coords --> x,y point.
  todo: calculate aspect ratio of number to add noise scaled in x,y accordingly
  """
  rotation = np.array([[0.0, -1.0], [1.0, 0.0]]) # 90 deg rotation

  def pixel_to_coords(img, i, j, Nextra=Nextra):
    pixel_points = []

    # cartesian coordinates of pixel
    x = i / cloud_size # Scale by size of image..,?
    y = j / cloud_size 
    
    # perturb position of point so img doesn't look gridded
    point = np.array([x, y]) 
    pixel_point = point + np.random.normal(loc=0.0, scale=noise_scale, size=(2,))

    pixel_points.append(pixel_point)

    # add some extra points for each pixel
    for i in range(Nextra):
      pixel_points.append(
        # add a few noisy points around the given point
        point + np.random.normal(loc=0.0, scale=noise_scale, size=(2,)))
    return pixel_points

  points = []  # (x, y) coordinates 
  density = [] # pixel values
  for i in range(Npix):
    for j in range(Npix):
      # skip the empty pixels
      if img[i,j] == 0:
        continue
      else:
        # make the (x,y) points for each pixel
        points.append(pixel_to_coords(img, i, j))
        # copy the pixel value accordingly
        density.extend([img[i, j]] * (Nextra + 1))

  density = np.asarray(density)
  points = np.asarray(points).reshape(-1, 2)
  # Images natively rotated differently 
  points = np.matmul(points, rotation) 

  if plot and 0:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(4.,2.), dpi=200)
    print("POINTS", points.shape)
    ax[0].scatter(*points.T, c=density, s=5.0), ax[1].axis("off")
    ax[1].imshow(img), ax[0].axis("off")
    plt.show()

  # randomly choose 1 / subsample_factor points
  if subsample_factor is not None:
    ix = np.random.randint(0, points.shape[0], size=(int(points.shape[0] / 2)))
    points, density = points[ix], density[ix]
  return points, density


def plot_graph(ax, points, density, r_link):
  ax.scatter(
    *points.T, s=2., c=density, zorder=15, cmap="PiYG_r") # s=4.
  kd_tree = KDTree(points)
  pairs = kd_tree.query_pairs(r=r_link)
  for (i, j) in pairs:
    # Plot a line between each point according to 
    # the KDTree queried pairs of points
    ax.plot(
      [points[i, 0], points[j, 0]],
      [points[i, 1], points[j, 1]], 
      #"-k", linewidth=0.25, zorder=8)
      "-k", linewidth=0.5, zorder=8)
  #ax.set_xlim([-14., 14.]); ax.set_ylim([-14., 14.])
  ax.axis("off")


def make_graph_from_cloud_just_points(cloud, r_link):
  points, label = cloud
  # Build the edge indices:
  # Get neighbouring points within linking distance
  edge_index = KDTree(points).query_pairs(
    r=r_link, output_type="ndarray")
  
  # for jraph graphs tuple (undirected graph)
  reverse_pairs = np.zeros((edge_index.shape[0], 2))
  for i, pair in enumerate(edge_index):
    reverse_pairs[i] = pair[::-1]

  # terminology for jraph
  senders, receivers = edge_index.astype(int).T

  row, col = edge_index.T
  # get distance as an edge feature for graphs
  delta_distance = points[row] - points[col]

  # edges features are euclidean distances 
  edges = delta_distance 

  # Get translation / rotation invariant edge attributes
  num_pairs = edge_index.shape[0]
  # Get translational and rotational invariant features
  # Distance
  dist = np.linalg.norm(delta_distance, axis=1)
  # Centroid of galaxy catalogue
  centroid = points.mean(axis=0)
  # Unit vectors of node, neighbor and difference vector
  unitrow = (points[row] - centroid) 
  unitrow /= np.linalg.norm((points[row] - centroid), axis=1).reshape(-1, 1)
  unitcol = (points[col] - centroid) 
  unitcol /= np.linalg.norm((points[col] - centroid), axis=1).reshape(-1, 1)
  unitdiff = delta_distance / (dist.reshape(-1,1) + 1e-8)
  # Dot products between unit vectors
  cos1 = np.array([
    np.dot(unitrow[i,:].T, unitcol[i,:]) 
    for i in range(num_pairs)])
  cos2 = np.array([
    np.dot(unitrow[i,:].T, unitdiff[i,:]) 
    for i in range(num_pairs)])
  # Normalize distance by linking radius
  dist /= r_link

  nodes = jnp.ones((len(points),))

  # jraph graph stuff 
  # Concatenate to get all edge attributes
  edge_attr = np.concatenate([
    dist.reshape(-1,1), 
    cos1.reshape(-1,1), 
    cos2.reshape(-1,1)], axis=1)
  n_node = jnp.array([nodes.shape[0]])
  n_edge = jnp.array([edges.shape[0]])
  global_context = jnp.array([label])

  edge_attr = jnp.nan_to_num(edge_attr, nan=0.0)

  graph = jraph.GraphsTuple(
    nodes=nodes.reshape(nodes.shape[0], 1),
    n_node=n_node,
    globals=global_context,
    edges=edge_attr, # rotation/translation invariant edge features (else 'edges')
    senders=senders,
    receivers=receivers,
    n_edge=n_edge)
  return graph


def make_graph_from_cloud(cloud, r_link, plot=True):
  points, density, label = cloud
  # Build the edge indices:
  # Get neighbouring points within linking distance
  edge_index = KDTree(points).query_pairs(
    r=r_link, output_type="ndarray")
  
  # for jraph graphs tuple (undirected graph)
  reverse_pairs = np.zeros((edge_index.shape[0], 2))
  for i, pair in enumerate(edge_index):
    reverse_pairs[i] = pair[::-1]

  # terminology for jraph
  senders, receivers = edge_index.astype(int).T

  row, col = edge_index.T
  # get distance as an edge feature for graphs
  delta_distance = points[row] - points[col]

  # node features are the pixel densities
  nodes = density 

  # edges features are euclidean distances 
  edges = delta_distance 

  # Get translation / rotation invariant edge attributes
  num_pairs = edge_index.shape[0]
  # Get translational and rotational invariant features
  # Distance
  dist = np.linalg.norm(delta_distance, axis=1)
  # Centroid of galaxy catalogue
  centroid = points.mean(axis=0)
  # Unit vectors of node, neighbor and difference vector
  unitrow = (points[row] - centroid) 
  unitrow /= np.linalg.norm((points[row] - centroid), axis=1).reshape(-1,1)
  unitcol = (points[col] - centroid) 
  unitcol /= np.linalg.norm((points[col] - centroid), axis=1).reshape(-1,1)
  unitdiff = delta_distance / (dist.reshape(-1,1) + 1e-8)
  # Dot products between unit vectors
  cos1 = np.array([
    np.dot(unitrow[i,:].T, unitcol[i,:]) 
    for i in range(num_pairs)])
  cos2 = np.array([
    np.dot(unitrow[i,:].T, unitdiff[i,:]) 
    for i in range(num_pairs)])
  # Normalize distance by linking radius
  dist /= r_link

  # jraph graph stuff 
  # Concatenate to get all edge attributes
  edge_attr = np.concatenate([
    dist.reshape(-1,1), 
    cos1.reshape(-1,1), 
    cos2.reshape(-1,1)], axis=1)
  n_node = jnp.array([nodes.shape[0]])
  n_edge = jnp.array([edges.shape[0]])
  global_context = jnp.array([label])

  edge_attr = jnp.nan_to_num(edge_attr, nan=0.0)

  graph = jraph.GraphsTuple(
    nodes=nodes.reshape(nodes.shape[0], 1),
    n_node=n_node,
    globals=global_context,
    edges=edge_attr, # rotation/translation invariant edge features (else 'edges')
    senders=senders,
    receivers=receivers,
    n_edge=n_edge)
  return graph



def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y


def pad_graph_to_nearest_power_of_two(
  graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """
      Pads a batched `GraphsTuple` to the nearest power of two.
      For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
      would pad the `GraphsTuple` nodes and edges:
        7 nodes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)
      And since padding is accomplished using `jraph.pad_with_graphs`, an extra
      graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs
      Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
      Returns:
        A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

  padded_graphs = jraph.pad_with_graphs(
      graphs_tuple, 
      pad_nodes_to, 
      pad_edges_to,
      pad_graphs_to)

  return padded_graphs


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

def get_mnist_graphs(
  images, 
  labels, 
  r_link, 
  n_graphs=100, 
  plot=False):

  # Plotting all graphs together
  if plot:
    n_plot = min(int(np.sqrt(n_graphs)), 10)
    fig, axs = plt.subplots(
      n_plot, n_plot, figsize=(4., 4.), dpi=200)

  graphs = []
  for i in range(n_graphs):
    image, label = images[i], labels[i]
    image = image.squeeze()
    # Make a point cloud from the mnist image
    points, density = image_to_point_cloud(
      image, plot=plot, subsample_factor=2)
    # Make a graph from the point cloud
    graph = make_graph_from_cloud(
      [points, density, label], plot=plot, r_link=r_link)
    graphs.append(graph)

    if plot and i < int(n_plot ** 2):
      plot_graph(axs.ravel()[i], points, density, r_link=r_link)

  if plot:
    plt.tight_layout()
    plt.show()

  # return graphs, labels
  dataset = [
    {"input_graph" : graph, "target" : label} 
    for graph, label in zip(graphs, labels)]
  return dataset

def rotation_matrix(phi_deg):
  phi = np.radians(phi_deg)
  c, s = np.cos(phi), np.sin(phi)
  R = np.array(((c, -s), (s, c)))
  return R

def get_rotated_mnist_graphs(
  images, 
  labels, 
  r_link,  
  n_graphs=4, 
  plot=True):
  #angles = [90., 180., 270., 360.]
  angles = [0, 1, 2, 3] # multiples of 90 degrees

  graphs = []
  for i in range(n_graphs):
    image, label = images[i], labels[i]
    image = image.squeeze()
    # Plotting all graphs together
    if plot:
      n_plot = int(np.sqrt(n_graphs * len(angles)))
      fig, axs = plt.subplots(
        n_plot, n_plot, figsize=(6., 6.), dpi=200)
    for phi, ax in zip(angles, axs.ravel()):
    #   image = np.matmul(
    #     rotation_matrix(phi), image)

      # Not quite the same as applying a rotation matrix to the nodes...
      image = np.rot90(image, k=phi)
      # Make a point cloud from the mnist image
      points, density = image_to_point_cloud(
        image, 
        plot=plot, 
        subsample_factor=2)
      # Make a graph from the point cloud
      graph = make_graph_from_cloud(
        [points, density, label], 
        plot=plot, 
        r_link=r_link)
      graphs.append(graph)

      if plot:
        plot_graph(ax, points, density, r_link=r_link)

  if plot:
    plt.tight_layout()
    plt.show()

  # return graphs, labels
  dataset = [
    {"input_graph" : graph, "target" : label} 
    for graph, label in zip(
      graphs, [label for _ in angles])]
  return dataset


def get_mnist_padded_graphs(images, labels, n_graphs=100, plot=False):
    clouds = []
    for i in range(n_graphs):
        image, label = images[i], labels[i]
        image = image.squeeze()
        # Make a point cloud from the mnist image
        points, density = image_to_point_cloud(
            image, plot=plot, subsample_factor=2)
        # Make a graph from the point cloud
        clouds.append([points, density, label])
        
    graphs = []
    for i, cloud in enumerate(clouds):
        graph = make_graph_from_cloud(cloud, plot=plot)
        graphs.append(graph)
    # Need to pad graphs to one consistent length. 
    # Gather all clouds first then graph them, padding to the length of the longest cloud.
    # max_edges = max([_nearest_bigger_power_of_two(graph.n_edge) for graph in graphs])
    # max_nodes = max([_nearest_bigger_power_of_two(graph.n_node) for graph in graphs])
    # Pad each graph to maximum graph size in dataset
    # graphs = [jraph.pad_with_graphs(graph, max_nodes, max_edges, 1) for graph in graphs]

    # return graphs, labels
    dataset = [
        {"input_graph" : graph, "target" : label} 
        for graph, label in zip(graphs, labels)]
    return dataset


def print_graph(graph):
  def if_shape(obj):
    if obj is not None:
      return obj.shape  
  print((f"nodes: {if_shape(graph.nodes)} " 
         f"edges: {if_shape(graph.edges)} " 
         f"receivers: {if_shape(graph.receivers)} " 
         f"senders: {if_shape(graph.senders)} " 
         # globals is just scalar label for now 
         f"globals: {graph.globals} "  
         f"n_nodes: {graph.n_node} "
         f"n_edges: {graph.n_edge} "))