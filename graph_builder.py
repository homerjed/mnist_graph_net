import jax
import jax.random as jr
import jax.numpy as jnp
import jraph
import numpy as np
import matplotlib.pyplot as plt


def get_distances(X):
  nx = X.shape[0]
  return (X[:, None, :] - X[None, :, :])[jnp.tril_indices(nx, k=-1)]


def get_receivers_senders(nx, dists, connect_radius):
  '''connect nodes within `connect_radius` units'''
  senders,receivers = jnp.tril_indices(nx, k=-1)
  dists = dists[jnp.tril_indices(nx, k=-1)]
  mask = dists < connect_radius
  # pad dummy s,r with n_node
  senders = jnp.where(mask > 0, senders, nx)
  receivers = jnp.where(mask > 0, receivers, nx)
  dists = jnp.where(mask > 0, dists, 0.)
  return senders, receivers, dists


def l2norm_einsum(X, eps=1e-9):
  """calculaute eucl distance with einsum"""
  a_min_b = X[:, None, :] - X[None, :, :]
  norm_sq = jnp.einsum("ijk, ijk -> ij", a_min_b, a_min_b)
  return jnp.where(norm_sq < eps, 0., jnp.sqrt(norm_sq))


def get_r2(X):
  """calculate euclidean distance from positional information"""
  alldists = l2norm_einsum(X)
  return alldists 


def edge_builder(
  pos, 
  r_connect, 
  n_node=None, 
  invert_edges=True):
  """ Get edges information"""
    
  if n_node is not None:
    pos = pos[:n_node]
    
  else:
    n_node = pos.shape[0]

  # mask out halos with distances < connect_radius
  dists = get_r2(pos)
    
  _receivers, _senders, dists = get_receivers_senders(
    n_node, 
    dists, 
    connect_radius=r_connect)

  diff = pos[_senders] - pos[_receivers]

  row, col = _senders, _receivers
    
  # Distance
  dist = dists
   
  # Centroid of galaxy catalogue
  centroid = pos.mean(axis=0)
    
  # Unit vectors of node, neighbor and difference vector
  unitrow = (pos[row] - centroid) 
  unitrow /= jnp.linalg.norm((pos[row] - centroid), axis=1).reshape(-1,1)
  unitcol = (pos[col] - centroid) 
  unitcol /= jnp.linalg.norm((pos[col] - centroid), axis=1).reshape(-1,1)
    
  unitdiff = jnp.where((dist.reshape(-1,1) > 0.), diff/dist.reshape(-1,1), 1.)
    
  # Dot products between unit vectors
  cos1 = jnp.einsum('ij, ij -> i', unitrow, unitcol)
  cos2 = jnp.einsum('ij, ij -> i', unitrow, unitdiff)

  # mask out nans
  cos1 = jnp.where(dist == 0., 0., cos1)
  cos2 = jnp.where(dist == 0., 0., cos2)
    
  if invert_edges:
    # flip the distance
    dist = jnp.where((dist > 0.), 1. / (dist*r_connect*100.), dist)
    # sort edges from biggest to smallest
    idx = jnp.argsort(dist)[::-1]
    dist = jnp.sort(dist)[::-1]
    
  else:
    # Normalize distance by linking radius
    dist /= r_connect
    # pad with large dummy edge
    mask = (dist > 0.)
    fillval = 100.
    dist = jnp.where(mask < 1, fillval, dist)
     
    # sort edges from SMALLEST to BIGGEST
    idx = jnp.argsort(dist)
    dist = jnp.sort(dist) 
        
    # replace all dummy distances with zeros again
    dist = jnp.where(dist == fillval, 0., dist)

  cos1 = cos1[idx]
  cos2 = cos2[idx]

  _senders = _senders[idx]
  _receivers = _receivers[idx]

  edge_attr = jnp.concatenate(
    [dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)
   
  return edge_attr, jnp.array(_senders), jnp.array(_receivers)


def padded_graph_builder(
  points,           # cartesian coordinates of image pixels
  density,          # density of pixel node inherits from
  alpha,            # global context / parameters of graph
  pad_nodes_to,
  pad_edges_to,
  r_connect=None, 
  n_node=None):
  """ 
    If padding not done by jraph, how do we unpad for training?
     - Specify mask explicitly here?
  """
      
  nodes = jnp.zeros((pad_nodes_to, 1)) # pixel density
  edges = jnp.zeros((pad_edges_to, 3)) # dist, angle, angle
    
  senders = (jnp.ones((pad_edges_to), dtype=int) * pad_nodes_to).astype(int)
  receivers = (jnp.ones((pad_edges_to), dtype=int) * pad_nodes_to).astype(int)
    
  # reset dummy positions to zero after passing to pos
  catalog = jnp.where((catalog[:, 0] > 0.)[:, jnp.newaxis], catalog, 0.)
    
  node_attr = density
      
  # GET EDGE INFORMATION
  edge_attr, _s, _r = edge_builder(
    jnp.array(points), 
    r_connect=r_connect,
    invert_edges=False) 
    
  n_node = jnp.sum(node_attr[:, 0] > 0.)
  n_edge = jnp.sum(edge_attr[:, 0] > 0.)
    
  # edge information
  edges = edges.at[:pad_edges_to, :].set(edge_attr[:pad_edges_to, :])
  senders = senders.at[:pad_edges_to].set(_s[:pad_edges_to])
  receivers = receivers.at[:pad_edges_to].set(_r[:pad_edges_to])

  # add in node information
  nodes = nodes.at[:pad_nodes_to, :].set(node_attr[:pad_nodes_to])

  graph = jraph.GraphsTuple(
    nodes=nodes, 
    edges=edges,
    senders=senders, 
    receivers=receivers,
    n_node=jnp.array([n_node]), 
    n_edge=jnp.array([n_edge]), 
    globals=jnp.array([alpha]))
  return graph


def make_graph_from_cloud(
  points,
  density, 
  label,
  r_link):

  edge_attr, senders, receivers = edge_builder(
    points, r_link)
  nodes = density

  n_node = jnp.array([nodes.shape[0]])
  n_edge = jnp.array([edge_attr.shape[0]])
  global_context = jnp.array([label])

  graph = jraph.GraphsTuple(
    nodes=nodes.reshape(nodes.shape[0], 1),
    n_node=n_node,
    globals=global_context,
    edges=edge_attr, # rotation/translation invariant edge features (else 'edges')
    senders=senders,
    receivers=receivers,
    n_edge=n_edge)
  return graph  


def image_to_point_cloud(
  img, 
  Npix=28,              # size of image 
  Nextra=0,             # number of extra points to add 
  noise_scale=0.05,     # perturbation to pixels, without this classification is easier
  subsample_factor=2,   # number of cloud points to subsample
  plot=True):
  """ 
  pixel coords --> x,y point.
  todo: calculate aspect ratio of number to add noise scaled in x,y accordingly
  """
  rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]]) # 90 deg rotation

  def pixel_to_coords(img, i, j, Nextra=Nextra):
    pixel_points = []

    # cartesian coordinates of pixel
    x = i #/ Npix * cloud_size
    y = j #/ Npix * cloud_size
    
    # perturb position of point so img doesn't look gridded
    point = np.array([x, y]) 
    pixel_point = point + np.random.normal(loc=0.0, scale=noise_scale, size=(2,))

    pixel_points.append(pixel_point)

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


if __name__ == "__main__":
  from get_mnist import load_mnist
  from mnist_to_graphs import image_to_point_cloud

  X, Y = load_mnist()
  
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
        #plot_graph(axs.ravel()[i], points, density, r_link=r_link)
        pass

    if plot:
      plt.tight_layout()
      plt.show()

    # return graphs, labels 
    dataset = [
      {"input_graph" : graph, "target" : label} 
      for graph, label in zip(graphs, labels)]
    return dataset 
    