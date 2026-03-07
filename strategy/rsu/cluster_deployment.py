import numpy as np
import networkx as nx
from sklearn.cluster import KMeans


# =========================================================
# helper: extract node coordinates
# =========================================================


def _get_node_coords(G):

    nodes = list(G.nodes())

    coords = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in nodes])

    return nodes, coords


# =========================================================
# cluster-head deployment
# =========================================================


def deploy_cluster_head_rsus(G, num_rsus):
    """
    Deploy RSUs using spatial clustering of the road network.
    Each cluster selects a representative intersection
    (cluster head) as RSU location.

    Parameters
    ----------
    G : networkx graph
    num_rsus : int

    Returns
    -------
    list of node IDs
    """

    nodes, coords = _get_node_coords(G)

    # K-means clustering of intersections
    kmeans = KMeans(n_clusters=num_rsus, random_state=42, n_init="auto")

    labels = kmeans.fit_predict(coords)

    centers = kmeans.cluster_centers_

    rsu_nodes = []

    # choose node closest to cluster center
    for i in range(num_rsus):

        cluster_nodes = np.where(labels == i)[0]

        if len(cluster_nodes) == 0:
            continue

        center = centers[i]

        best_node = None
        best_dist = float("inf")

        for idx in cluster_nodes:

            lat, lon = coords[idx]

            dist = np.linalg.norm(center - np.array([lat, lon]))

            if dist < best_dist:
                best_dist = dist
                best_node = nodes[idx]

        rsu_nodes.append(best_node)

    return rsu_nodes
