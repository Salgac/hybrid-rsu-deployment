import networkx as nx
import numpy as np


def _distance_meters(lat1, lon1, lat2, lon2):

    meter_per_deg_lat = 111320
    meter_per_deg_lon = 111320 * np.cos(np.radians((lat1 + lat2) / 2))

    dx = (lon1 - lon2) * meter_per_deg_lon
    dy = (lat1 - lat2) * meter_per_deg_lat

    return np.sqrt(dx**2 + dy**2)


def crop_graph_radius(G, center_lat, center_lng, radius_m):

    keep_nodes = []

    for node in G.nodes():

        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]

        dist = _distance_meters(center_lat, center_lng, lat, lon)

        if dist <= radius_m:
            keep_nodes.append(node)

    subgraph = G.subgraph(keep_nodes).copy()

    return subgraph
