import numpy as np
import pandas as pd
import networkx as nx


# ---------------------------------------------------------
# distance helper
# ---------------------------------------------------------


def _distance_m(lat1, lon1, lat2, lon2):

    meter_per_deg_lat = 111320
    meter_per_deg_lon = 111320 * np.cos(np.radians((lat1 + lat2) / 2))

    dx = (lon1 - lon2) * meter_per_deg_lon
    dy = (lat1 - lat2) * meter_per_deg_lat

    return np.sqrt(dx**2 + dy**2)


# ---------------------------------------------------------
# snap coordinate to nearest node
# ---------------------------------------------------------


def nearest_node(G, lat, lon):

    best_node = None
    best_dist = float("inf")

    for node in G.nodes():

        lat2 = G.nodes[node]["y"]
        lon2 = G.nodes[node]["x"]

        d = _distance_m(lat, lon, lat2, lon2)

        if d < best_dist:
            best_node = node
            best_dist = d

    return best_node


# ---------------------------------------------------------
# vehicle density on nodes
# ---------------------------------------------------------


def vehicle_node_density(G, vehicle_df):

    visit_counts = {}

    for _, row in vehicle_df.iterrows():

        lat = row["Latitude"]
        lon = row["Longitude"]

        node = nearest_node(G, lat, lon)

        visit_counts[node] = visit_counts.get(node, 0) + 1

    return visit_counts


# ---------------------------------------------------------
# node coverage
# ---------------------------------------------------------


def node_coverage(G, node, radius):

    lat = G.nodes[node]["y"]
    lon = G.nodes[node]["x"]

    covered = set()

    for n in G.nodes():

        lat2 = G.nodes[n]["y"]
        lon2 = G.nodes[n]["x"]

        if _distance_m(lat, lon, lat2, lon2) <= radius:
            covered.add(n)

    return covered


# ---------------------------------------------------------
# compute node demand from congestion
# ---------------------------------------------------------


def compute_node_demand(G):

    demand = {}

    for node in G.nodes():

        weights = [data.get("congestion", 0) for _, _, data in G.edges(node, data=True)]

        demand[node] = float(np.mean(weights)) if weights else 0

    nx.set_node_attributes(G, demand, "demand")

    return demand


# ---------------------------------------------------------
# Hybrid RSU deployment (node-based)
# ---------------------------------------------------------


def hybrid_rsu_deployment(
    G, vehicle_df, total_rsu_budget=20, srsu_radius=300, mrsu_radius=300, theta=0.2
):

    compute_node_demand(G)

    # candidate static nodes
    H = [n for n in G.nodes() if G.nodes[n]["demand"] >= theta]

    # vehicle density hotspots
    visit_counts = vehicle_node_density(G, vehicle_df)

    ranked_mobile_nodes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)

    ranked_mobile_nodes = [n for n, _ in ranked_mobile_nodes]

    S_static = []
    M_mobile = []

    coverage = set()

    while len(S_static) + len(M_mobile) < total_rsu_budget:

        # best static RSU
        best_s = None
        best_s_gain = 0
        best_s_cover = set()

        for v in H:

            if v in S_static:
                continue

            cover = node_coverage(G, v, srsu_radius)

            gain = len(cover - coverage)

            if gain > best_s_gain:
                best_s_gain = gain
                best_s = v
                best_s_cover = cover

        # best mobile RSU
        best_m = None
        best_m_gain = 0
        best_m_cover = set()

        for node in ranked_mobile_nodes:

            if node in M_mobile:
                continue

            cover = node_coverage(G, node, mrsu_radius)

            gain = len(cover - coverage)

            if gain > best_m_gain:
                best_m_gain = gain
                best_m = node
                best_m_cover = cover

        # choose better
        if best_s_gain >= best_m_gain:

            if best_s is None:
                break

            S_static.append(best_s)
            coverage |= best_s_cover

        else:

            if best_m is None:
                break

            M_mobile.append(best_m)
            coverage |= best_m_cover

    return S_static, M_mobile
