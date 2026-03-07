from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


# =========================================================
# build KD-tree for graph nodes
# =========================================================


def build_node_kdtree(G):

    nodes = list(G.nodes())

    coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes])

    tree = cKDTree(coords)

    return nodes, coords, tree


# =========================================================
# nearest node using KD-tree
# =========================================================


def nearest_node_kdtree(nodes, tree, lat, lon):

    dist, idx = tree.query([lat, lon])

    return nodes[idx]


# =========================================================
# node coverage using KD-tree
# =========================================================


def node_coverage_kdtree(nodes, coords, tree, node_idx, radius_deg):

    center = coords[node_idx]

    idxs = tree.query_ball_point(center, radius_deg)

    return {nodes[i] for i in idxs}


# =========================================================
# convert meters → degrees (approx)
# =========================================================


def meters_to_degrees(radius_m):

    return radius_m / 111320.0


# =========================================================
# compute node demand from congestion
# =========================================================


def compute_node_demand(G):

    demand = {}

    for node in G.nodes():

        weights = [data.get("congestion", 0) for _, _, data in G.edges(node, data=True)]

        demand[node] = float(np.mean(weights)) if weights else 0

    return demand


# =========================================================
# vehicle density → mobile RSU candidates
# =========================================================


def vehicle_node_density(G, vehicle_df, nodes, tree):

    visits = {}

    for _, row in vehicle_df.iterrows():

        lat = row["Latitude"]
        lon = row["Longitude"]

        node = nearest_node_kdtree(nodes, tree, lat, lon)

        visits[node] = visits.get(node, 0) + 1

    ranked = sorted(visits.items(), key=lambda x: x[1], reverse=True)

    return [n for n, _ in ranked]


# =========================================================
# Hybrid RSU deployment (KD-tree optimized)
# =========================================================


def hybrid_rsu_deployment(
    G, vehicle_df, total_rsu_budget=20, srsu_radius=300, mrsu_radius=500, theta=0.2
):

    nodes, coords, tree = build_node_kdtree(G)

    node_index = {n: i for i, n in enumerate(nodes)}

    demand = compute_node_demand(G)

    H = [n for n in nodes if demand.get(n, 0) >= theta]

    ranked_mobile_nodes = vehicle_node_density(G, vehicle_df, nodes, tree)

    radius_s = meters_to_degrees(srsu_radius)
    radius_m = meters_to_degrees(mrsu_radius)

    S_static = []
    M_mobile = []

    coverage = set()

    while len(S_static) + len(M_mobile) < total_rsu_budget:

        best_s = None
        best_s_gain = 0
        best_s_cover = set()

        for v in H:

            if v in S_static:
                continue

            cover = node_coverage_kdtree(nodes, coords, tree, node_index[v], radius_s)

            gain = len(cover - coverage)

            if gain > best_s_gain:
                best_s_gain = gain
                best_s = v
                best_s_cover = cover

        best_m = None
        best_m_gain = 0
        best_m_cover = set()

        for node in ranked_mobile_nodes:

            if node in M_mobile:
                continue

            cover = node_coverage_kdtree(
                nodes, coords, tree, node_index[node], radius_m
            )

            gain = len(cover - coverage)

            if gain > best_m_gain:
                best_m_gain = gain
                best_m = node
                best_m_cover = cover

        static_ratio = len(S_static) / max(1, (len(S_static) + len(M_mobile)))
        min_static_ratio = 0.3
        if static_ratio < min_static_ratio:
            choose_static = True
        else:
            choose_static = best_s_gain >= best_m_gain

        if choose_static:
            S_static.append(best_s)
            coverage |= best_s_cover
        else:
            M_mobile.append(best_m)
            coverage |= best_m_cover

    # safeguard
    S_static = [n for n in S_static if n in G.nodes]
    M_mobile = [n for n in M_mobile if n in G.nodes]

    return S_static, M_mobile
