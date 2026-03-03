import numpy as np
import networkx as nx


# =========================================================
# Compute node demand
# =========================================================


def compute_node_demand(graph, edge_weight_attr="congestion"):

    demand = {}

    for node in graph.nodes():
        weights = [
            data.get(edge_weight_attr, 0) for _, _, data in graph.edges(node, data=True)
        ]
        demand[node] = float(np.mean(weights)) if weights else 0.0

    nx.set_node_attributes(graph, demand, "demand")
    return graph


# =========================================================
# Precompute Euclidean coverage sets (FAST)
# =========================================================


def precompute_coverage_sets(graph, radius_meters):

    nodes = list(graph.nodes())
    coords = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in nodes])

    # Convert degrees to meters approximation
    # 1 degree lat ≈ 111320 m
    # 1 degree lon scaled by cos(lat)
    lat_mean = np.mean(coords[:, 1])
    meter_per_deg_lat = 111320
    meter_per_deg_lon = 111320 * np.cos(np.radians(lat_mean))

    # Convert to meters
    coords_m = np.zeros_like(coords)
    coords_m[:, 0] = coords[:, 0] * meter_per_deg_lon
    coords_m[:, 1] = coords[:, 1] * meter_per_deg_lat

    coverage_sets = {}

    for i, node in enumerate(nodes):
        center = coords_m[i]

        distances = np.linalg.norm(coords_m - center, axis=1)

        covered_indices = np.where(distances <= radius_meters)[0]
        covered_nodes = {nodes[j] for j in covered_indices}

        coverage_sets[node] = covered_nodes

    return coverage_sets


# =========================================================
# Greedy Static RSU Deployment
# =========================================================


def deploy_static_rsus(
    graph,
    num_rsus,
    coverage_radius_m=300,
    edge_weight_attr="congestion",
):

    G = compute_node_demand(graph, edge_weight_attr=edge_weight_attr)

    coverage_sets = precompute_coverage_sets(G, coverage_radius_m)

    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n].get("demand", 0),
        reverse=True,
    )
    nodes_sorted = nodes_sorted[:500]

    uncovered_nodes = set(G.nodes())
    selected_rsus = []

    for _ in range(num_rsus):

        best_node = None
        best_gain = -1

        for node in nodes_sorted:

            if node in selected_rsus:
                continue

            covered = coverage_sets[node]

            gain = sum(G.nodes[n]["demand"] for n in covered if n in uncovered_nodes)

            if gain > best_gain:
                best_gain = gain
                best_node = node
                best_covered = covered

        if best_node is None:
            break

        selected_rsus.append(best_node)
        uncovered_nodes -= best_covered

    return selected_rsus
