import numpy as np
from scipy.spatial import cKDTree


# =========================================================
# compute node traffic demand from weighted graph
# =========================================================


def _compute_node_demand(G):

    demand = {}

    for node in G.nodes():

        weights = [data.get("congestion", 0) for _, _, data in G.edges(node, data=True)]

        demand[node] = np.mean(weights) if weights else 0

    return demand


# =========================================================
# compute vehicle density using KD-tree
# =========================================================


def _vehicle_density_kdtree(G, vehicles_df, radius_m):

    nodes = list(G.nodes())

    node_coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes])

    vehicle_coords = vehicles_df[["Latitude", "Longitude"]].values

    # convert to meters approximation
    meter_per_deg = 111320

    node_coords_m = node_coords * meter_per_deg
    vehicle_coords_m = vehicle_coords * meter_per_deg

    tree = cKDTree(vehicle_coords_m)

    densities = {}

    for i, node in enumerate(nodes):

        neighbors = tree.query_ball_point(node_coords_m[i], radius_m)

        densities[node] = len(neighbors)

    return densities


# =========================================================
# greedy deployment using residual demand
# =========================================================


def deploy_hybrid_rsus(
    G,
    vehicles_df,
    num_rsus,
    coverage_radius_m=300,
):

    nodes = list(G.nodes())

    # traffic demand
    demand = _compute_node_demand(G)

    # vehicle density
    vehicle_density = _vehicle_density_kdtree(G, vehicles_df, coverage_radius_m)

    max_vehicle = max(vehicle_density.values()) or 1

    # residual demand (where mobile RSUs are insufficient)
    residual = {}

    for n in nodes:

        mobile_cov = vehicle_density[n] / max_vehicle

        residual[n] = demand[n] * (1 - mobile_cov)

    # greedy selection
    selected = []
    uncovered = set(nodes)

    for _ in range(num_rsus):

        best_node = None
        best_gain = -1

        for node in nodes:

            if node in selected:
                continue

            gain = 0

            lat1 = G.nodes[node]["y"]
            lon1 = G.nodes[node]["x"]

            for other in uncovered:

                lat2 = G.nodes[other]["y"]
                lon2 = G.nodes[other]["x"]

                dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111320

                if dist <= coverage_radius_m:
                    gain += residual[other]

            if gain > best_gain:
                best_gain = gain
                best_node = node

        selected.append(best_node)

        # remove covered nodes
        lat1 = G.nodes[best_node]["y"]
        lon1 = G.nodes[best_node]["x"]

        covered = set()

        for other in uncovered:

            lat2 = G.nodes[other]["y"]
            lon2 = G.nodes[other]["x"]

            dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111320

            if dist <= coverage_radius_m:
                covered.add(other)

        uncovered -= covered

    return selected
