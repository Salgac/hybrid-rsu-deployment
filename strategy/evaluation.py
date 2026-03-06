import numpy as np
import pandas as pd
import math


# =========================================================
# INTERNAL: distance helper
# =========================================================


def _distance_meters(lat1, lon1, lat2, lon2):

    meter_per_deg_lat = 111320
    meter_per_deg_lon = 111320 * np.cos(np.radians((lat1 + lat2) / 2))

    dx = (lon1 - lon2) * meter_per_deg_lon
    dy = (lat1 - lat2) * meter_per_deg_lat

    return np.sqrt(dx**2 + dy**2)


# =========================================================
# NODE COVERAGE
# =========================================================


def node_coverage_ratio(G, rsu_nodes, radius_m):

    covered = set()

    for node in G.nodes():

        lat2 = G.nodes[node]["y"]
        lon2 = G.nodes[node]["x"]

        for rsu in rsu_nodes:

            lat1 = G.nodes[rsu]["y"]
            lon1 = G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                covered.add(node)
                break

    return len(covered) / len(G.nodes())


# =========================================================
# ROAD COVERAGE
# =========================================================


def road_coverage_ratio(G, rsu_nodes, radius_m):

    total_length = 0
    covered_length = 0

    for u, v, data in G.edges(data=True):

        length = data.get("length", 0)
        total_length += length

        lat2 = G.nodes[u]["y"]
        lon2 = G.nodes[u]["x"]

        for rsu in rsu_nodes:

            lat1 = G.nodes[rsu]["y"]
            lon1 = G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                covered_length += length
                break

    if total_length == 0:
        return 0

    return covered_length / total_length


# =========================================================
# TRAFFIC COVERAGE (using weighted graph)
# =========================================================


def traffic_coverage_graph(G, rsu_nodes, radius_m):

    total_demand = 0
    covered_demand = 0

    for u, v, data in G.edges(data=True):

        weight = data.get("congestion", 0)
        length = data.get("length", 0)

        demand = weight * length
        total_demand += demand

        lat2 = G.nodes[u]["y"]
        lon2 = G.nodes[u]["x"]

        for rsu in rsu_nodes:

            lat1 = G.nodes[rsu]["y"]
            lon1 = G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                covered_demand += demand
                break

    if total_demand == 0:
        return 0

    return covered_demand / total_demand


# =========================================================
# AVERAGE DISTANCE TO NEAREST RSU
# =========================================================


def avg_distance_to_rsu(G, rsu_nodes):

    distances = []

    for node in G.nodes():

        lat2 = G.nodes[node]["y"]
        lon2 = G.nodes[node]["x"]

        best = float("inf")

        for rsu in rsu_nodes:

            lat1 = G.nodes[rsu]["y"]
            lon1 = G.nodes[rsu]["x"]

            dist = _distance_meters(lat1, lon1, lat2, lon2)

            if dist < best:
                best = dist

        distances.append(best)

    return np.mean(distances)


# =========================================================
# REDUNDANCY
# =========================================================


def redundancy_ratio(G, rsu_nodes, radius_m):

    redundant = 0

    for node in G.nodes():

        lat2 = G.nodes[node]["y"]
        lon2 = G.nodes[node]["x"]

        count = 0

        for rsu in rsu_nodes:

            lat1 = G.nodes[rsu]["y"]
            lon1 = G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                count += 1

        if count >= 2:
            redundant += 1

    return redundant / len(G.nodes())


# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================


def evaluate_deployment(G, rsu_nodes, radius_m):

    results = {}

    results["num_rsus"] = len(rsu_nodes)

    results["node_coverage"] = node_coverage_ratio(G, rsu_nodes, radius_m)

    results["road_coverage"] = road_coverage_ratio(G, rsu_nodes, radius_m)

    results["traffic_coverage"] = traffic_coverage_graph(G, rsu_nodes, radius_m)

    results["avg_distance_to_rsu"] = avg_distance_to_rsu(G, rsu_nodes)

    results["redundancy"] = redundancy_ratio(G, rsu_nodes, radius_m)

    return results


# =========================================================
# BUILD COMPARISON TABLE
# =========================================================


def build_comparison_table(G, deployments, radius_m):

    rows = []

    for strategy_name, rsu_nodes in deployments.items():

        metrics = evaluate_deployment(G, rsu_nodes, radius_m)

        metrics["strategy"] = strategy_name

        rows.append(metrics)

    df = pd.DataFrame(rows)

    cols = ["strategy"] + [c for c in df.columns if c != "strategy"]

    return df[cols]
