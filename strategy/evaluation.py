import numpy as np
import pandas as pd


# =========================================================
# INTERNAL: distance calculation (meters)
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

        lat2, lon2 = G.nodes[node]["y"], G.nodes[node]["x"]

        for rsu in rsu_nodes:

            lat1, lon1 = G.nodes[rsu]["y"], G.nodes[rsu]["x"]

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

        lat2, lon2 = G.nodes[u]["y"], G.nodes[u]["x"]

        for rsu in rsu_nodes:

            lat1, lon1 = G.nodes[rsu]["y"], G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                covered_length += length
                break

    if total_length == 0:
        return 0

    return covered_length / total_length


# =========================================================
# AVERAGE DISTANCE TO RSU
# =========================================================


def avg_distance_to_rsu(G, rsu_nodes):

    distances = []

    for node in G.nodes():

        lat2, lon2 = G.nodes[node]["y"], G.nodes[node]["x"]

        best = float("inf")

        for rsu in rsu_nodes:

            lat1, lon1 = G.nodes[rsu]["y"], G.nodes[rsu]["x"]

            dist = _distance_meters(lat1, lon1, lat2, lon2)

            if dist < best:
                best = dist

        distances.append(best)

    return np.mean(distances)


# =========================================================
# REDUNDANT COVERAGE
# =========================================================


def redundancy_ratio(G, rsu_nodes, radius_m):

    redundant = 0

    for node in G.nodes():

        lat2, lon2 = G.nodes[node]["y"], G.nodes[node]["x"]

        count = 0

        for rsu in rsu_nodes:

            lat1, lon1 = G.nodes[rsu]["y"], G.nodes[rsu]["x"]

            if _distance_meters(lat1, lon1, lat2, lon2) <= radius_m:
                count += 1

        if count >= 2:
            redundant += 1

    return redundant / len(G.nodes())


# ---------------------------------------------------------
# Helper: pixel → geographic coordinate
# ---------------------------------------------------------


def _pixel_to_geo(x, y, bounds, width, height):

    (lat_min, lng_min), (lat_max, lng_max) = bounds

    lat = lat_max - (y / height) * (lat_max - lat_min)
    lng = lng_min + (x / width) * (lng_max - lng_min)

    return lat, lng


# ---------------------------------------------------------
# Helper: traffic intensity
# ---------------------------------------------------------


def _traffic_intensity(rgb, baseline=(150, 255, 200)):
    baseline = np.array(baseline)
    return np.linalg.norm(rgb - baseline)


# ---------------------------------------------------------
# TRAFFIC COVERAGE USING TRAFFIC MAP
# ---------------------------------------------------------


def traffic_map_coverage(
    G,
    rsu_nodes,
    raster_image,
    bounds,
    radius_m,
):

    raster = np.array(raster_image)

    height, width = raster.shape[:2]

    total_traffic = 0
    covered_traffic = 0

    for y in range(height):

        for x in range(width):

            rgb = raster[y, x, :3]
            alpha = raster[y, x, 3]

            if alpha == 0:
                continue

            intensity = _traffic_intensity(rgb)

            if intensity <= 0:
                continue

            total_traffic += intensity

            lat, lon = _pixel_to_geo(x, y, bounds, width, height)

            for rsu in rsu_nodes:

                lat1, lon1 = G.nodes[rsu]["y"], G.nodes[rsu]["x"]

                meter_per_deg_lat = 111320
                meter_per_deg_lon = 111320 * np.cos(np.radians(lat1))

                dx = (lon1 - lon) * meter_per_deg_lon
                dy = (lat1 - lat) * meter_per_deg_lat

                if np.sqrt(dx**2 + dy**2) <= radius_m:

                    covered_traffic += intensity
                    break

    if total_traffic == 0:
        return 0

    return covered_traffic / total_traffic


# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================


def evaluate_deployment(
    G,
    rsu_nodes,
    radius_m,
    raster_image=None,
    bounds=None,
):

    results = {}

    results["num_rsus"] = len(rsu_nodes)

    results["node_coverage"] = node_coverage_ratio(G, rsu_nodes, radius_m)

    results["road_coverage"] = road_coverage_ratio(G, rsu_nodes, radius_m)

    results["avg_distance_to_rsu"] = avg_distance_to_rsu(G, rsu_nodes)

    results["redundancy"] = redundancy_ratio(G, rsu_nodes, radius_m)

    if raster_image is not None and bounds is not None:

        results["traffic_coverage"] = traffic_map_coverage(
            G,
            rsu_nodes,
            raster_image,
            bounds,
            radius_m,
        )

    return results


# =========================================================
# TABLE BUILDER FOR MULTIPLE STRATEGIES
# =========================================================


def build_comparison_table(G, deployments, radius_m, raster_image=None, bounds=None):
    """
    deployments: dict
        {
            "strategy_name": [node_list],
            "strategy2": [node_list]
        }
    """

    rows = []

    for name, rsu_nodes in deployments.items():

        metrics = evaluate_deployment(G, rsu_nodes, radius_m, raster_image, bounds)

        metrics["strategy"] = name

        rows.append(metrics)

    df = pd.DataFrame(rows)

    return df
