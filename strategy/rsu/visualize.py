import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree


def visualize_rsu_deployment(graph, rsu_nodes, center_lat, center_lng, zoom=14):

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            continue

        coords = [(lat, lon) for lon, lat in geom.coords]
        folium.PolyLine(coords, color="gray", weight=2).add_to(m)

    for node in rsu_nodes:
        data = graph.nodes[node]
        lat = data.get("y")
        lon = data.get("x")

        folium.CircleMarker(
            location=[lat, lon], radius=8, color="blue", fill=True, fill_opacity=1
        ).add_to(m)

    return m


def compute_mobile_density(G, vehicles_df, radius_m):

    nodes = list(G.nodes())

    node_coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes])

    vehicle_coords = vehicles_df[["Latitude", "Longitude"]].values

    meter_per_deg = 111320

    node_coords_m = node_coords * meter_per_deg
    vehicle_coords_m = vehicle_coords * meter_per_deg

    tree = cKDTree(vehicle_coords_m)

    density = {}

    for i, node in enumerate(nodes):

        neighbors = tree.query_ball_point(node_coords_m[i], radius_m)

        density[node] = len(neighbors)

    return density


def visualize_hybrid_deployment(
    G,
    rsu_nodes,
    vehicles_df,
    center_lat,
    center_lng,
    vehicle_radius=300,
):

    m = folium.Map(location=[center_lat, center_lng], zoom_start=14)

    mobile_density = compute_mobile_density(G, vehicles_df, vehicle_radius)

    max_density = max(mobile_density.values()) or 1

    cmap = cm.get_cmap("Blues")

    # mobile RSU corridors
    for u, v, data in G.edges(data=True):

        geom = data.get("geometry")
        if geom is None:
            continue

        density = mobile_density.get(u, 0) / max_density

        if density < 0.2:
            continue

        color = mcolors.to_hex(cmap(density))

        coords = [(lat, lon) for lon, lat in geom.coords]

        folium.PolyLine(coords, color=color, weight=4, opacity=0.6).add_to(m)

    # static RSUs
    for node in rsu_nodes:

        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]

        folium.CircleMarker(
            location=[lat, lon], radius=8, color="red", fill=True, fill_opacity=1
        ).add_to(m)

        # optional coverage radius
        folium.Circle(
            location=[lat, lon],
            radius=vehicle_radius,
            color="red",
            fill=False,
            opacity=0.3,
        ).add_to(m)

    return m
