import osmnx as ox
import networkx as nx


def build_road_graph(
    place_name: str,
    network_type: str = "drive",
    major_road_types=None,
):
    """
    Download OSM road network and return a graph filtered to major roads.

    Parameters
    ----------
    place_name : str
        Name of place for OSM query (e.g. "Bratislava, Slovakia")

    network_type : str
        OSMnx network type (default: "drive")

    major_road_types : list
        List of OSM highway types to keep.
        Default: motorway, trunk, primary, secondary, tertiary

    Returns
    -------
    G_major : networkx.MultiDiGraph
        Filtered graph

    nodes_major : GeoDataFrame
        Node geometries

    edges_major : GeoDataFrame
        Edge geometries
    """

    if major_road_types is None:
        major_road_types = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
        ]

    # Download full graph
    G = ox.graph_from_place(place_name, network_type=network_type)

    # Convert to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    # Filtering function
    def is_major(highway_value):
        if isinstance(highway_value, list):
            return any(h in major_road_types for h in highway_value)
        return highway_value in major_road_types

    # Filter edges
    edges_major = edges[edges["highway"].apply(is_major)]

    # Build edge-induced subgraph
    G_major = G.edge_subgraph(edges_major.index).copy()

    # Remove isolated nodes
    G_major.remove_nodes_from(list(nx.isolates(G_major)))

    # Convert filtered graph back to GeoDataFrames
    nodes_major, edges_major = ox.graph_to_gdfs(G_major)

    return G_major, nodes_major, edges_major


#####################

import folium


def visualize_graph(nodes_gdf, edges_gdf, zoom_start=13):
    """
    Create a folium map from nodes and edges GeoDataFrames.
    """

    center_lat = nodes_gdf["y"].mean()
    center_lon = nodes_gdf["x"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Draw edges
    for _, row in edges_gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="black", weight=2, opacity=0.8).add_to(m)

    # Draw nodes
    for _, row in nodes_gdf.iterrows():
        folium.CircleMarker(
            location=[row["y"], row["x"]], radius=3, color="red", fill=True
        ).add_to(m)

    return m
