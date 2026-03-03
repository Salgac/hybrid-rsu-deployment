import folium


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
