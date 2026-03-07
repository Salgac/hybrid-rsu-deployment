import numpy as np
import math
import networkx as nx
from shapely.geometry import LineString
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# =========================================================
# INTERNAL: Compute geographic bounds
# (Hardcoded Web Mercator approximation)
# =========================================================


def _calculate_bounds(center_lat, center_lng, zoom, width, height):
    scale_factor = 104300 / (2**zoom)

    lat_span = (height * scale_factor) / 111320
    lng_span = (width * scale_factor) / (
        111320 * abs(math.cos(math.radians(center_lat)))
    )

    bounds = [
        [center_lat - lat_span / 2, center_lng - lng_span / 2],
        [center_lat + lat_span / 2, center_lng + lng_span / 2],
    ]

    return bounds


def add_congestion_to_graph(
    graph,
    raster_image,
    center_lat,
    center_lng,
    zoom,
    normalize=True,
):
    """
    Adds congestion weight [0,1] to graph edges using traffic raster.
    Parameters
    ----------
    graph : networkx graph
        Graph with WGS84 geometries (LineString edges)
    raster_image : numpy array
        Traffic raster (congestion intensity values)
    center_lat, center_lng : float
        Center used to generate the raster
    zoom : int
        Map zoom level
    normalize : bool
        Whether to normalize congestion values
    """

    # Ensure numpy raster
    raster = np.array(raster_image)

    height, width = raster.shape
    # MapQuest tile size (original)
    MAP_WIDTH = 2304
    MAP_HEIGHT = 2304

    bounds = _calculate_bounds(center_lat, center_lng, zoom, MAP_WIDTH, MAP_HEIGHT)

    G = graph.copy()

    raw_values = []

    for u, v, key, data in G.edges(keys=True, data=True):

        geom = data.get("geometry")

        if geom is None or not isinstance(geom, LineString):
            G[u][v][key]["congestion"] = 0
            continue

        intensities = []

        # sample multiple points along the edge
        num_samples = max(8, int(geom.length / 15))

        for i in range(num_samples):

            point = geom.interpolate(i / (num_samples - 1), normalized=True)
            lon, lat = point.x, point.y

            # convert to original pixel coordinate
            x_full = (lon - bounds[0][1]) / (bounds[1][1] - bounds[0][1]) * MAP_WIDTH
            y_full = (bounds[1][0] - lat) / (bounds[1][0] - bounds[0][0]) * MAP_HEIGHT

            # map to downsampled raster
            x = int(x_full * width / MAP_WIDTH)
            y = int(y_full * height / MAP_HEIGHT)

            if 0 <= x < width and 0 <= y < height:
                value = raster[y, x]

                if value > 0:
                    intensities.append(value)

        mean_intensity = float(np.mean(intensities)) if intensities else 0.0

        G[u][v][key]["raw_congestion"] = mean_intensity
        raw_values.append(mean_intensity)

    # Normalize congestion weights
    if normalize and raw_values:

        max_val = max(raw_values)

        if max_val > 0:

            for u, v, key in G.edges(keys=True):

                raw = G[u][v][key].get("raw_congestion", 0.0)

                G[u][v][key]["congestion"] = raw / max_val

        else:

            for u, v, key in G.edges(keys=True):
                G[u][v][key]["congestion"] = 0.0

    else:

        for u, v, key in G.edges(keys=True):

            G[u][v][key]["congestion"] = G[u][v][key].get("raw_congestion", 0.0)

    return G


# =========================================================
# VISUALIZATION FUNCTION
# =========================================================


def visualize_weighted_graph(
    graph,
    center_lat,
    center_lng,
    zoom=14,
    weight_attr="congestion",
    colormap="YlOrRd",
):
    """
    Visualize weighted graph using folium.
    """

    m = folium.Map(
        location=[center_lat, center_lng], tiles="Cartodb Positron", zoom_start=zoom
    )

    cmap = cm.get_cmap(colormap)

    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            continue

        weight = data.get(weight_attr, 0.0)
        color = mcolors.to_hex(cmap(weight))

        coords = [(lat, lon) for lon, lat in geom.coords]

        folium.PolyLine(coords, color=color, weight=3, opacity=0.9).add_to(m)

    return m
