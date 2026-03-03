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


# =========================================================
# INTERNAL: Geo → Pixel
# =========================================================


def _geo_to_pixel(lat, lng, bounds, width, height):
    (lat_min, lng_min), (lat_max, lng_max) = bounds

    x = int((lng - lng_min) / (lng_max - lng_min) * width)
    y = int((lat_max - lat) / (lat_max - lat_min) * height)

    return x, y


# =========================================================
# INTERNAL: RGB → congestion intensity
# =========================================================


def _rgb_to_intensity(rgb, baseline=(150, 255, 200)):
    baseline = np.array(baseline)
    return np.linalg.norm(rgb - baseline)


# =========================================================
# MAIN: Add congestion weights to graph
# =========================================================


def add_congestion_to_graph(
    graph,
    raster_image,
    center_lat,
    center_lng,
    zoom,
    width,
    height,
    normalize=True,
):
    """
    Adds congestion weight [0,1] to graph edges.

    Parameters:
    - graph: networkx graph with WGS84 geometries
    - raster_image: PIL image or numpy array (RGBA)
    - center_lat, center_lng: center used to fetch raster
    - zoom: zoom level used to fetch raster
    - width, height: raster pixel size
    """

    if not isinstance(raster_image, np.ndarray):
        raster = np.array(raster_image)
    else:
        raster = raster_image

    bounds = _calculate_bounds(center_lat, center_lng, zoom, width, height)

    h, w = raster.shape[:2]

    G = graph.copy()
    raw_values = []

    for u, v, key, data in G.edges(keys=True, data=True):

        geom = data.get("geometry")

        if geom is None or not isinstance(geom, LineString):
            G[u][v][key]["congestion"] = 0
            continue

        coords = list(geom.coords)
        intensities = []

        for lon, lat in coords:
            x, y = _geo_to_pixel(lat, lon, bounds, w, h)

            if 0 <= x < w and 0 <= y < h:
                pixel = raster[y, x]
                rgb = pixel[:3]
                alpha = pixel[3]

                if alpha > 0:
                    intensities.append(_rgb_to_intensity(rgb))

        mean_intensity = float(np.mean(intensities)) if intensities else 0.0

        G[u][v][key]["raw_congestion"] = mean_intensity
        raw_values.append(mean_intensity)

    # Normalize to [0,1]
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

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

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
