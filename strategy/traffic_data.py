import os
import glob
import numpy as np
from PIL import Image
from datetime import datetime
import folium
from folium.raster_layers import ImageOverlay
import math


# =========================================================
# Parse timestamp from filename
# =========================================================


def _parse_timestamp(path):
    name = os.path.basename(path)
    ts = name.split("_")[0]
    return datetime.strptime(ts, "%Y%m%d%H%M")


# =========================================================
# Identify congestion pixels (ignore green)
# =========================================================


def _congestion_intensity(rgb):
    r, g, b = rgb

    # ignore free-flow green/cyan roads
    if g > r:
        return 0

    # congestion intensity
    return r - g


# =========================================================
# STREAMING aggregation
# =========================================================


def build_traffic_raster(folder, start_time, end_time, step=4):
    """
    Builds congestion raster using streaming approach.
    step reduces resolution to speed up computation.
    """

    paths = sorted(glob.glob(os.path.join(folder, "*.png")))

    paths = [p for p in paths if start_time <= _parse_timestamp(p) <= end_time]

    if not paths:
        raise ValueError("No traffic images in interval")

    print("Processing", len(paths), "images")

    # load first image to determine shape
    first = np.array(Image.open(paths[0]).convert("RGBA"))

    height, width = first.shape[:2]

    height //= step
    width //= step

    accumulator = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    for path in paths:

        img = np.array(Image.open(path).convert("RGBA"))

        for y in range(0, img.shape[0], step):
            for x in range(0, img.shape[1], step):

                rgb = img[y, x, :3]
                alpha = img[y, x, 3]

                if alpha == 0:
                    continue

                intensity = _congestion_intensity(rgb)

                if intensity <= 0:
                    continue

                iy = y // step
                ix = x // step

                accumulator[iy, ix] += intensity
                counts[iy, ix] += 1

    counts[counts == 0] = 1

    raster = accumulator / counts

    # normalize
    if raster.max() > 0:
        raster = raster / raster.max()

    return raster


# =========================================================
# Map bounds
# =========================================================


def calculate_bounds(lat, lng, zoom, width, height):

    scale = 104300 / (2**zoom)

    lat_span = (height * scale) / 111320
    lng_span = (width * scale) / (111320 * abs(math.cos(math.radians(lat))))

    return [
        [lat - lat_span / 2, lng - lng_span / 2],
        [lat + lat_span / 2, lng + lng_span / 2],
    ]


# =========================================================
# Visualization
# =========================================================


def visualize_traffic_raster(raster, center_lat, center_lng, zoom, width, height):

    bounds = calculate_bounds(center_lat, center_lng, zoom, width, height)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

    ImageOverlay(
        image=raster, bounds=bounds, opacity=0.7, colormap=lambda x: (1, 0, 0, x)
    ).add_to(m)

    return m
