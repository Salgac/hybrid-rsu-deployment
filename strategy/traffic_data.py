import os
import glob
import math
from datetime import datetime
import numpy as np
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay


# ---------------------------------------------------------
# Helper: Parse timestamp from filename
# Expected format: YYYYMMDDhhmm_traffic.png
# ---------------------------------------------------------


def _parse_timestamp_from_filename(filename):
    basename = os.path.basename(filename)
    timestamp_str = basename.split("_")[0]
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M")


# ---------------------------------------------------------
# 1. Load traffic images within time interval
# ---------------------------------------------------------


def load_traffic_images_in_interval(
    folder,
    start_time: datetime,
    end_time: datetime,
):
    image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))

    selected_images = []

    for path in image_paths:
        try:
            timestamp = _parse_timestamp_from_filename(path)

            if start_time <= timestamp <= end_time:
                img = Image.open(path).convert("RGBA")
                selected_images.append(np.array(img))

        except Exception:
            continue

    if not selected_images:
        raise ValueError("No traffic images found in the specified time interval.")

    print(f"Loaded {len(selected_images)} traffic images in selected interval.")
    return selected_images


# ---------------------------------------------------------
# 2. Aggregate Images (Preserve Colors)
# ---------------------------------------------------------


def aggregate_traffic_images(images):
    stack = np.stack(images, axis=0)

    rgb = stack[:, :, :, :3]
    alpha = stack[:, :, :, 3]

    mask = alpha > 0

    count = np.maximum(mask.sum(axis=0), 1)
    summed = (rgb * mask[..., None]).sum(axis=0)

    avg_rgb = summed / count[..., None]
    avg_alpha = (mask.sum(axis=0) > 0).astype(np.uint8) * 255

    result = np.dstack([avg_rgb.astype(np.uint8), avg_alpha])

    return Image.fromarray(result, mode="RGBA")


# ---------------------------------------------------------
# 3. Compute geographic bounds
# ---------------------------------------------------------


def calculate_image_bounds(center_lat, center_lng, zoom, width, height):
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


# ---------------------------------------------------------
# 4. Build aggregated traffic raster
# ---------------------------------------------------------


def build_traffic_raster(
    folder,
    start_time: datetime,
    end_time: datetime,
):
    images = load_traffic_images_in_interval(folder, start_time, end_time)
    aggregated_image = aggregate_traffic_images(images)
    return aggregated_image


# ---------------------------------------------------------
# 5. Display function (Folium overlay)
# ---------------------------------------------------------


def visualize_traffic_raster(
    aggregated_image,
    center_lat,
    center_lng,
    zoom,
    width,
    height,
    opacity=0.85,
):
    bounds = calculate_image_bounds(center_lat, center_lng, zoom, width, height)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

    ImageOverlay(
        image=np.array(aggregated_image),
        bounds=bounds,
        opacity=opacity,
        interactive=False,
        cross_origin=False,
    ).add_to(m)

    return m
