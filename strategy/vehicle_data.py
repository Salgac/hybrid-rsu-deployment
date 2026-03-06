import pandas as pd
import folium
from folium.plugins import HeatMap
from datetime import datetime


# =========================================================
# Load vehicle data from CSV
# =========================================================


def load_vehicle_positions(csv_path, start_time, end_time):
    """
    Load vehicle positions within time interval.

    Parameters
    ----------
    csv_path : str
        Path to CSV file

    start_time : datetime

    end_time : datetime

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(csv_path)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)]

    return df


# =========================================================
# Create heatmap of vehicle positions
# =========================================================


def vehicle_heatmap(df, center_lat, center_lng, zoom=13):
    """
    Create a folium heatmap of vehicle density.
    """

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

    heat_points = df[["Latitude", "Longitude"]].values.tolist()

    HeatMap(heat_points, radius=8, blur=12, max_zoom=15).add_to(m)

    return m


# =========================================================
# Optional: create spatial demand grid
# =========================================================


def compute_vehicle_density(df, grid_size=0.001):
    """
    Aggregate vehicle counts into spatial grid cells.

    Returns
    -------
    pandas.DataFrame
    """

    df["lat_bin"] = (df["Latitude"] / grid_size).round() * grid_size
    df["lon_bin"] = (df["Longitude"] / grid_size).round() * grid_size

    density = (
        df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="vehicle_count")
    )

    return density
