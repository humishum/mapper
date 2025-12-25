"""utilities for coordinate transforms"""

import numpy as np
from typing import Tuple

# Earth radius in meters
EARTH_RADIUS_M = 6371000.0


#TODO: this only holds for flat earth approximation, so fine for now, but should be improved once we have more data/longer sequences 
# Implement https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
# https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84 
# and use ellipsoid model 

def transform_pointcloud_to_gps(
    points: np.ndarray, origin_lat: float, origin_lon: float, origin_alt: float
) -> np.ndarray:
    """
    transform from a local cartesian system to gps coordinates. 

    Args:
        points(np.ndarray): Nx3 array of [x, y, z] points in local coordinates
        origin_lat(float): Origin latitude in degrees
        origin_lon(float): Origin longitude in degrees
        origin_alt(float): Origin altitude in meters

    retuns:
        np.ndarray: Nx3 array of [lon, lat, alt] points in GPS coordinates
    """

    if points.shape[0] == 0:
        return np.array([]).reshape(0, 3)

    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dlat = y / EARTH_RADIUS_M
    dlon = x / (EARTH_RADIUS_M * np.cos(origin_lat_rad))

    lat = np.degrees(origin_lat_rad + dlat)
    lon = np.degrees(origin_lon_rad + dlon)
    alt = origin_alt + z

    # return as [lon, lat, alt] for deck.gl compatibility
    return np.column_stack([lon, lat, alt])
