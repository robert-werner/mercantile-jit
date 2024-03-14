import numpy as np
from numba import float64, njit, prange

from vect_rcantile import RE, R2D


@njit('float64[:](float64, float64)', parallel=True, fastmath=True)
def truncate_lnglat(lng, lat):
    result = np.zeros((2), dtype=float64)
    if lng > 180.0:
        result[0] = 180.0
    elif lng < -180.0:
        result[0] = -180.0
    else:
        result[0] = lng

    # Truncate latitudes: similarly, directly modifying the output array
    if lat > 90.0:
        result[1] = 90.0
    elif lat < -90.0:
        result[1] = -90.0
    else:
        result[1] = lat
    return result


@njit('float64[:](float64, float64, boolean)', parallel=True, fastmath=True)
def xy(lng, lat, truncate=False):
    result = np.zeros((2), dtype=float64)
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)

    result[0] = RE * np.radians(lng)

    if lat <= -90:
        result[1] = -np.inf
    elif lat >= 90:
        result[1] = np.inf
    else:
        result[1] = RE * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lat))))

    return result


@njit('float64[:](float64, float64, boolean)', parallel=True, fastmath=True)
def lnglat(x, y, truncate=False):
    result = np.zeros((2), dtype=float64)
    result[0], result[1] = (
        x * R2D / RE,
        ((np.pi * 0.5) - 2.0 * np.arctan(np.exp(-y / RE))) * R2D,
    )
    if truncate:
        result[0], result[1] = truncate_lnglat(result[0], result[1])
    return result


@njit('float64[:](int64, int64, int64)', parallel=True, fastmath=True)
def ul(x, y, zoom):
    result = np.zeros((2), dtype=float64)
    Z2 = 2 ** zoom
    lon_deg = x / Z2 * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / Z2)))
    lat_deg = np.degrees(lat_rad)
    result[0] = lon_deg
    result[1] = lat_deg
    return result
