import numpy as np
from numba import float64, njit, prange

from vect_rcantile import RE, R2D


@njit('float64[:](float64, float64)', parallel=True, fastmath=True)
def truncate_lnglat(lng, lat):
    temp = np.zeros((2), dtype=float64)
    if lng > 180.0:
        temp[0] = 180.0
    elif lng < -180.0:
        temp[0] = -180.0
    else:
        temp[0] = lng

    # Truncate latitudes: similarly, directly modifying the output array
    if lat > 90.0:
        temp[1] = 90.0
    elif lat < -90.0:
        temp[1] = -90.0
    else:
        temp[1] = lat
    return temp


@njit('float64[:](float64, float64, boolean)', parallel=True, fastmath=True)
def xy(lng, lat, truncate=False):
    temp = np.zeros((2), dtype=float64)
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)

    temp[0] = RE * np.radians(lng)

    if lat <= -90:
        temp[1] = -np.inf
    elif lat >= 90:
        temp[1] = np.inf
    else:
        temp[1] = RE * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lat))))

    return temp


@njit('float64[:](float64, float64, boolean)', parallel=True, fastmath=True)
def lnglat(x, y, truncate=False):
    temp = np.zeros((2), dtype=float64)
    temp[0], temp[1] = (
        x * R2D / RE,
        ((np.pi * 0.5) - 2.0 * np.arctan(np.exp(-y / RE))) * R2D,
    )
    if truncate:
        temp[0], temp[1] = truncate_lnglat(temp[0], temp[1])
    return temp
