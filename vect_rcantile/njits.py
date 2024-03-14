import numpy as np
from mercantile import ParentTileError, InvalidZoomError
from numba import float64, int64, njit, prange

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


from vect_rcantile.constants import LL_EPSILON, EPSILON


@njit('int64[:](float64,float64,int64)', parallel=True, fastmath=True)
def tile(lng, lat, zoom):
    result = np.zeros((3), dtype=int64)
    x = lng / 360.0 + 0.5
    y = 0.5 - 0.25 * np.log((1.0 + np.sin(np.radians(lat))) / (1.0 - np.sin(np.radians(lat)))) / np.pi
    Z2 = 2 ** zoom
    result[2] = zoom
    if x <= 0:
        result[0] = 0
    elif x >= 1:
        result[0] = np.int64(Z2 - 1)
    else:
        result[0] = np.int64(np.floor((x + EPSILON) * Z2))

    if y <= 0:
        result[1] = 0
    elif y >= 1:
        result[1] = np.int64(Z2 - 1)
    else:
        result[1] = np.int64(np.floor((y + EPSILON) * Z2))
    return result


@njit('int64[:,:](float64,float64,float64,float64,int64,boolean)', parallel=True, fastmath=True)
def generate_tiles(w, s, e, n, zoom, truncate=False):
    if truncate:
        w, s = truncate_lnglat(w, s)
        e, n = truncate_lnglat(e, n)

    ul_tile = tile(w, n, zoom)
    lr_tile = tile(e - LL_EPSILON, s + LL_EPSILON, zoom)

    num_tiles = (lr_tile[0] - ul_tile[0] + 1) * (lr_tile[1] - ul_tile[1] + 1)

    result = np.zeros((num_tiles, 3), dtype=np.int64)
    index = 0
    while index < num_tiles:
        result[index][0] = index % (lr_tile[0] - ul_tile[0] + 1) + ul_tile[0]
        result[index][1] = index // (lr_tile[0] - ul_tile[0] + 1) + ul_tile[1]
        result[index][2] = zoom
        index += 1
    return result
