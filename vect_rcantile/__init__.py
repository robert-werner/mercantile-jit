import math

import numpy as np
from numba import guvectorize, float64, int64

from vect_rcantile.constants import RE, R2D, CE, EPSILON


@guvectorize([(float64, float64, float64[:], float64[:])],
             '(),()->(),()',
             nopython=True,
             target='parallel')
def truncate_lnglat(lng, lat, out_lng, out_lat):
    if lng > 180.0:
        out_lng[0] = 180.0
    elif lng < -180.0:
        out_lng[0] = -180.0
    else:
        out_lng[0] = lng

    # Truncate latitudes: similarly, directly modifying the output array
    if lat > 90.0:
        out_lat[0] = 90.0
    elif lat < -90.0:
        out_lat[0] = -90.0
    else:
        out_lat[0] = lat


@guvectorize([(float64, float64, float64[:], float64[:])],
             '(),()->(),()',
             nopython=True,
             target='parallel')
def xy(lng, lat, out_xs, out_ys):
    out_xs[0] = RE * np.radians(lng)
    if lat <= -90:
        out_ys[0] = -np.inf
    elif lat >= 90:
        out_ys[0] = np.inf
    else:
        out_ys[0] = RE * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lat))))


@guvectorize([(float64, float64, float64[:], float64[:])],
             '(),()->(),()',
             nopython=True,
             target='parallel')
def lnglat(x, y, out_lngs, out_lats):
    out_lngs[0] = x * R2D / RE
    out_lats[0] = ((np.pi * 0.5) - 2.0 * np.arctan(np.exp(-y / RE))) * R2D


@guvectorize([(int64, int64, int64, float64[:], float64[:])],
             '(),(),()->(),()',
             nopython=True,
             target='parallel')
def ul(x, y, zoom, out_lngs, out_lats):
    Z2 = 2 ** zoom
    lon_deg_i = x / Z2 * 360.0 - 180.0
    lat_rad_i = np.arctan(np.sinh(np.pi * (1 - 2 * y / Z2)))
    lat_deg_i = np.degrees(lat_rad_i)
    out_lngs[0] = lon_deg_i
    out_lats[0] = lat_deg_i


@guvectorize([(int64, int64, int64, float64[:], float64[:], float64[:], float64[:])],
             '(),(),()->(),(),(),()',
             nopython=True,
             target='parallel')
def bounds(x, y, zoom, ul_lons_deg, lr_lats_deg, lr_lons_deg, ul_lats_deg):
    Z2 = 2 ** zoom
    ul_lons_deg[0] = x / Z2 * 360.0 - 180.0
    lr_lats_deg[0] = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / Z2))))
    lr_lons_deg[0] = (x + 1) / Z2 * 360.0 - 180.0
    ul_lats_deg[0] = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / Z2))))


@guvectorize([(int64, int64, int64, float64[:], float64[:], float64[:], float64[:])],
             '(),(),()->(),(),(),()',
             nopython=True,
             target='parallel')
def xy_bounds(x, y, zoom, lefts, bottoms, rights, tops):
    tile_size = CE / 2 ** zoom
    lefts[0] = x * tile_size - CE / 2
    rights[0] = lefts[0] + tile_size
    tops[0] = CE / 2 - y * tile_size
    bottoms[0] = tops[0] - tile_size


@guvectorize([(float64, float64, float64[:], float64[:])],
             '(),()->(),()',
             nopython=True,
             target='parallel')
def _xy(lng, lat, xs, ys):
    xs[0] = lng / 360.0 + 0.5
    sinlat = np.sin(np.radians(lat))
    ys[0] = 0.5 - 0.25 * np.log((1.0 + sinlat) / (1.0 - sinlat)) / np.pi

