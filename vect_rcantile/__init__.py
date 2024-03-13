import numpy as np
from numba import guvectorize, float64, int64

from vect_rcantile.constants import RE, R2D


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', nopython=True, target='parallel')
def truncate_lnglat(lngs, lats, out_lngs, out_lats):
    for i in range(lngs.shape[0]):
        # Truncate longitudes: directly modifying the output array
        if lngs[i] > 180.0:
            out_lngs[i] = 180.0
        elif lngs[i] < -180.0:
            out_lngs[i] = -180.0
        else:
            out_lngs[i] = lngs[i]

        # Truncate latitudes: similarly, directly modifying the output array
        if lats[i] > 90.0:
            out_lats[i] = 90.0
        elif lats[i] < -90.0:
            out_lats[i] = -90.0
        else:
            out_lats[i] = lats[i]


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', nopython=True, target='parallel')
def xy(lngs, lats, out_xs, out_ys):
    for i in range(lngs.shape[0]):
        out_xs[i] = RE * np.radians(lngs[i])

        if lats[i] <= -90:
            out_ys[i] = -np.inf
        elif lats[i] >= 90:
            out_ys[i] = np.inf
        else:
            out_ys[i] = RE * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lats[i]))))


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', nopython=True, target='parallel')
def lnglat(xs, ys, out_lngs, out_lats):
    for i in range(xs.shape[0]):
        out_lngs[i] = xs[i] * R2D / RE
        out_lats[i] = ((np.pi * 0.5) - 2.0 * np.arctan(np.exp(-ys[i] / RE))) * R2D


@guvectorize([(int64[:], int64[:], int64[:], float64[:], float64[:])], '(n),(n),(n)->(n),(n)', nopython=True,
             target='parallel')
def ul(xs, ys, zooms, out_lngs, out_lats):
    for i in range(zooms.shape[0]):
        Z2 = zooms[i] ** 2
        lon_deg_i = xs[i] / Z2 * 360.0 - 180.0
        lat_rad_i = np.arctan(np.sinh(np.pi * (1 - 2 * ys[i] / Z2)))
        lat_deg_i = np.degrees(lat_rad_i)
        out_lngs[i] = lon_deg_i
        out_lats[i] = lat_deg_i


@guvectorize([(int64[:], int64[:], int64[:], float64[:], float64[:], float64[:], float64[:])],
             '(n),(n),(n)->(n),(n),(n),(n)',
             nopython=True,
             target='parallel')
def bounds(xs, ys, zooms, ul_lons_deg, lr_lats_deg, lr_lons_deg, ul_lats_deg):
    for i in range(zooms.shape[0]):
        Z2 = zooms[i] ** 2
        ul_lons_deg[i] = xs[i] / Z2 * 360.0 - 180.0
        lr_lats_deg[i] = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (ys[i] + 1) / Z2))))
        lr_lons_deg[i] = (xs[i] + 1) / Z2 * 360.0 - 180.0
        ul_lats_deg[i] = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ys[i] / Z2))))