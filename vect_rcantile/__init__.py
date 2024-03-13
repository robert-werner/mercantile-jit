import numpy as np
from numba import guvectorize, float64

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


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', target='parallel')
def xy(lngs, lats, out_xs, out_ys):
    for i in range(lngs.shape[0]):
        out_xs[i] = RE * np.radians(lngs[i])

        if lats[i] <= -90:
            out_ys[i] = -np.inf
        elif lats[i] >= 90:
            out_ys[i] = np.inf
        else:
            out_ys[i] = RE * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lats[i]))))


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', target='parallel')
def lnglat(xs, ys, out_lngs, out_lats):
    for i in range(xs.shape[0]):
        out_lngs[i] = xs[i] * R2D / RE
        out_lats[i] = ((np.pi * 0.5) - 2.0 * np.arctan(np.exp(-ys[i] / RE))) * R2D
