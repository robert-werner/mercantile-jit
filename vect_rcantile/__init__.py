from numba import guvectorize, float64

@guvectorize([(float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n),(n)', target='parallel')
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