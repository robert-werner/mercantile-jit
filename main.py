from morecantile.models import LL_EPSILON

from vect_rcantile import tile


def generate_tiles(w, s, e, n, zoom):
    tiles = []
    ul_tile = tile(w, n, zoom)
    lr_tile = tile(e - LL_EPSILON, s + LL_EPSILON, zoom)

    num_tiles = (lr_tile[0] - ul_tile[0] + 1) * (lr_tile[1] - ul_tile[1] + 1)
    index = 0
    while index < num_tiles:
        i = index % (lr_tile[0] - ul_tile[0] + 1) + ul_tile[0]
        j = index // (lr_tile[0] - ul_tile[0] + 1) + ul_tile[1]
        tiles.append((i, j, zoom))
        index += 1
    return tiles


if __name__ == '__main__':
    print(generate_tiles(37.5730769937248326, 55.4657993649133800, 37.9937437415740078, 55.8309614724213503, 20))
