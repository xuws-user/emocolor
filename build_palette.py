import itertools
import math
import random


def LABtoXYZ(LAB):
    def f(n):
        return n ** 3 if n > 6 / 29 else 3 * ((6 / 29) ** 2) * (n - 4 / 29)

    L, a, b = LAB
    X = 95.047 * f((L + 16) / 116 + a / 500)
    Y = 100.000 * f((L + 16) / 116)
    Z = 108.883 * f((L + 16) / 116 - b / 200)
    return X, Y, Z


def XYZtoRGB(XYZ):
    def f(n):
        return n * 12.92 if n <= 0.0031308 else (n ** (1 / 2.4)) * 1.055 - 0.055

    X, Y, Z = [x / 100 for x in XYZ]
    R = f(3.2406 * X + -1.5372 * Y + -0.4986 * Z) * 255
    G = f(-0.9689 * X + 1.8758 * Y + 0.0415 * Z) * 255
    B = f(0.0557 * X + -0.2040 * Y + 1.0570 * Z) * 255
    return R, G, B


def distance(color_a, color_b):
    return (sum([(a - b) ** 2 for a, b in zip(color_a, color_b)])) ** 0.5


def simple_bins(bins, size=16):
    level = 256 // size
    temp = {}
    for x in itertools.product(range(size), repeat=3):
        temp[x] = {'size': 0, 'sum': [0, 0, 0]}

    for color, count in bins.items():
        index = tuple([c // level for c in color])
        for i in range(3):
            temp[index]['sum'][i] += color[i] * count
        temp[index]['size'] += count

    result = {}
    for color in temp.values():
        if color['size'] != 0:
            result[tuple([color['sum'][j] / color['size'] for j in range(3)])] = color['size']

    return result


def k_means(bins, means, k, maxiter=1000, black=True):
    # init
    record = {}
    for color in bins.keys():
        record[color] = -1

    if black:
        means.append((0, 128, 128))

    for _ in range(maxiter):
        done = True
        cluster_sum = [[0, 0, 0] for _ in range(len(means))]
        cluster_size = [0 for _ in range(len(means))]

        # assign
        for color, count in bins.items():
            dists = [distance(color, mean) for mean in means]
            cluster = dists.index(min(dists))

            if record[color] != cluster:
                record[color] = cluster
                done = False

            for i in range(3):
                cluster_sum[cluster][i] += color[i] * count
            cluster_size[cluster] += count

        # update
        for i in range(k):
            if cluster_size[i] > 0:
                means[i] = tuple([cluster_sum[i][j] / cluster_size[i] for j in range(3)])

        if done:
            break

    return means[:k]


def init_means(bins, k):
    def attenuation(color, target):
        return 1 - math.exp(((distance(color, target) / 80) ** 2) * -1)

    # init
    colors = []
    for color, count in bins.items():
        colors.append([count, color])
    colors.sort(reverse=True)

    # select
    result = []
    for _ in range(k):
        for color in colors:
            if color[1] not in result:
                result.append(color[1])
                break

        for i in range(len(colors)):
            colors[i][0] *= attenuation(colors[i][1], result[-1])

        colors.sort(reverse=True)

    return result


def build_palette(image, k, random_init=False, black=True):
    # get colors
    colors = image.getcolors(image.width * image.height)
    # print('colors num:', len(colors))

    # build bins
    bins = {}
    for count, pixel in colors:
        bins[pixel] = count
    bins = simple_bins(bins)

    # init means
    if random_init:
        init = random.sample(list(bins), k)
    else:
        init = init_means(bins, k)

    # k-means
    means = k_means(bins, init, k, black=black)
    means.sort(reverse=True)
    colors = [tuple([int(x) for x in color]) for color in means]
    coloss = []
    coool = []
    inis = []
    inii = []
    for colors_i in range(0, len(colors)):
        coloss.append(
            XYZtoRGB(LABtoXYZ([colors[colors_i][0] / 255 * 100, colors[colors_i][1] - 128, colors[colors_i][2] - 128])))
    for init_i in range(0, len(init)):
        inis.append(XYZtoRGB(LABtoXYZ([init[init_i][0] / 255 * 100, init[init_i][1] - 128, init[init_i][2] - 128])))
    for colors_i in range(0, len(coloss)):
        coool.append([coloss[colors_i][0], coloss[colors_i][1], coloss[colors_i][2]])
    for inii_i in range(0, len(inis)):
        inii.append([inis[inii_i][0], inis[inii_i][1], inis[inii_i][2]])
    return colors, coool, inii
