import math


def get_4_neighborhood(y, x, shape):
    out = []

    # top
    if y - 1 < 0:
        out.append((0, x))
    else:
        out.append((y - 1, x))

    # left
    if x - 1 < 0:
        out.append((y, 0))
    else:
        out.append((y, x - 1))

    # right
    if x + 1 > shape[1] - 1:
        out.append((y, shape[1] - 1))
    else:
        out.append((y, x + 1))

    # bottom
    if y + 1 > shape[0] - 1:
        out.append((shape[0] - 1, x))
    else:
        out.append((y + 1, x))

    return out


def get_seeds(mask):
    seeds = []

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 255:
                seeds.append((y, x))

    return seeds


def get_initial_sum_and_amount(image, mask):
    s = 0
    n = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 255:
                n += 1
                s += image[y][x]

    return s, n


def get_initial_square_sum_deviations(image, mask, mean):
    square_sum_deviations = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 255:
                square_sum_deviations += (image[y][x] - mean) ** 2

    return square_sum_deviations


def region_growing(image, mask):
    s, n = get_initial_sum_and_amount(image, mask)
    square_sum_deviations = get_initial_square_sum_deviations(image, mask, s/n)

    seeds = get_seeds(mask)

    while len(seeds) > 0:
        p = seeds[0]
        for coord in get_4_neighborhood(p[0], p[1], image.shape):
            if mask[coord[0], coord[1]] == 0 and (s / n) - math.sqrt(square_sum_deviations / n) <= image[coord[0]][coord[1]] <= (s / n) + math.sqrt(square_sum_deviations / n):
                mask[coord[0], coord[1]] = 255
                seeds.append((coord[0], coord[1]))

                s += image[coord[0], coord[1]]
                n += 1
                square_sum_deviations += (image[coord[0]][coord[1]] - (s / n)) ** 2

        seeds.pop(0)
