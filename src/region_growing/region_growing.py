from cv2 import cv2
import numpy as np


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


def region_growing(grayscale_image, mask, delta):
    """
        Duvída sobre o código:
        - ele está certo? se sim, qual seria a propriedade P certa para o nosso caso? pois nesse caso estou usando um
        limiar delta, o que não dá muito certo pois ele muda de vaca para vaca e também estou usando a imagem em tons de
        cinza, acho que usar no RGB é melhor.
    """

    s = 0
    n = 0
    seeds = get_seeds(mask)

    # calculate the quantity of white pixels (the region to grow) and the sum of the grayscale
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 255:
                n += 1
                s += grayscale_image[y][x]

    while len(seeds) > 0:
        pixel = seeds[0]
        for coords in get_4_neighborhood(pixel[0], pixel[1], grayscale_image.shape):
            # if the pixel is not in the region of the mask and the property (P) are true then the pixel is added to the region
            # the property P: if the difference between the grayscale of the pixel and the grayscale mean of the region is less than delta
            if mask[coords[0], coords[1]] == 0 and abs(grayscale_image[coords[0], coords[1]] - s / n) <= delta:
                mask[coords[0], coords[1]] = 255
                seeds.append((coords[0], coords[1]))

                s += grayscale_image[coords[0], coords[1]]
                n += 1

        seeds.pop(0)
