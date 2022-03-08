from cv2 import cv2
import numpy as np

from skimage.segmentation import slic
from skimage.util import img_as_float

from region_growing import *
from analysis_superpixels import analysis_superpixels
from merge_criterion import *

import os


def configure_mask_image(mask_img):
    # set all pixel to 0 or 255, because the border pixels of the mask has other values
    for y in range(mask_img.shape[0]):
        for x in range(mask_img.shape[1]):
            if mask_img[y][x] != 255:
                mask_img[y][x] = 0

    # decrease the size of the mask to stay within the outline of the cow
    kernel = np.ones((5, 5), np.uint8)
    mask_img = cv2.erode(mask_img, kernel, iterations=20)

    return mask_img

if __name__ == "__main__":
    images_path = r"C:\\Users\\pedro\\workspace\\cow-bcs-classification\\images\\region_growing_test\\"
    image = cv2.imread(images_path + "vaca1.jpeg")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_image = cv2.imread(images_path + "mask1.png", 0)
    mask_image = configure_mask_image(mask_image)

    segments_slic = slic(img_as_float(image), n_segments=150, compactness=20, sigma=1, start_label = 0)
    seeds = get_initial_seed(segments_slic, mask_image, image)
    graph_matrix = create_connected_superpixels_graph(segments_slic)

    new_seeds = generate_new_mask(image, seeds, segments_slic)
    borders = border_superpixels(graph_matrix, segments_slic, seeds)
    borders_mask = generate_new_mask(image, borders, segments_slic)

    growed_mask = region_growing_superpixels(
        image, graph_matrix, segments_slic, seeds, borders.tolist(), algebric_rgb, c=1.3)
    growed_mask[new_seeds == 255] = 255
    growed_mask = np.where((growed_mask == 255), 1, 0).astype("uint8")

    final_image = image * growed_mask[:, :, np.newaxis]
    analysis_superpixels(image, mask_image, final_image,
                         segments_slic, growed_mask, new_seeds, borders_mask)