from cv2 import cv2
import numpy as np

from skimage.segmentation import slic
from skimage.util import img_as_float

from region_growing import *
from analysis_superpixels import analysis_superpixels
from merge_criterion import *

import os


def configure_mask_image(mask_img, mask_img_reverse):
    # set all pixel to 0 or 255, because the border pixels of the mask has other values
    for y in range(mask_img.shape[0]):
        for x in range(mask_img.shape[1]):
            if mask_img[y][x] != 255:
                mask_img[y][x] = 0

    for y in range(mask_img.shape[0]):
        for x in range(mask_img.shape[1]):
            if mask_img[y][x] != 255:
                mask_img_reverse[y][x] = 255

    # decrease the size of the mask to stay within the outline of the cow
    kernel = np.ones((5, 5), np.uint8)
    mask_img = cv2.erode(mask_img, kernel, iterations=20)
    mask_img_reverse = cv2.dilate(mask_img_reverse, kernel, iterations=20)

    return mask_img, mask_img_reverse


if __name__ == "__main__":
    images_path = "C:\\Users\\usuario\\Projects\\cow-bcs-classification\\images\\region_growing_test\\"
    image = cv2.imread(images_path + "vaca1.jpeg")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_image = cv2.imread(images_path + "mask1.png", 0)
    mask_image_reverse = np.zeros(mask_image.shape).astype("uint8")

    mask_image, mask_image_reverse = configure_mask_image(
        mask_image, mask_image_reverse)

    segments_slic_mask = slic(img_as_float(image),
                              n_segments=150, compactness=10, sigma=3, mask=mask_image_reverse)
    seeds = [get_initial_seed(segments_slic_mask, mask_image, image)]
    graph_matrix = create_connected_superpixels_graph(segments_slic_mask)

    # region_growing_superpixels_gray(grayscale_image, graph_matrix, segments_slic_mask, seeds, mask_image)
    mask = region_growing_superpixels(
        image, graph_matrix, segments_slic_mask, seeds, mask_image, euclidean_rgb, c=1.3)

    mask = np.where((mask == 255), 1, 0).astype("uint8")

    final_image = image * mask[:, :, np.newaxis]

    analysis_superpixels(image, mask_image, final_image,
                         segments_slic_mask, mask)
