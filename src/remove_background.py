import numpy as np
from cv2 import cv2


def remove_background(image, image_mask, iterations=5):
    # creating the mask
    mask = np.zeros(image_mask.shape[:2], np.uint8)
    for y in range(image_mask.shape[0]):
        for x in range(image_mask.shape[1]):
            if image_mask[y][x] == 255:
                mask[y][x] = cv2.GC_PR_FGD
            else:
                mask[y][x] = cv2.GC_BGD

    # create temporary arrays used by grabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # run grabCut
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

    # where is sure or likely background, set to 0, otherwise set to 1
    mask_2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')

    # multiply image with new mask to subtract background
    return image * mask_2[:, :, np.newaxis]
