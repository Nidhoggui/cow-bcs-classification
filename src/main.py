from cv2 import cv2, imread
import numpy as np
import os

from remove_background import remove_background

if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\pedro\workspace\cow-bcs-classification\images\test_cow.jpeg")
    image_mask = cv2.imread(r"C:\Users\pedro\workspace\cow-bcs-classification\images\test_mask.png", 0)

    image_no_bgd = remove_background(image, image_mask)

    cv2.imwrite(r"C:\Users\pedro\workspace\cow-bcs-classification\output\images_without_background\test_cow.jpeg", image_no_bgd)
