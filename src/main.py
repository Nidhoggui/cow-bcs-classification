from cv2 import cv2, imread
import numpy as np
import os

from remove_background import remove_background
from interpolacao import redimensiona

if __name__ == "__main__":

    imagemMascara = cv2.imread('..\images\mask_vaca_traseira.png', 0)
    imagemBase = cv2.imread('..\images\test_cow.jpeg')

    image_mask = redimensiona(imagemMascara, imagemBase)
    image = cv2.imread('..\images\test_cow.jpeg')

    image_no_bgd = remove_background(image, image_mask)

    cv2.imwrite("..\output\images_without_background\test_cow.jpeg", image_no_bgd)
