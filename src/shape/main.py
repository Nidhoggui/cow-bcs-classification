from cv2 import cv2
import numpy as np

import os


def show_images(images):
    horizontal_images = np.concatenate(images, axis=1)

    cv2.imshow('HORIZONTAL', horizontal_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def erode(image, iterations=2):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def get_top_back_shape(image):
    image = image.copy()
    mean = np.mean(np.where(image != 0)[0])  # mean of the y coordinates of the board
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if y > mean:
                image[y][x] = 0

    return image


def main():
    images_path = os.path.abspath('../../images/segmented_images/ECCs/ECC 3,0/result_vaca traseira 1(perfeito_pasto).jpeg')
    cow_tail_image = cv2.imread(images_path, 0)

    blur_image = cv2.blur(cow_tail_image, (3, 3))
    erode_image = erode(blur_image, 4)
    erode_image[erode_image != 0] = 255
    subtract_image = cv2.subtract(blur_image, erode_image)

    top_back_shape = get_top_back_shape(subtract_image)

    show_images((cow_tail_image, blur_image, subtract_image, top_back_shape))


if __name__ == "__main__":
    main()
