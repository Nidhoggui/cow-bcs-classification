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


def hu_moments_with_log_transformation(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    # log transformation
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

    return hu_moments


def main(images):
    for image in images:
        blur_image = cv2.blur(image, (3, 3))
        erode_image = erode(blur_image, 4)
        erode_image[erode_image != 0] = 255
        subtract_image = cv2.subtract(blur_image, erode_image)

        top_back_shape = get_top_back_shape(subtract_image)

        print(hu_moments_with_log_transformation(top_back_shape).flatten())

        show_images((image, blur_image, subtract_image, top_back_shape))


if __name__ == "__main__":
    images_path = os.path.abspath('../../images/segmented_images/ECCs/ECC 3,0')
    cow_tail_image_1 = cv2.imread(images_path + "/result_vaca traseira 1(perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2 = cv2.imread(images_path + "/result_vaca traseira 5(quase_perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2[cow_tail_image_2 > 150] = 0  # remove the part of the sky that was left from the background

    main([cow_tail_image_1, cow_tail_image_2])
