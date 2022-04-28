from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

import os


def configure_image_display(images):
    fig, ax = plt.subplots(1, 5, figsize=(14, 4))
    fig.set_figwidth(14)
    fig.set_figheight(4)

    ax[0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    ax[0].set_title('IMAGE')

    ax[1].imshow(images[1], cmap='gray')
    ax[1].set_title('KERNEL (3, 3)')

    ax[2].imshow(images[2], cmap='gray')
    ax[2].set_title('KERNEL (5, 5)')

    ax[3].imshow(images[3], cmap='gray')
    ax[3].set_title("KERNEL (7, 7)")

    ax[4].imshow(images[4], cmap='gray')
    ax[4].set_title("KERNEL (9, 9)")


def erode(image, kernel_size=(5, 5), iterations=2):
    kernel = np.ones(kernel_size, np.uint8)
    erode_image = cv2.erode(image, kernel, iterations=iterations)

    return erode_image


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


def test_kernels_size(image, kernel_size):
    erode_image = erode(image, kernel_size=kernel_size)
    subtract_image = cv2.subtract(image, erode_image)
    top_back_shape = get_top_back_shape(subtract_image)

    return top_back_shape


def main(images):
    for image in images:
        _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
        blur_image = cv2.blur(thresh, (3, 3))
        blur_image[blur_image != 255] = 0

        top_back_shape_3_3 = test_kernels_size(blur_image, kernel_size=(3, 3))
        top_back_shape_5_5 = test_kernels_size(blur_image, kernel_size=(5, 5))
        top_back_shape_7_7 = test_kernels_size(blur_image, kernel_size=(7, 7))
        top_back_shape_9_9 = test_kernels_size(blur_image, kernel_size=(9, 9))

        print(f"Hu Moments with erode kernel size (3, 3): {hu_moments_with_log_transformation(top_back_shape_3_3).flatten()}")
        print(f"Hu Moments with erode kernel size (5, 5): {hu_moments_with_log_transformation(top_back_shape_5_5).flatten()}")
        print(f"Hu Moments with erode kernel size (7, 7): {hu_moments_with_log_transformation(top_back_shape_7_7).flatten()}")
        print(f"Hu Moments with erode kernel size (9, 9): {hu_moments_with_log_transformation(top_back_shape_9_9).flatten()}")
        print()

        configure_image_display([image, top_back_shape_3_3, top_back_shape_5_5, top_back_shape_7_7, top_back_shape_9_9])
    plt.show()


if __name__ == "__main__":
    images_path = os.path.abspath('../../images/segmented_images/ECCs/ECC 3,0')
    cow_tail_image_1 = cv2.imread(images_path + "/result_vaca traseira 1(perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2 = cv2.imread(images_path + "/result_vaca traseira 5(quase_perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2[cow_tail_image_2 > 150] = 0  # remove the part of the sky that was left from the background

    main([cow_tail_image_1, cow_tail_image_2])
