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


def run(image, kernel_size):
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    blur_image = cv2.blur(thresh, (3, 3))
    blur_image[blur_image != 255] = 0

    erode_image = erode(blur_image, kernel_size=kernel_size)
    subtract_image = cv2.subtract(blur_image, erode_image)
    top_back_shape = get_top_back_shape(subtract_image)

    return top_back_shape


def show_different_back_shapes(images):
    for image in images:
        top_back_shape_3_3 = run(image, kernel_size=(3, 3))
        top_back_shape_5_5 = run(image, kernel_size=(5, 5))
        top_back_shape_7_7 = run(image, kernel_size=(7, 7))
        top_back_shape_9_9 = run(image, kernel_size=(9, 9))

        print(f"Hu Moments with erode kernel size (3, 3): {hu_moments_with_log_transformation(top_back_shape_3_3).flatten()}")
        print(f"Hu Moments with erode kernel size (5, 5): {hu_moments_with_log_transformation(top_back_shape_5_5).flatten()}")
        print(f"Hu Moments with erode kernel size (7, 7): {hu_moments_with_log_transformation(top_back_shape_7_7).flatten()}")
        print(f"Hu Moments with erode kernel size (9, 9): {hu_moments_with_log_transformation(top_back_shape_9_9).flatten()}")
        print()

        configure_image_display([image, top_back_shape_3_3, top_back_shape_5_5, top_back_shape_7_7, top_back_shape_9_9])
    plt.show()


def find_the_center_pixel(image):
    mean_y = np.mean(np.where(image != 0)[0])
    mean_x = np.mean(np.where(image != 0)[1])
    return int(mean_x), int(mean_y)


def translate_shape_coords_to_origin(image):
    flipped_image = image[::-1, :]
    x_distance, y_distance = find_the_center_pixel(flipped_image)

    trans_y = np.where(flipped_image != 0)[0] - y_distance
    trans_x = np.where(flipped_image != 0)[1] - x_distance

    return trans_x, trans_y


def show_poly_fit(images):
    for image in images:
        top_back_shape = run(image, kernel_size=(3, 3))

        x, y = translate_shape_coords_to_origin(top_back_shape)

        poly = np.polyfit(x, y, deg=6)

        fig, ax = plt.subplots(1, 3, figsize=(14, 6))
        fig.set_figwidth(14)
        fig.set_figheight(6)

        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('image')

        ax[1].imshow(cv2.cvtColor(top_back_shape, cv2.COLOR_BGR2RGB))
        ax[1].set_title('contour')

        x_range = (top_back_shape.shape[1] - 1) // 2
        y_range = (top_back_shape.shape[0] - 1) // 2
        plt.xlim([-x_range, x_range])
        plt.ylim([-y_range, y_range])
        ax[2].plot(x, y, "o", markersize=2, color="orange")
        ax[2].plot(x, np.polyval(poly, x), "o", markersize=3)
        ax[2].set_title('curve')
    plt.show()


if __name__ == "__main__":
    images_path = os.path.abspath('../../images/segmented_images/ECCs/ECC 3,0')
    cow_tail_image_1 = cv2.imread(images_path + "/result_vaca traseira 1(perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2 = cv2.imread(images_path + "/result_vaca traseira 5(quase_perfeito_pasto).jpeg", cv2.IMREAD_GRAYSCALE)
    cow_tail_image_2[cow_tail_image_2 > 150] = 0  # remove the part of the sky that was left from the background

    # show different top back shapes according to the kernel size with respectively hu moments
    show_different_back_shapes([cow_tail_image_1, cow_tail_image_2])
    # show the top back pixels centered on the cartesian plane origin and their polynomial fit
    show_poly_fit([cow_tail_image_1, cow_tail_image_2])
