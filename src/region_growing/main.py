from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from region_growing import region_growing


def configure_image_mask(mask_img):
    # set all pixel to 0 or 255, because the border pixels of the mask has other values
    for y in range(mask_img.shape[0]):
        for x in range(mask_img.shape[1]):
            if mask_img[y][x] != 255:
                mask_img[y][x] = 0

    # decrease the size of the mask to stay within the outline of the cow
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(mask_img, kernel, iterations=20)


if __name__ == "__main__":
    mask_image = cv2.imread('../../images/test_mask.png', 0)
    mask_image = configure_image_mask(mask_image)

    image = cv2.imread('../../images/test_cow.jpeg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    region_growing(grayscale_image, mask_image, 30)
    mask = np.where((mask_image == 255), 1, 0).astype('uint8')

    final_image = image * mask[:, :, np.newaxis]

    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
