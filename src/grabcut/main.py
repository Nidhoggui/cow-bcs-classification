from cv2 import cv2
import matplotlib.pyplot as plt

from interpolation import resize
from remove_background import remove_background

if __name__ == "__main__":
    mask_image = cv2.imread('../../images/test_mask.png', 0)
    image = cv2.imread('../../images/test_cow.jpeg')

    mask_image = resize(mask_image, image)

    image_no_bgd = remove_background(image, mask_image)

    plt.imshow(cv2.cvtColor(image_no_bgd, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
