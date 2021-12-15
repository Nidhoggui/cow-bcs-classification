from cv2 import cv2


def resize(mask_image, image):
    h, w, c = image.shape
    dimensions = (w, h)

    return cv2.resize(mask_image, dimensions, interpolation=cv2.INTER_LINEAR)
