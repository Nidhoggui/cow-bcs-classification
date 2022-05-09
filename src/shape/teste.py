import cv2 
import os
images_path = os.path.abspath('../../images/perfect_segmentation')
cow_tail_image_1 = cv2.imread(images_path + "/result_#ECC-2.75(1).png", cv2.IMREAD_GRAYSCALE)
cow_tail_image_2 = cv2.imread(images_path + "/result_#ECC-3.5.png", cv2.IMREAD_GRAYSCALE)

print(type(cv2.imread(images_path)))
print(images_path)
#cv2.imshow('img',cow_tail_image_1)
