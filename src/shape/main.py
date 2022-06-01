from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os


class CowPolynomialFit:

    def __init__(self):
        # constant parameters
        self.__kernel_size = (3, 3)
        self.__threshold = 20
        self.__polynomial_degree = 30

        self.__characteristic_bcs_info = {}
        self.__characteristic_bcs_images = {}

    def set_characteristic_bsc_images(self, bcs_images: dict):
        self.__characteristic_bcs_images = bcs_images

    def create_characteristic_polynomials(self):
        for bcs, image_path in self.__characteristic_bcs_images.items():
            cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial = self.__create_polynomial(image_path)
            self.__characteristic_bcs_info[bcs] = {
                "image": cow_image,
                "thresh": thresh,
                "top_back_shape": top_back_shape,
                "x": x,
                "y": y,
                "polynomial_coefficients": polynomial_coefficients,
                "polynomial": np.poly1d(polynomial_coefficients)
            }

    # QUESTION: When I use the polynomial coefficients to measure the MSE the it works, but doesn't make sense to me,
    # because we need to pass the y values to MSE function, so I think the correct is to create a range for the x values
    # and pass it to the polynomials, but when I do this, the results are worse than before.
    def predict(self, image_path):
        mse_scores = {}
        cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial = self.__create_polynomial(image_path)

        for bcs, info in self.__characteristic_bcs_info.items():
            mse_scores[bcs] = mean_squared_error(info["polynomial"], polynomial)

        print(mse_scores)

        return min(mse_scores, key=mse_scores.get)

    def show_characteristic_polynomials(self):
        for bcs, info in self.__characteristic_bcs_info.items():
            fig, ax = plt.subplots(1, 4, figsize=(14, 6))
            fig.set_figwidth(14)
            fig.set_figheight(6)

            ax[0].imshow(cv2.cvtColor(info["image"], cv2.COLOR_GRAY2RGB))
            ax[0].set_title(f"Cow BCS = {bcs}")

            ax[1].imshow(cv2.cvtColor(info["thresh"], cv2.COLOR_GRAY2RGB))
            ax[1].set_title("Thresh image")

            ax[2].imshow(cv2.cvtColor(info["top_back_shape"], cv2.COLOR_GRAY2RGB))
            ax[2].set_title("Contour")

            x_range = (info["top_back_shape"].shape[1] - 1) // 2
            y_range = (info["top_back_shape"].shape[0] - 1) // 2
            plt.xlim([-x_range, x_range])
            plt.ylim([-y_range, y_range])
            ax[3].plot(info["x"], info["y"], "o", markersize=2, color="orange")
            ax[3].plot(info["x"], np.polyval(info["polynomial_coefficients"], info["x"]), "o", markersize=3)
            ax[3].set_title(f"Polynomial degree = {self.__polynomial_degree}")
        plt.show()

    def __create_polynomial(self, image_path):
        cow_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(cow_image, self.__threshold, 255, cv2.THRESH_BINARY)

        blur_image = cv2.blur(thresh, self.__kernel_size)
        blur_image[blur_image != 255] = 0

        erode_image = self.__erode(blur_image, kernel_size=self.__kernel_size)
        subtract_image = cv2.subtract(blur_image, erode_image)
        top_back_shape = self.__get_top_back_shape(subtract_image)

        x, y = self.__translate_shape_coords_to_origin(top_back_shape)
        polynomial_coefficients = np.polyfit(x, y, deg=self.__polynomial_degree)
        polynomial = np.poly1d(polynomial_coefficients)

        return cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial

    def __erode(self, image, kernel_size=(5, 5), iterations=2):
        kernel = np.ones(kernel_size, np.uint8)
        erode_image = cv2.erode(image, kernel, iterations=iterations)

        return erode_image

    def __get_top_back_shape(self, image):
        image = image.copy()
        mean = np.mean(np.where(image != 0)[0])  # mean of the y coordinates of the board
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if y > mean:
                    image[y][x] = 0

        return image

    def __find_the_center_pixel(self, image):
        mean_y = np.mean(np.where(image != 0)[0])
        mean_x = np.mean(np.where(image != 0)[1])

        return int(mean_x), int(mean_y)

    def __translate_shape_coords_to_origin(self, image):
        flipped_image = image[::-1, :]
        x_distance, y_distance = self.__find_the_center_pixel(flipped_image)

        trans_y = np.where(flipped_image != 0)[0] - y_distance
        trans_x = np.where(flipped_image != 0)[1] - x_distance

        return trans_x, trans_y


if __name__ == "__main__":
    images_path = os.path.abspath('../../images/grabcut')

    cow_polynomial_fit = CowPolynomialFit()
    cow_polynomial_fit.set_characteristic_bsc_images({
        "2.75": images_path + "/ECC_2.75/grabcut_output(2).png",
        "3.0": images_path + "/ECC_3.0/grabcut_output(1).png",
        "4.0": images_path + "/ECC_4.0/grabcut_4.png"
    })
    cow_polynomial_fit.create_characteristic_polynomials()
    # cow_polynomial_fit.show_characteristic_polynomials()

    cow_test = images_path + "/ECC_4.0/grabcut_3.png"
    print(f"The BCS of the cow is probably: {cow_polynomial_fit.predict(cow_test)}")

