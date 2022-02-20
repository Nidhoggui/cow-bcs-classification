import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from cv2 import cv2
import pandas as pd

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops


def analysis_superpixels(path_image, path_mask):
    image = cv2.imread(path_image)
    image_as_float = img_as_float(image)

    image_mask = cv2.imread(path_mask, 0)
    image_mask_reverse = np.zeros(image_mask.shape).astype("uint8")

    for y in range(image_mask.shape[0]):
        for x in range(image_mask.shape[1]):
            if image_mask[y][x] != 255:
                image_mask[y][x] = 0
                image_mask_reverse[y][x] = 255

    kernel = np.ones((5, 5), np.uint8)
    image_mask = cv2.erode(image_mask, kernel, iterations=20)
    image_mask_reverse = cv2.dilate(image_mask_reverse, kernel, iterations=20)

    segments_slic_mask = slic(
        image_as_float, n_segments=150, compactness=10, sigma=3, mask=image_mask_reverse)

    centroids = []
    means = []
    std = []

    regions = regionprops(segments_slic_mask)

    centroids.append((image.shape[1] // 2, image.shape[0] // 2))

    for props in regions:
        cy, cx = props.centroid
        centroids.append((cx, cy))

    for segment_value in np.unique(segments_slic_mask):
        mask = np.zeros(image.shape[0:2]).astype('uint8')
        mask[segments_slic_mask == segment_value] = 1

        image_chunk = image * mask[:, :, np.newaxis]

        image_chunk_mean_red = np.mean(image_chunk[mask == 1][:, 2])
        image_chunk_std_red = np.std(image_chunk[mask == 1][:, 2])
        image_chunk_mean_green = np.mean(image_chunk[mask == 1][:, 1])
        image_chunk_std_green = np.std(image_chunk[mask == 1][:, 1])
        image_chunk_mean_blue = np.mean(image_chunk[mask == 1][:, 0])
        image_chunk_std_blue = np.std(image_chunk[mask == 1][:, 0])

        chunk_mean_vector = (int(image_chunk_mean_red), int(image_chunk_mean_green),
                             int(image_chunk_mean_blue))
        chunk_std_vector = (int(image_chunk_std_red), int(image_chunk_std_green),
                            int(image_chunk_std_blue))

        means.append(chunk_mean_vector)
        std.append(chunk_std_vector)

    data = pd.DataFrame(centroids, columns=['x', 'y'])

    data['mean'] = means
    data['std'] = std

    print(data)

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(12)

    plt.imshow(
        mark_boundaries(img_as_float(cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)), segments_slic_mask, color=(255, 0, 0)),
        zorder=0)
    sc = ax.scatter(data['x'], data['y'], c="red", zorder=1)

    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        m = data.loc[sel.index]["mean"]
        s = data.loc[sel.index]["std"]
        sel.annotation.set(text=f"mean: {m}\nstd: {s}")
        # print(f"index: {sel.index}, coord: {sel.target}")

    plt.show()


if __name__ == "__main__":
    analysis_superpixels(f"C:\\Users\\usuario\\Projects\\cow-bcs-classification\\images\\region_growing_test\\vaca1.jpeg",
                         f"C:\\Users\\usuario\\Projects\\cow-bcs-classification\\images\\region_growing_test\\mask1.png")
