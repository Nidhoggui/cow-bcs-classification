import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from cv2 import cv2
import pandas as pd

from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops


def analysis_superpixels(image, initial_mask, final_image, segments_slic_mask, final_mask):
    centroids = []
    means = []
    std = []
    belong = []
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

        belong.append(any(final_mask[mask == 1] == 1))

    data = pd.DataFrame(centroids, columns=['x', 'y'])

    data['mean'] = means
    data['std'] = std
    data['belong'] = belong

    fig, ax = plt.subplots(1, 4, figsize=(16, 16),
                           sharex=True, sharey=True)
    fig.set_figwidth(8)
    fig.set_figheight(12)

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('IMAGE')

    ax[1].imshow(initial_mask, cmap='gray')
    ax[1].set_title('INITIAL MASK')

    ax[2].imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                                 segments_slic_mask, color=(0, 0, 255)))
    sc = ax[2].scatter(data['x'], data['y'], c=data['belong'], zorder=1, s=5)
    ax[2].set_title("SLIC")

    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        m = data.loc[sel.index]["mean"]
        s = data.loc[sel.index]["std"]
        c = data.loc[sel.index]["belong"]
        sel.annotation.set(text=f"mean: {m}\nstd: {s}\nbelong: {c}")

    ax[3].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    ax[3].set_title('IMAGE WITHOUT BACKGROUND')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
