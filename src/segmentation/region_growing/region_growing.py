from scipy.spatial import distance
from skimage.segmentation import find_boundaries
import numpy as np


def create_connected_superpixels_graph(segments_slic_mask):
    number_segments = len(np.unique(segments_slic_mask))
    graph_matrix = np.zeros((number_segments, number_segments))
    boundaries = find_boundaries(segments_slic_mask)

    for row in range(boundaries.shape[0]):
        for column in range(boundaries.shape[1]):
            if boundaries[row, column] == True:
                if row != 0 and row != boundaries.shape[0] - 1 and column != 0 and column != boundaries.shape[1] - 1:
                    if segments_slic_mask[row - 1, column] != segments_slic_mask[row + 1, column]:
                        graph_matrix[segments_slic_mask[row - 1, column],
                                     segments_slic_mask[row + 1, column]] = 1
                        graph_matrix[segments_slic_mask[row + 1, column],
                                     segments_slic_mask[row - 1, column]] = 1
                    if segments_slic_mask[row, column - 1] != segments_slic_mask[row, column + 1]:
                        graph_matrix[segments_slic_mask[row, column - 1],
                                     segments_slic_mask[row, column + 1]] = 1
                        graph_matrix[segments_slic_mask[row, column + 1],
                                     segments_slic_mask[row, column - 1]] = 1

                elif (row == 0 or row == boundaries.shape[0] - 1) and column != 0 and column != boundaries.shape[1] - 1:
                    if segments_slic_mask[row, column - 1] != segments_slic_mask[row, column + 1]:
                        graph_matrix[segments_slic_mask[row, column - 1],
                                     segments_slic_mask[row, column + 1]] = 1
                        graph_matrix[segments_slic_mask[row, column + 1],
                                     segments_slic_mask[row, column - 1]] = 1

                elif (column == 0 or column == boundaries.shape[1] - 1) and row != 0 and row != boundaries.shape[0] - 1:
                    if segments_slic_mask[row - 1, column] != segments_slic_mask[row + 1, column]:
                        graph_matrix[segments_slic_mask[row - 1, column],
                                     segments_slic_mask[row + 1, column]] = 1
                        graph_matrix[segments_slic_mask[row + 1, column],
                                     segments_slic_mask[row - 1, column]] = 1

    return graph_matrix


def get_initial_seed(segments_slic_mask, image_mask, image):
    segment_value_mask = 0
    for segment_value in np.unique(segments_slic_mask):
        mask = np.zeros(image.shape[0:2]).astype('uint8')
        mask[segments_slic_mask == segment_value] = 255

        if (mask == image_mask).all():
            segment_value_mask = segment_value

    return segment_value_mask


def region_growing_superpixels(gray_image, graph_matrix, segments_slic_mask, seeds, image_mask, c=1):
    superpixels_in_mask = []

    while len(seeds) > 0:
        for segment_value, vertice in enumerate(graph_matrix[seeds[0]]):
            if vertice == 1 and segment_value not in superpixels_in_mask:
                mask = np.zeros(gray_image.shape).astype('uint8')
                mask[segments_slic_mask == segment_value] = 1

                image_chunk = gray_image * mask
                image_chunk_mean = np.mean(image_chunk[np.where(mask == 1)])

                mean = np.mean(gray_image[np.where(image_mask == 255)])
                std = np.std(gray_image[np.where(image_mask == 255)])

                if mean - std * c <= image_chunk_mean <= mean + std * c:
                    image_mask[mask == 1] = 255
                    if segment_value not in seeds:
                        seeds.append(segment_value)
                    superpixels_in_mask.append(segment_value)

        seeds.pop(0)


def region_growing_superpixels_rgb(rgb_image, graph_matrix, segments_slic_mask, seeds, image_mask, c=1):
    superpixels_in_mask = []

    while len(seeds) > 0:
        for segment_value, vertice in enumerate(graph_matrix[seeds[0]]):
            if vertice == 1 and segment_value not in superpixels_in_mask:
                mask = np.zeros(rgb_image.shape[:2]).astype('uint8')
                mask[segments_slic_mask == segment_value] = 1

                image_chunk = rgb_image * mask[:, :, np.newaxis]
                image_chunk_mean_red = np.mean(image_chunk[mask == 1][:, 2])
                image_chunk_mean_green = np.mean(image_chunk[mask == 1][:, 1])
                image_chunk_mean_blue = np.mean(image_chunk[mask == 1][:, 0])

                mean_red = np.mean(rgb_image[image_mask == 255][:, 2])
                std_red = np.std(rgb_image[image_mask == 255][:, 2])
                mean_green = np.mean(rgb_image[image_mask == 255][:, 1])
                std_green = np.std(rgb_image[image_mask == 255][:, 1])
                mean_blue = np.mean(rgb_image[image_mask == 255][:, 0])
                std_blue = np.std(rgb_image[image_mask == 255][:, 0])

                if (mean_red - std_red * c <= image_chunk_mean_red <= mean_red + std_red * c) and (mean_green - std_green * c <= image_chunk_mean_green <= mean_green + std_green * c) and (mean_blue - std_blue * c <= image_chunk_mean_blue <= mean_blue + std_blue * c):
                    image_mask[mask == 1] = 255
                    if segment_value not in seeds:
                        seeds.append(segment_value)
                    superpixels_in_mask.append(segment_value)

        seeds.pop(0)


def region_growing_superpixels_ed_fixed(rgb_image, graph_matrix, segments_slic_mask, seeds, image_mask, c=1):
    superpixels_in_mask = []

    copy_image_mask = image_mask.copy()

    while len(seeds) > 0:
        for segment_value, vertice in enumerate(graph_matrix[seeds[0]]):
            if vertice == 1 and segment_value not in superpixels_in_mask:
                mask = np.zeros(rgb_image.shape[:2]).astype('uint8')
                mask[segments_slic_mask == segment_value] = 1

                image_chunk = rgb_image * mask[:, :, np.newaxis]

                image_chunk_mean_red = np.mean(image_chunk[mask == 1][:, 2])
                image_chunk_std_red = np.std(image_chunk[mask == 1][:, 2])
                image_chunk_mean_green = np.mean(image_chunk[mask == 1][:, 1])
                image_chunk_std_green = np.std(image_chunk[mask == 1][:, 1])
                image_chunk_mean_blue = np.mean(image_chunk[mask == 1][:, 0])
                image_chunk_std_blue = np.std(image_chunk[mask == 1][:, 0])

                chunk_mean_vector = (
                    image_chunk_mean_blue, image_chunk_mean_green, image_chunk_mean_red)
                chunk_std_vector = (image_chunk_std_blue,
                                    image_chunk_std_green, image_chunk_std_red)
                
                mean_red = np.mean(rgb_image[copy_image_mask == 255][:, 2])
                std_red = np.std(rgb_image[copy_image_mask == 255][:, 2])
                mean_green = np.mean(rgb_image[copy_image_mask == 255][:, 1])
                std_green = np.std(rgb_image[copy_image_mask == 255][:, 1])
                mean_blue = np.mean(rgb_image[copy_image_mask == 255][:, 0])
                std_blue = np.std(rgb_image[copy_image_mask == 255][:, 0])
                
                mean_vector = (mean_blue, mean_green, mean_red)
                std_vector = (std_blue, std_green, std_red)

                if distance.euclidean(chunk_mean_vector, mean_vector) <= distance.euclidean(chunk_std_vector, std_vector) * c:
                    copy_image_mask[mask == 1] = 255
                    if segment_value not in seeds:
                        seeds.append(segment_value)
                    superpixels_in_mask.append(segment_value)

        seeds.pop(0)

    return copy_image_mask

def region_growing_superpixels_ed_dinamic(rgb_image, graph_matrix, segments_slic_mask, seeds, image_mask, c=1):
    superpixels_in_mask = []

    copy_image_mask = image_mask.copy()

    mean_red = np.mean(rgb_image[copy_image_mask == 255][:, 2])
    std_red = np.std(rgb_image[copy_image_mask == 255][:, 2])
    mean_green = np.mean(rgb_image[copy_image_mask == 255][:, 1])
    std_green = np.std(rgb_image[copy_image_mask == 255][:, 1])
    mean_blue = np.mean(rgb_image[copy_image_mask == 255][:, 0])
    std_blue = np.std(rgb_image[copy_image_mask == 255][:, 0])
    while len(seeds) > 0:
        for segment_value, vertice in enumerate(graph_matrix[seeds[0]]):
            if vertice == 1 and segment_value not in superpixels_in_mask:
                mask = np.zeros(rgb_image.shape[:2]).astype('uint8')
                mask[segments_slic_mask == segment_value] = 1

                image_chunk = rgb_image * mask[:, :, np.newaxis]

                image_chunk_mean_red = np.mean(image_chunk[mask == 1][:, 2])
                image_chunk_std_red = np.std(image_chunk[mask == 1][:, 2])
                image_chunk_mean_green = np.mean(image_chunk[mask == 1][:, 1])
                image_chunk_std_green = np.std(image_chunk[mask == 1][:, 1])
                image_chunk_mean_blue = np.mean(image_chunk[mask == 1][:, 0])
                image_chunk_std_blue = np.std(image_chunk[mask == 1][:, 0])

                chunk_mean_vector = (
                    image_chunk_mean_blue, image_chunk_mean_green, image_chunk_mean_red)
                chunk_std_vector = (image_chunk_std_blue,
                                    image_chunk_std_green, image_chunk_std_red)

                mean_vector = (mean_blue, mean_green, mean_red)
                std_vector = (std_blue, std_green, std_red)

                if distance.euclidean(chunk_mean_vector, mean_vector) <= distance.euclidean(chunk_std_vector, std_vector) * c:
                    copy_image_mask[mask == 1] = 255
                    if segment_value not in seeds:
                        seeds.append(segment_value)
                    superpixels_in_mask.append(segment_value)

        seeds.pop(0)

    return copy_image_mask