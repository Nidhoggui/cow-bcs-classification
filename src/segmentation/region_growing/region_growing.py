from scipy.spatial import distance
from skimage.segmentation import find_boundaries
import numpy as np


def create_connected_superpixels_graph(segments_slic):
    number_segments = len(np.unique(segments_slic))
    graph_matrix = np.zeros((number_segments, number_segments))
    boundaries = find_boundaries(segments_slic)

    for row in range(boundaries.shape[0]):
        for column in range(boundaries.shape[1]):
            if boundaries[row, column] == True:
                if row != 0 and row != boundaries.shape[0] - 1 and column != 0 and column != boundaries.shape[1] - 1:
                    if segments_slic[row - 1, column] != segments_slic[row + 1, column]:
                        graph_matrix[segments_slic[row - 1, column], segments_slic[row + 1, column]] = 1
                        graph_matrix[segments_slic[row + 1, column], segments_slic[row - 1, column]] = 1
                    if segments_slic[row, column - 1] != segments_slic[row, column + 1]:
                        graph_matrix[segments_slic[row, column - 1], segments_slic[row, column + 1]] = 1
                        graph_matrix[segments_slic[row, column + 1], segments_slic[row, column - 1]] = 1
                
                elif (row == 0 or row == boundaries.shape[0] - 1) and column != 0 and column != boundaries.shape[1] - 1:
                    if segments_slic[row, column - 1] != segments_slic[row, column + 1]:
                        graph_matrix[segments_slic[row, column - 1], segments_slic[row, column + 1]] = 1
                        graph_matrix[segments_slic[row, column + 1], segments_slic[row, column - 1]] = 1

                elif (column == 0 or column == boundaries.shape[1] - 1) and row != 0 and row != boundaries.shape[0] - 1:
                    if segments_slic[row - 1, column] != segments_slic[row + 1, column]:
                        graph_matrix[segments_slic[row - 1, column], segments_slic[row + 1, column]] = 1
                        graph_matrix[segments_slic[row + 1, column], segments_slic[row - 1, column]] = 1
    
    return graph_matrix


def get_initial_seed(segments_slic_mask, image_mask, image):
    segment_value_mask = []
    for segment_value in np.unique(segments_slic_mask):
        mask = np.zeros(image.shape[0:2]).astype('uint8')
        mask[segments_slic_mask == segment_value] = 255
        
        if any(image_mask[mask == 255] == 255):
            segment_value_mask.append(segment_value)
    return segment_value_mask

def generate_new_mask(image, segment_value, segments_slic_mask):
    mask = np.zeros(image.shape[0:2]).astype('uint8')
    for value in segment_value:
        chunk_mask = np.zeros(image.shape[0:2]).astype('uint8')
        chunk_mask[segments_slic_mask == value] = 255
        mask[chunk_mask == 255] = 255

    return mask

def border_superpixels(graph_matrix, segments_slic_mask, seeds):
    borders = []
    for chunk in (seeds):
      for segment_value, vertice in enumerate(graph_matrix[chunk]):
        if vertice == 1 and segment_value not in seeds:
          borders.append(chunk)
    return np.unique(borders)

def region_growing_superpixels(rgb_image, graph_matrix, segments_slic_mask, total_seeds, seeds, merge_criterion, c):
    superpixels_in_mask = total_seeds
    final_mask = np.zeros(rgb_image.shape[:2]).astype('uint8')
    count = np.zeros(len(np.unique(segments_slic_mask)))
    
    while len(seeds) > 0:
        current_chunk_mask = np.zeros(rgb_image.shape[:2]).astype('uint8')
        current_chunk_mask[segments_slic_mask == seeds[0]] = 1
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

                chunk_mean_vector = (image_chunk_mean_blue, image_chunk_mean_green, image_chunk_mean_red)
                chunk_std_vector = (image_chunk_std_blue, image_chunk_std_green, image_chunk_std_red)

                mean_red = np.mean(rgb_image[current_chunk_mask == 1][:, 2])
                std_red = np.std(rgb_image[current_chunk_mask == 1][:, 2])
                mean_green = np.mean(rgb_image[current_chunk_mask == 1][:, 1])
                std_green = np.std(rgb_image[current_chunk_mask == 1][:, 1])
                mean_blue = np.mean(rgb_image[current_chunk_mask == 1][:, 0])
                std_blue = np.std(rgb_image[current_chunk_mask == 1][:, 0])

                mean_vector = (mean_blue, mean_green, mean_red)
                std_vector = (std_blue, std_green, std_red)
                print(seeds[0], " check ",segment_value)
                if merge_criterion(mean_vector, std_vector, chunk_mean_vector, chunk_std_vector, c):
                    print(seeds[0], " add ",segment_value," - ", mean_vector + std_vector, chunk_mean_vector )
                    count[segment_value]+=1
                    if count[segment_value] > 1:
                        print(seeds[0], " confirm ",segment_value," - ", mean_vector + std_vector, chunk_mean_vector )
                        if segment_value not in seeds:
                            final_mask[mask == 1] = 255
                            seeds.append(segment_value)
                        superpixels_in_mask.append(segment_value)                    

        seeds.pop(0)
    print(count)
    return final_mask
