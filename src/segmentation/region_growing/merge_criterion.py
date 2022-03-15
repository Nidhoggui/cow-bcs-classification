from scipy.spatial import distance
import numpy as np


def algebric_rgb(mean_vector, std_vector, chunk_mean_vector, chunk_std_vector, c):
    if (mean_vector[0] - std_vector[0] * c <= chunk_mean_vector[0] <= mean_vector[0] + std_vector[0] * c) and (mean_vector[1] - std_vector[1] * c <= chunk_mean_vector[1] <= mean_vector[1] + std_vector[1] * c) and (mean_vector[2] - std_vector[2] * c <= chunk_mean_vector[2] <= mean_vector[2] + std_vector[2] * c):
        return True
    else:
        return False


def vector_rgb(mean_vector, std_vector, chunk_mean_vector, chunk_std_vector, c):
    if distance.euclidean(mean_vector, chunk_mean_vector) <= np.linalg.norm(std_vector) * c:
        return True
    else:
        return False


def euclidean_rgb(mean_vector, std_vector, chunk_mean_vector, chunk_std_vector, c):
    #vector format= (255,255,255)
    if distance.euclidean(chunk_mean_vector, mean_vector) <= distance.euclidean(chunk_std_vector, std_vector) * c:
        return True
    else:
        return False
