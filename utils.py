import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def image_local_max(index_i, index_j, image, window_size, threshold):
    relative_index = np.unravel_index(
        np.argmax(
            image[index_i:index_i+window_size, index_j:index_j+window_size]
        ), (window_size, window_size)
    )
    absolute_index = (index_i+relative_index[0], index_j+relative_index[1])
    if image[absolute_index[0], absolute_index[1]] < threshold:
        absolute_index = (0, 0)
    return absolute_index


def list_max_value(image, step_size, window_size, threshold):
    max_list = []
    for i in range(0, image.shape[0]-window_size, step_size):
        for j in range(0, image.shape[1]-window_size, step_size):
            max_list.append(image_local_max(
                i, j, image, window_size, threshold))
    return max_list


def matrix_form_list(items_list, dimensions, value=1):
    matrix = np.zeros(dimensions)
    for point in items_list:
        matrix[point[0], point[1]] = value
    return matrix


def get_clusters(matrix, min_distance, i_limit, j_limit):
    clusters = defaultdict(list)
    current_cluster = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == -1:
                current_cluster += 1
                matrix[i, j] = current_cluster
                clusters[current_cluster].append((i, j))
            if matrix[i, j] > 0:
                for k in range(i, min(i+min_distance, i_limit)):
                    for l in range(j, min(j+min_distance, j_limit)):
                        if matrix[k, l] == -1:
                            matrix[k, l] = current_cluster
                            clusters[current_cluster].append((i, j))
    return clusters


def get_points_from_clusters(clusters):
    trees_pos = []
    for cluster in clusters.values():
        i_pos, j_pos = 0, 0
        for point in cluster:
            i_pos += point[0]
            j_pos += point[1]
        i_pos = int(i_pos/len(cluster))
        j_pos = int(j_pos/len(cluster))
        trees_pos.append((i_pos, j_pos))
    return trees_pos


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def in_center(batch, batch_size, step_size):
    batch_middle = int(batch_size/2 + 1)
    step_size_middle = int(step_size/2)
    for i in range(batch_middle-step_size_middle, batch_middle+step_size_middle+1):
        for j in range(batch_middle-step_size_middle, batch_middle+step_size_middle+1):
            if batch[i, j] != 0:
                return True

    return False


def is_tree(other_tags):
    if other_tags is None:
        return False

    tags = other_tags.split(',')
    for tag in tags:
        key, item = tag.split('=>')
        if item == '"tree"':
            return True
    return False


def get_trees_for_frames(frames):
    new_frames = zip(frames['other_tags'], frames['geometry'])
    return [frame[1] for frame in new_frames if is_tree(frame[0])]


def get_matrix_from_point(matrix, point, dim=(17, 17)):
    top = max(0, point[0] - int(dim[0]/2))
    left = max(0, point[1] - int(dim[1]/2))
    bottom = min(matrix.shape[1], point[0] + int(dim[0]/2 + 1))
    right = min(matrix.shape[2], point[1] + int(dim[1]/2 + 1))

    return matrix[:, top: bottom, left: right] if bottom-top > 0 and right - left > 0 else None


def search_point_in_list(point, points_list):
    for current_point in points_list:
        if current_point[0]-6 > point[0] and point[0] < current_point[0]+6 and\
                current_point[1]-6 > point[1] and point[1] < current_point[1]+6:
            return True
    return False
