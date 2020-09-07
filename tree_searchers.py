from utils import *
import numpy as np


class TreeSearcher():
    def __init__(self):
        self.LiDAR_image = None
        self.image = None

    def add_LiDAR_image(self, image):
        self.LiDAR_image = image

    def add_satelital_image(self, image):
        self.image = image


class TreeSearcherJustLiDAR(TreeSearcher):
    def __init__(self, step_size, windows_size, min_tree_distance):
        super().__init__()
        self.__step_size = step_size
        self.__windows_size = windows_size
        self.__min_tree_distance = min_tree_distance

    def get_trees_position(self, image_shape):
        ave = np.average(self.LiDAR_image.get_image_as_array()) + 6

        max_list = list_max_value(
            self.LiDAR_image.get_image_as_array(), self.__step_size, self.__windows_size, ave)
        print(len(max_list))
        matrix = matrix_form_list(max_list, image_shape, -1)
        clusters = get_clusters(
            matrix, self.__min_tree_distance, image_shape[0], image_shape[1])
        trees_pos = get_points_from_clusters(clusters)
        mask = matrix_form_list(trees_pos, image_shape)

        return mask


class TreeSearcherJustNir(TreeSearcher):
    def __init__(self, step_size, windows_size, min_tree_distance):
        super().__init__()
        self.__step_size = step_size
        self.__windows_size = windows_size
        self.__min_tree_distance = min_tree_distance

    def calculate_nvbm(self, nir, red):
        nir = int(nir)
        red = int(red)
        return (nir-red)/(nir+red)

    def get_trees_position(self, image_shape):
        image_as_array = self.image.get_image_as_array()
        nvim = np.zeros(image_shape)

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                nvim[i, j] = self.calculate_nvbm(
                    image_as_array[3, i, j], image_as_array[0, i, j])

        max_list = list_max_value(
            nvim, self.__step_size, self.__windows_size, 0.3)
        matrix = matrix_form_list(max_list, image_shape, -1)
        clusters = get_clusters(
            matrix, self.__min_tree_distance, image_shape[0], image_shape[1])
        trees_pos = get_points_from_clusters(clusters)
        mask = matrix_form_list(trees_pos, image_shape)

        return mask


class TreeSearcherMix(TreeSearcher):
    def __init__(self, step_size, windows_size, min_tree_distance, max_core_distance):
        super().__init__()
        self.__step_size = step_size
        self.__windows_size = windows_size
        self.__min_tree_distance = min_tree_distance
        self.__max_core_distance = max_core_distance

    def calculate_nvbm(self, nir, red):
        nir = int(nir)
        red = int(red)
        return (nir-red)/(nir+red)

    def valid_nvim(self, value, index_i, index_j, array):
        for i in range(max(0, index_i-self.__max_core_distance), min(array.shape[0], index_i+self.__max_core_distance)):
            for j in range(max(0, index_j-self.__max_core_distance), min(array.shape[1], index_j+self.__max_core_distance)):
                if array[i, j] > value:
                    return True

        return False

    def get_trees_position(self, image_shape):
        image_as_array = self.image.get_image_as_array()
        nvim = np.zeros(image_shape)

        ave = np.average(self.LiDAR_image.get_image_as_array()) + 6

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                nvim[i, j] = self.calculate_nvbm(
                    image_as_array[3, i, j], image_as_array[0, i, j])

        max_list = list_max_value(
            self.LiDAR_image.get_image_as_array(), self.__step_size, self.__windows_size, ave)

        new_max_list = []

        for item in max_list:
            if self.valid_nvim(0.45, item[0], item[1], nvim):
                new_max_list.append(item)

        print(len(new_max_list))

        matrix = matrix_form_list(new_max_list, image_shape, -1)
        clusters = get_clusters(
            matrix, self.__min_tree_distance, image_shape[0], image_shape[1])
        trees_pos = get_points_from_clusters(clusters)
        mask = matrix_form_list(trees_pos, image_shape)

        return mask
