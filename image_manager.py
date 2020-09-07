import rasterio
import rasterio.features
import numpy as np
from matplotlib import pyplot as plt
from rasterio import crs
from rasterio.enums import Resampling
import os
from rasterio.warp import reproject, Resampling
from utils import normalize

IMAGES_PATH = './data/train/image/'


class ImageManager():
    def __init__(self, path):
        self.image = rasterio.open(path)
        self.transform_images = None
        self.crs = self.image.crs
        self.height = self.image.height
        self.width = self.image.width
        self.count = self.image.count
        self.dtypes = self.image.dtypes
        self.transform = self.image.transform
        self.name = self.__get_image_name()

    def __get_image_name(self):
        image_name = self.image.name.split('/')[-1]
        image_name = image_name.split('.')[0]
        return image_name

    def crop_image(self, coords_to_crop):
        min_x = 999999999999
        min_y = 999999999999
        max_x = 0
        max_y = 0

        for coord in coords_to_crop:
            x, y = self.image.index(coord[0], coord[1])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        new_image = np.zeros(
            (self.count, max_x-min_x, max_y-min_y), dtype=self.dtypes[0])

        if self.count == 1:
            image = self.image.read(1)
            new_image[:, :] = np.array(image[min_x:max_x, min_y:max_y])
        elif self.count == 3:
            blue, green, red = self.image.read()
            new_image[0, :, :] = np.array(blue[min_x:max_x, min_y:max_y])
            new_image[1, :, :] = np.array(green[min_x:max_x, min_y:max_y])
            new_image[2, :, :] = np.array(red[min_x:max_x, min_y:max_y])
        elif self.count == 4:
            blue, green, red, nir = self.image.read()
            new_image[0, :, :] = np.array(blue[min_x:max_x, min_y:max_y])
            new_image[1, :, :] = np.array(green[min_x:max_x, min_y:max_y])
            new_image[2, :, :] = np.array(red[min_x:max_x, min_y:max_y])
            new_image[3, :, :] = np.array(nir[min_x:max_x, min_y:max_y])

        return new_image

    def calculate_transform_coordinates(self, new_system=None):
        coordinates = []

        # Read the dataset's valid data mask as a ndarray.
        mask = np.ones(
            (self.height, self.width), dtype='uint8') * 255

        # Extract feature shapes and values from the array.
        for geom, val in rasterio.features.shapes(
                mask, transform=self.transform):

            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rasterio.warp.transform_geom(
                self.crs, new_system, geom, precision=6)

            # Print GeoJSON shapes to stdout.
            coordinates = geom["coordinates"][0]

        return coordinates

    def get_image_as_array(self):
        return self.image.read() if self.count > 1 else self.image.read(1)

    def show_image_coordinates(self):
        if self.transform_images is None:
            print(
                'You can\'t do this, before you need to execute the function get_transform_coordinates.')
        else:
            for coordinate in self.transform_images:
                print('Longitude:', coordinate[0],
                      ' -------  Latitude:', coordinate[1])

    def write_rgb(self, path):
        image = self.image.read()

        with rasterio.open(path, 'w', driver='GTiff', height=image.shape[1],
                           width=image.shape[2], count=3, dtype=str(image.dtype)) as dst:
            dst.write(image[:3, :, :])

    def write_image(self, path, image_as_array=None, shape=None):
        if image_as_array is None:
            image = self.image.read()
            height = self.height
            width = self.width
            count = self.count
            dtype = self.dtypes[0]
        else:
            image = image_as_array
            height = image_as_array.shape[1]
            width = image_as_array.shape[2]
            count = image_as_array.shape[0]
            dtype = image_as_array.dtype
        if shape is not None:
            height = shape[0]
            width = shape[1]
        with rasterio.open(path, 'w', driver='GTiff', height=height,
                           width=width, count=count, dtype=str(dtype)) as dst:
            dst.write(image)

    def redim(self, image_as_array, shape=(1000, 1000)):
        self.write_image('temp', image_as_array, shape)

        image = rasterio.open('temp')

        ans_image = image.read(
            out_shape=(
                image.count,
                shape[0],
                shape[1]
            ),
            resampling=Resampling.nearest
        )

        self.write_image('temp', ans_image)
        image = rasterio.open('temp')
        os.remove('temp')

        self.image = image
        self.crs = image.crs
        self.height = image.height
        self.width = image.width
        self.count = image.count
        self.dtype = image.dtypes
        self.transform = image.transform

    def redim_and_crop_image(self, shape):
        image = self.crop_image(self.transform_images)
        self.redim(image, shape)

    def get_batches(self, batch_size=(200, 200), step=200):
        batch_list = []

        image = self.get_image_as_array()

        for i in range(0, self.height, step):
            begin_i = i if i + \
                batch_size[0] < self.height else self.height - batch_size[0]
            for j in range(0, self.width, step):
                begin_j = j if j + \
                    batch_size[1] < self.width else self.width - batch_size[1]
                batch = image[:, begin_i:begin_i+batch_size[0], begin_j:begin_j +
                              batch_size[1]] if self.count > 1 else image[begin_i:begin_i+batch_size[0], begin_j:begin_j+batch_size[1]]
                batch_list.append(batch)

        return batch_list

    def show(self):
        if self.count > 1:
            blue = self.image.read(1)
            green = self.image.read(2)
            red = self.image.read(3)

            blue_norm = normalize(blue)
            green_norm = normalize(green)
            red_norm = normalize(red)

            bgr = np.dstack((blue_norm, green_norm, red_norm))

            plt.imshow(bgr)

        else:
            plt.imshow(self.get_image_as_array())

    def __get_intercept_array(self, image_manager):

        image_coords = self.calculate_transform_coordinates(self.crs)
        another_image_coords = image_manager.calculate_transform_coordinates(
            self.crs)

        points = [
            [max(image_coords[0][0], another_image_coords[0][0]), min(
                image_coords[0][1], another_image_coords[0][1])],
            [max(image_coords[1][0], another_image_coords[1][0]), max(
                image_coords[1][1], another_image_coords[1][1])],
            [min(image_coords[2][0], another_image_coords[2][0]), max(
                image_coords[2][1], another_image_coords[2][1])],
            [min(image_coords[3][0], another_image_coords[3][0]), min(
                image_coords[3][1], another_image_coords[3][1])]
        ]

        return self.crop_image(points)

    def intercept(self, image_manager):
        self_intercept_array = self.__get_intercept_array(image_manager)
        another_image_intercept_array = image_manager.__get_intercept_array(
            self)

        shape = (min(self_intercept_array.shape[1], another_image_intercept_array.shape[1]), min(
            self_intercept_array.shape[2], another_image_intercept_array.shape[2]))

        self.redim(self_intercept_array, shape)
