import rasterio
from os import listdir
from os.path import join
from sys import exit

from shapely.geometry import point
from tensorflow.python.keras.backend import shape
from image_manager import ImageManager
from utils import in_center, get_trees_for_frames, get_matrix_from_point, search_point_in_list
import geopandas as gpd
from random import sample
import numpy as np
from rasterio.warp import transform


def generate_labels(raw_image_path, LiDAR_image_path, searcher):
    images_tuples = list(get_image_and_LiDAR(raw_image_path, LiDAR_image_path))
    # for image_tuple in images_tuples:
    image = ImageManager(images_tuples[0][0])
    temp_image = ImageManager(images_tuples[0][0])
    LiDAR_image = ImageManager(images_tuples[0][1])
    print('\n\n[PROCESSING IMAGE:', image.name, ']\n')

    print('[IMAGE GENERATION STARTED.....]')

    image.intercept(LiDAR_image)
    LiDAR_image.intercept(temp_image)

    image.show()
    image.write_rgb("data/train/image/" +
                    image.name + ".tif")

    print('[IMAGE GENERATION FINISHED]')
    print('\n[TREES DETECTION STARTED.....]')

    searcher.add_satelital_image(image)
    searcher.add_LiDAR_image(LiDAR_image)

    label_image = searcher.get_trees_position(
        (image.height, image.width))

    with rasterio.open("data/train/label/" + image.name + ".tif", 'w', driver='GTiff', height=label_image.shape[0],
                       width=label_image.shape[1], count=1, dtype=rasterio.uint8) as dst:
        dst.write((label_image*255).astype(rasterio.uint8), 1)

    image_array = image.get_image_as_array()

    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            if label_image[i, j] != 0:
                paint(image_array, i, j)

    with rasterio.open("data/train/trees_identified/" + image.name + ".tif", 'w', driver='GTiff', height=label_image.shape[0],
                       width=label_image.shape[1], count=3, dtype=rasterio.uint8) as dst:
        dst.write(image_array[:3, 40:, :])

    print('[TREES DETECTION FINISHED]')


def get_images(path):
    onlyfiles = [join(path, f) for f in listdir(path)]
    onlyfiles.sort()
    return onlyfiles


def get_image_and_LiDAR(raw_image_path, LiDAR_image_path):
    raw_images = get_images(raw_image_path)
    LiDAR_images = get_images(LiDAR_image_path)

    if len(raw_images) != len(LiDAR_images):
        print("Error raw images amount is different that lidar images amount!!!!")
        exit()

    tuples = ((x, y) for (x, y) in zip(raw_images, LiDAR_images))

    return tuples


def batch_image(image_name, dim=(13, 13), step=3):
    image = ImageManager('data/train/image/%s.tif' % (image_name))
    mask = ImageManager('data/train/label/%s.tif' % (image_name))
    image_batches = image.get_batches(dim, step)
    mask_batches = mask.get_batches(dim, step)
    im_mask = list(zip(image_batches, mask_batches))

    positive_list = [image for image, mask in im_mask if in_center(
        mask, dim[0], step+1)]
    positive_list = sample(positive_list, min(len(positive_list), 4000))
    negative_list = sample([image for image, mask in im_mask if not in_center(
        mask, dim[0], 6)],  min(len(im_mask), 5000))

    index = 0
    for positive in positive_list:
        with rasterio.open(
            'data/train/image/True/%s__%s.tif' %
            (str(image_name), str(index)),
            'w', driver='GTiff', height=positive.shape[1],
                width=positive.shape[2], count=positive.shape[0], dtype=positive.dtype) as dst:
            dst.write(positive)
        index += 1

    index = 0
    for negative in negative_list:
        with rasterio.open(
            'data/train/image/False/%s__%s.tif' %
            (str(image_name), str(index)),
            'w', driver='GTiff', height=negative.shape[1],
                width=negative.shape[2], count=negative.shape[0], dtype=negative.dtype) as dst:
            dst.write(negative)
        index += 1


def load_test_tree_point(path, new_crs):
    locations = gpd.read_file(path)
    location_proj = locations.copy()
    location_proj['geometry'] = location_proj['geometry'].to_crs(new_crs)
    frames = location_proj.head(len(location_proj))
    return get_trees_for_frames(frames)


def show_paint_image(image_manager, points):
    image_array = image_manager.get_image_as_array()[:, 0:1000, 0:1000]

    count = 1

    for point in points:
        real_point = image_manager.image.index(point.x, point.y)
        if (real_point[0] < 1000 and real_point[1] < 1000):
            count += 1
            paint(image_array, real_point[0], real_point[1])

    with rasterio.open('temp.tif', 'w', driver='GTiff', height=1000,
                       width=1000, count=3, dtype=image_array.dtype) as dst:
        dst.write(image_array)

    new_image = ImageManager('temp.tif')
    new_image.show()


def generate_test_labels(image_manager, tree_points):
    print('[GENERATING TEST DATA...]\n')

    print('-----')

    pos_points = [image_manager.image.index(
        point.x, point.y) for point in tree_points]

    print('-----')

    image_array = image_manager.get_image_as_array()

    print('-----')

    positive_list = sample([get_matrix_from_point(
        image_array, point) for point in pos_points], min(len(pos_points), 15000))

    print('-----')

    neg_x = np.random.randint(30, image_manager.height-30, 17700)
    neg_y = np.random.randint(30, image_manager.width-30, 17700)
    xy = zip(neg_x, neg_y)

    negative_list = [get_matrix_from_point(image_array, point)
                     for point in xy if not search_point_in_list(point, pos_points) and not image_array[1, point[0], point[1]] > 130]

    train_positive_data = positive_list[:int(len(positive_list)*0.7)]
    test_positive_data = positive_list[int(len(positive_list)*0.7):]

    print('-----')

    train_negative_data = negative_list[:int(len(negative_list)*0.7)]
    test_negative_data = negative_list[int(len(negative_list)*0.7):]

    print('len:', len(train_positive_data))
    # train
    index = 0
    for positive in train_positive_data:
        if positive is None:
            continue
        with rasterio.open(
            'data/testing/train/True/%s.tif' %
            (str(index)),
            'w', driver='GTiff', height=positive.shape[1],
                width=positive.shape[2], count=3, dtype=positive.dtype) as dst:
            dst.write(positive)
        index += 1

    index = 2000
    for negative in train_negative_data:
        if negative is None:
            continue
        with rasterio.open(
            'data/testing/train/False/%s.tif' %
            (str(index)),
            'w', driver='GTiff', height=negative.shape[1],
                width=negative.shape[2], count=3, dtype=negative.dtype) as dst:
            dst.write(negative)
        index += 1

    # test
    index = 0
    for positive in test_positive_data:
        if positive is None:
            continue
        with rasterio.open(
            'data/testing/test/True/%s.tif' %
            (str(index)),
            'w', driver='GTiff', height=positive.shape[1],
                width=positive.shape[2], count=3, dtype=positive.dtype) as dst:
            dst.write(positive)
        index += 1

    index = 2000
    for negative in test_negative_data:
        if negative is None:
            continue
        with rasterio.open(
            'data/testing/test/False/%s.tif' %
            (str(index)),
            'w', driver='GTiff', height=negative.shape[1],
                width=negative.shape[2], count=3, dtype=negative.dtype) as dst:
            dst.write(negative)
        index += 1

    print('\n[TEST DATA GENERATED]')


def paint(image, i, j):
    top = int(max(0, i-1))
    left = int(max(0, j-1))
    bottom = int(min(image.shape[0]-1, i+1))
    right = int(min(image.shape[1]-1, j+1))

    image[0, top, j] = 255
    image[0, top, right] = 255
    image[0, i, right] = 255
    image[0, bottom, right] = 255
    image[0, bottom, j] = 255
    image[0, bottom, left] = 255
    image[0, i, left] = 255
    image[0, top, left] = 255

    image[1, top, j] = 0
    image[1, top, right] = 0
    image[1, i, right] = 0
    image[1, bottom, right] = 0
    image[1, bottom, j] = 0
    image[1, bottom, left] = 0
    image[1, i, left] = 0
    image[1, top, left] = 0

    image[2, top, j] = 0
    image[2, top, right] = 0
    image[2, i, right] = 0
    image[2, bottom, right] = 0
    image[2, bottom, j] = 0
    image[2, bottom, left] = 0
    image[2, i, left] = 0
    image[2, top, left] = 0
