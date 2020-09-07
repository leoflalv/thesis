from keras.preprocessing.image import ImageDataGenerator


def image_train_data_generator(directory, rotation_range=0, zoom_range=0,
                               classes=['False', 'True'], batch_size=16,  size=(13, 13)):
    datagen_args = dict(
        zoom_range=zoom_range,
        rotation_range=rotation_range,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    image_datagen = ImageDataGenerator(**datagen_args,)
    generator = image_datagen.flow_from_directory(
        directory, target_size=size, batch_size=batch_size, class_mode='binary')

    return generator


def image_test_data(directory, batch_size=16, size=(13, 13)):
    test_datagen = ImageDataGenerator()
    validation_ds = test_datagen.flow_from_directory(
        directory=directory,
        classes=['False', 'True'],
        class_mode='binary',
        batch_size=batch_size,
        target_size=size)

    return validation_ds


def image_train_validation_data_generator(directory, rotation_range=0, zoom_range=0,
                                          classes=['False', 'True'], batch_size=16,  size=(13, 13)):
    datagen_args = dict(
        zoom_range=zoom_range,
        rotation_range=rotation_range,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )

    image_datagen = ImageDataGenerator(**datagen_args,)

    train_generator = image_datagen.flow_from_directory(
        directory, target_size=size, batch_size=batch_size, class_mode='binary', subset='training')

    validation_generator = image_datagen.flow_from_directory(
        directory, target_size=size, batch_size=batch_size, class_mode='binary', subset='validation')

    return (train_generator, validation_generator)
