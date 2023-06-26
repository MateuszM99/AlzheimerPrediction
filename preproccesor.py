import os
import os.path
from os import path
import shutil
import splitfolders
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMAGE_SIZE = [176, 208]
SEED = 1337
WORK_DIR = './Alzheimer_Sampled'
INPUT_DIR = './Alzheimer_s Dataset'
FOLDERS = ['train', 'test', 'val']


def images_by_class_counter():
    normalCounter, verymildCounter, mildCounter, moderateCounter = 0, 0, 0, 0
    msg = '{:8} {:8} {:11} {:7} {:9}'.format('folder', 'normal', 'verymild', 'mild', 'moderate')
    print(msg)
    print("-" * len(msg))

    for folder in FOLDERS:
        for dirname, _, filenames in os.walk(os.path.join(WORK_DIR, folder)):
            for file in filenames:
                if "NonDemented" in dirname:
                    normalCounter += 1
                if "VeryMildDemented" in dirname:
                    verymildCounter += 1
                if 'MildDemented' in dirname:
                    mildCounter += 1
                if 'ModerateDemented' in dirname:
                    moderateCounter += 1

        print("{:6} {:8} {:10} {:7} {:11}".format(folder, normalCounter, verymildCounter, mildCounter,
                                                  moderateCounter))
        normalCounter, verymildCounter, mildCounter, moderateCounter = 0, 0, 0, 0


def resample_data(seed=SEED, split=(0.80, 0.20)):
    dir_test = os.path.join(INPUT_DIR, 'test')
    dir_train = os.path.join(INPUT_DIR, 'train')

    for folder in FOLDERS:
        if path.exists(os.path.join(WORK_DIR, folder)):
            shutil.rmtree(os.path.join(WORK_DIR, folder))

    shutil.copytree(dir_test, os.path.join(WORK_DIR, 'test'))
    splitfolders.ratio(dir_train, WORK_DIR, seed=seed, ratio=split)


class DataPreprocessor:
    train = []
    test = []
    val = []

    def load_image_datasets(self):
        resample_data()
        images_by_class_counter()

        train_images = ImageDataGenerator(rescale=1. / 255)
        val_images = ImageDataGenerator(rescale=1. / 255)
        test_images = ImageDataGenerator(rescale=1. / 255)

        self.train = train_images.flow_from_directory(
            os.path.join(WORK_DIR, 'train'),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            seed=SEED
        )
        self.test = test_images.flow_from_directory(
            os.path.join(WORK_DIR, 'test'),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            seed=SEED
        )
        self.val = val_images.flow_from_directory(
            os.path.join(WORK_DIR, 'val'),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            seed=SEED
        )
