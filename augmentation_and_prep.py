from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import glob
from PIL import Image

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def augment_and_save_images(source_folder, target_folder, augment_times=500):
    os.makedirs(target_folder, exist_ok=True)
    images = glob.glob(source_folder + '/*.png')
    total_images = len(images)

    for i, filename in enumerate(images):
        img = load_img(filename)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Augment and save images
        j = 0
        for batch in datagen.flow(img_array, batch_size=1):
            aug_img = Image.fromarray(batch[0].astype('uint8'))
            aug_img = aug_img.resize((32, 32))  # Ensure all images are resized to 32x32
            aug_img.save(os.path.join(target_folder, f'aug_{i}_{j}.png'))
            j += 1
            if j >= augment_times // total_images:
                break


# Paths to the folders
cockroach_train_folder = '10TrainOrig'
augmented_train_folder = 'cifar10_train/10'

# Augment the cockroach images
augment_and_save_images(cockroach_train_folder, augmented_train_folder,3000)

cockroach_train_folder = '10TestOrig'
augmented_train_folder = 'cifar10_test/10'

# Augment the cockroach test images
augment_and_save_images(cockroach_train_folder, augmented_train_folder,500)