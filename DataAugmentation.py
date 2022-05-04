import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
                        channel_shift_range=10., horizontal_flip=True)

chosen_image = random.choice(os.listdir('dogs-vs-cats/train/dog'))
image_path = r'C:\Users\Kadıoğlu\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\dogs-vs-cats/train/dog/' + chosen_image

image = np.expand_dims(plt.imread(image_path),0)
plt.imshow(image[0])

aug_iter = gen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
plotImages(aug_images)
plt.show()

aug_iter = gen.flow(image, save_to_dir=r'C:\Users\Kadıoğlu\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\dogs-vs-cats/train/dog/',
                    save_prefix='aug-image-', save_format='jpeg')

