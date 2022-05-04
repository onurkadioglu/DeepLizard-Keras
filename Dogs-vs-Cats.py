import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
import shutil
import random
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('dogs-vs-cats')

if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('cat*'), 200):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 200):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 40):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 40):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('cat*'), 20):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 20):
        shutil.move(c, 'test/dog')

os.chdir('../../')

train_path = r'C:\Users\Kadıoğlu\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\dogs-vs-cats/train/'
valid_path = r'C:\Users\Kadıoğlu\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\dogs-vs-cats/valid/'
test_path = r'C:\Users\Kadıoğlu\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\dogs-vs-cats/test/'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

assert train_batches.n == 400
assert valid_batches.n == 80
assert test_batches.n == 40
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize =(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)

# This model is created to observe different layers of CNNs

#model = Sequential([
#    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
#    MaxPool2D(pool_size=(2,2), strides=2),
#    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
#    MaxPool2D(pool_size=(2,2), strides=2),
#    Flatten(),
#    Dense(units=2, activation='softmax'),
#])

#model.summary()

#model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

#predictions = model.predict(x=test_batches, verbose=0)
#np.round(predictions)
#cm = confusion_matrix(y_true=test_labels, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes,
            normalize=False,
            title= 'Confusion Matrix',
            cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')

    print (cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


vgg16_model = tf.keras.applications.vgg16.VGG16()
type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)

predictions = model.predict(x=test_batches, verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion_Matrix')
plt.show()


