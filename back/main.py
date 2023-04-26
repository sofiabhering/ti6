import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, GlobalAvgPool2D, Rescaling
from tensorflow.keras import layers

from utils import *

PATH_FA = './datasets/face_age/'
PATH_UTK = './datasets/utk/'
PATH_MERGED = './datasets/merged_ds/'

def plotNumImgsPerAge():

    def get_age(filename):
        return int(filename.split('_')[0])

    images = {}

    dir_fa = os.listdir(PATH_FA)
    dir_utk = sorted(os.listdir(PATH_UTK), key=get_age)

    for i in range(17):
        age = dir_fa[i]
        curr_path = os.path.join(PATH_FA, age)
        num_imgs = len(os.listdir(curr_path))
        images[int(age)] = num_imgs

    for file in dir_utk:
        age = int(file.split('_')[0])
        if age < 18:
            images[int(age)] += 1

    pltDataDistribution(images)


def processDataSets(plot=False):
    """
    Processa os datasets, separando entre as classes <= 18 e > 18 

    Parametros:
    -----------
    plot:
        Opcional, plota gráfico dos datasets
    """
    dir_fa = os.listdir(PATH_FA)
    dir_utk = os.listdir(PATH_UTK)

    _imgs = {
        "0": [],  # under 18
        "1": []
    }

    for age in dir_fa:
        curr_path = os.path.join(PATH_FA, age)
        key = "0" if int(age) <= 18 else "1"
        _imgs[key].extend(age + '_' + name for name in os.listdir(curr_path))

    if plot:
        pltDataDistribution(_imgs)

    return _imgs


def createModel(input_shape):
    # Definimos que estamos criando um modelo sequencial
    model = Sequential()

    # Primeira camada do modelo:
    model.add(Conv2D(32, 3, padding="same",
              activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada do modelo:
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Terceira camada do modelo:
    model.add(Conv2D(128, 3, padding="same",
              activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(132, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

def run():
    input_shape = (200, 200, 3)  # Height, Width, Depth (1 -> cinza, 3 -> rgb)
    height = 200
    width = 200
    batch_size = 32
    epochs = 10
    model_path = os.path.join(
        './datasets/models/', "test-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.model")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_MERGED,
        seed=123,
        validation_split=0.2,
        subset="training",
        color_mode="rgb",
        image_size=(height, width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_MERGED,
        seed=123,
        validation_split=0.2,
        subset="validation",
        color_mode="rgb",
        image_size=(height, width),
        batch_size=batch_size,
    )

    # rescaling os valores de RGB entre 0 e 1
    rescaling_layer = Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (
        rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    model = createModel(input_shape)

    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    checkpoint_loss = ModelCheckpoint(
        model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = ModelCheckpoint(
        model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoints = [checkpoint_loss, checkpoint_acc]

    H = model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=epochs,
                  verbose=1,
                  callbacks=checkpoints)

    plot_results(H.history, range(epochs))

def preprocess():
    merged_ds = processDataSets(True)
    if not os.path.exists(PATH_MERGED):
        os.makedirs('./datasets/merged_ds/0/')
        for i in merged_ds['0']:
            age, filename = i.split('_')
            src = os.path.join(PATH_FA+age+'/', filename)
            shutil.copy(src, './datasets/merged_ds/0/')
        os.makedirs('./datasets/merged_ds/1/')
        for i in merged_ds['1']:
            age, filename = i.split('_')
            src = os.path.join(PATH_FA+age+'/', filename)
            shutil.copy(src, './datasets/merged_ds/1/')

def main():
    preprocess()
    run()

if __name__ == "__main__":
    main()
