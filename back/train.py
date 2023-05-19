import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, Rescaling
from tensorflow.keras import layers

from utils import plot_results

PATH_MERGED = './datasets/merged_ds/'


def createModel(input_shape):
    # Definimos que estamos criando um modelo sequencial
    model = Sequential()

    # Primeira camada do modelo:
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada do modelo:
    model.add(Conv2D(64, (3, 3),  activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada do modelo:
    model.add(Conv2D(128, (3, 3),  activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


def run():
    input_shape = (200, 200, 3)  # Height, Width, Depth (1 -> cinza, 3 -> rgb)
    height = 200
    width = 200
    batch_size = 32
    epochs = 15
    model_path = os.path.join('./datasets/models/', "test-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.model")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_MERGED,
        seed=123,
        validation_split=0.3,
        subset="training",
        color_mode="rgb",
        image_size=(height, width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_MERGED,
        seed=123,
        validation_split=0.3,
        subset="validation",
        color_mode="rgb",
        image_size=(height, width),
        batch_size=batch_size,
    )

    # rescaling os valores de RGB entre 0 e 1
    rescaling_layer = Rescaling(1./255)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical', input_shape=input_shape),
        layers.RandomRotation(0.2),
    ])

    train_ds = train_ds.map(lambda x, y: ( rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda x, y: ( data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    model = createModel(input_shape)

    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    checkpoint_loss = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoints = [checkpoint_loss, checkpoint_acc]

    H = model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=epochs,
                  verbose=1,
                  callbacks=checkpoints)

    plot_results(H.history, range(epochs))


def main():
    run()


if __name__ == "__main__":
    main()
