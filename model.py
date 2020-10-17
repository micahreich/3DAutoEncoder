import tensorflow as tf
from tensorflow.keras.layers import *


class AutoEncoder:
    def __init__(self):
        self.img_size = (256, 256, 3)
        self.voxel_size = (256, 256, 256)

    def build_autoencoder(self):
        #  encoder
        filter_array = [512, 256, 128]

        persp_1 = Input(shape=self.img_size)
        features_1 = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(persp_1)
        features_1 = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(features_1)

        persp_2 = Input(shape=self.img_size)
        features_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(persp_2)
        features_2 = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(features_2)

        persp_3 = Input(shape=self.img_size)
        features_3 = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(persp_3)
        features_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(features_3)

        merged = Concatenate()([features_1, features_2, features_3])

        x = merged
        #  encoder
        for filter in filter_array:
            x = Conv2D(filters=filter, kernel_size=(3, 3), strides=(2, 2),
                       activation="relu", padding="same")(x)

        #  decoder
        for filter in filter_array[::-1]:
            x = Conv2DTranspose(filters=filter, kernel_size=(3, 3), strides=(2, 2),
                                activation="relu", padding="same")(x)

        x = Conv2DTranspose(filters=1024, kernel_size=(3, 3), strides=(2, 2),
                                activation="relu", padding="same")(x)

        voxel_out = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2),
                            activation="sigmoid", padding="same")(x)

        # print(tf.keras.Model(
        #     [persp_1, persp_2, persp_3], voxel_out
        # ).summary())

        return tf.keras.Model([persp_1, persp_2, persp_3], voxel_out)