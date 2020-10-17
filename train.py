import numpy as np
import pickle
import tensorflow as tf
import model
import matplotlib.pyplot as plt


class Train:
    def __init__(self):
        self.epochs = 1000
        self.batch_size = 32

        self.img_size = (256, 256, 3)
        self.voxel_size = (256, 256, 256)

        AutoEncoder = model.AutoEncoder()
        #strategy = tf.distribute.MirroredStrategy()
        #with strategy.scope():
        self.autoencoder = AutoEncoder.build_autoencoder()
        self.autoencoder.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(0.0002)
        )

        print("Loading dataset...")
        self.x = np.asarray(pickle.load(open("data/images.pkl", "rb"))).repeat(64, axis=0)
        self.y = np.asarray(pickle.load(open("data/voxels.pkl", "rb"))).repeat(64, axis=0)

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int32)

        self.x = self.x.reshape((-1, 3, 256, 256, 3))
        self.y = np.squeeze(self.y)

        print(self.x.shape, self.y.shape)

    def sample_training_data(self, batch_size):
        idx = np.random.randint(0, self.x.shape[0], batch_size)
        return self.x[idx], self.y[idx]

    def train(self):
        for epoch in range(self.epochs):
            images, voxels = self.sample_training_data(self.batch_size)
            ae_loss = self.autoencoder.train_on_batch([images[:, 0], images[:, 1], images[:, 2]], voxels)

            print("Epoch %d AE loss: %f" % (epoch, ae_loss))

        print("Training complete, saving model...")
        self.autoencoder.save("pix_2_model")


if __name__ == "__main__":
    Train = Train()
    Train.train()
