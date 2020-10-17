import numpy as np
import os
from PIL import Image, ImageOps
import pickle
import scipy.io
import csv


class DataLoader:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.img_size = (192, 256, 3)

    def resize_images(self, id, fname):
        img = Image.open("data/images/{}/{}.png".format(id, fname)).convert("RGB")
        img_reshape = ImageOps.expand(img, (0, 64, 0, 0), fill=0)

        return np.asarray(img_reshape)

    def voxel_to_array(self, id):
        return np.asarray(scipy.io.loadmat("data/voxels/{}/model.mat".format(id))['input'])

    def pickle_dataset(self):
        images = []
        voxels = []

        for model in os.listdir(self.path_to_data)[1:]:
            persp_stack = [self.resize_images(id=model, fname="000"),
                           self.resize_images(id=model, fname="006"),
                           self.resize_images(id=model, fname="001")]

            images.append(persp_stack)
            voxels.append(self.voxel_to_array(model))

        image_outfile = open("data/images.pkl", 'wb')
        voxel_outfile = open("data/voxels.pkl", 'wb')

        print("Dumping into pickle files...")
        pickle.dump(np.asarray(images), image_outfile)
        pickle.dump(np.asarray(voxels), voxel_outfile)

        image_outfile.close()
        voxel_outfile.close()

        print(np.asarray(images).shape)
        print(np.asarray(voxels).shape)


if __name__ == "__main__":
    DataLoader = DataLoader("data/images/")
    DataLoader.pickle_dataset()
