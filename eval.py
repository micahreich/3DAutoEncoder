import numpy as np
import pickle
import tensorflow as tf
import model
import matplotlib.pyplot as plt
import open3d as o3d


class Eval:
    def __init__(self):
        self.path_to_model = "pix_2_model"
        self.autoencoder = tf.keras.models.load_model(self.path_to_model)

        print("Loading dataset...")
        self.x = np.asarray(pickle.load(open("data/images.pkl", "rb"))).repeat(64, axis=0)
        self.y = np.asarray(pickle.load(open("data/voxels.pkl", "rb"))).repeat(64, axis=0)

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int32)

        self.x = self.x.reshape((-1, 3, 256, 256, 3))
        self.y = np.squeeze(self.y)

        print(self.x.shape, self.y.shape)

    def predict_point_cloud(self):
        image, voxel = self.x[0], self.y[0]
        return self.autoencoder.predict([image[:, 0], image[:, 1], image[:, 2]])

    def point_cloud_viz(self):
        point_cloud = self.predict_point_cloud()

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(point_cloud))

        pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
                  center=pcd.get_center())

        o3d.visualization.draw_geometries([pcd])

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                    voxel_size=0.05)
        o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    Eval = Eval()
    Eval.point_cloud_viz()

