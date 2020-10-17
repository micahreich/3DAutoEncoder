# 3D Autoencoder
A generative autoencoder to create 3D models from perspective images

### Model
The model used is an encoder-decoder network created with Tensorflow for Python. The model takes as input three RGB perspective images of the object and outputs a voxel matrix of the 3D model--this voxel object can be converted to STL or a point cloud for visualization or use in CAD programs. The autoencoder performs feature extraction from the input images with independent weights, then the feature maps are concatenated depth-wise to create one representation of the input images. Then, this image embedding is downsampled with convolutional layers to a latent representation of the object. A decoder network upsamples the latent representation into a 3D matrix of points which represents the voxel representation of the object.

### Dataset
This model uses a subset of the ShapeNet dataset which can be found at https://shapenet.org.
