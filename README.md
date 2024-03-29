# Neural Terrain 2
This project contains code for generating realistic procedural heightmaps using neural networks. Concretly, it uses a stack of two Generative Adversarial Networks (GAN). The first generates low resolution data from noise. The second is a Conditional GAN, which upsamples the low resolution heightmaps from the first GAN. This system is trained on a very small dataset (1.5 MB) of real heightmaps from satellite data. The code automatically balances the training frequency of generator and discriminator.
Since both GANs contain only convolutional layers (no fully-connected ones) it is possible to generate unlimited seamless heightmaps on-the-fly. By creating input noise in a cyclic way, it is also possible to generate tiling heightmaps.

This is an example heightmap generated using this model:
![alt text](https://raw.githubusercontent.com/liquidnode/neural_terrain_2/master/HMaps/HMap.png)

This system is used in this project https://www.youtube.com/watch?v=PfyQ_DYArwA.

To train this model you need:
- Caffe: a fast open framework for deep learning. 
