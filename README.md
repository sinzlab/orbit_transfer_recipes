# Can Functional Transfer Methods Capture Simple Inductive Biases?

This repository contains all experiments that we show in our publication *Can Functional Transfer Methods Capture Simple Inductive Biases?*

## :gear: Usage:

The code underlying the experiments described in this repository can be found in [orbit_transfer](https://github.com/sinzlab/orbit_transfer).

For details on how to run the experiment, please see [nntransfer_recipes](https://github.com/sinzlab/nntransfer_recipes). 

The experiments further require installation of:
- [nntransfer](https://github.com/sinzlab/nntransfer)
- [nnfabrik](https://github.com/sinzlab/nnfabrik)
- [neuralpredictors](https://github.com/sinzlab/neuralpredictors)
- [pytorch_warmup](https://github.com/ArneNx/pytorch_warmup)

## :microscope: Experiments:

The configuration of the following experiments can be found in `orbit_transfer_recipes/_2021_09_24_aistats`

### 1D MNIST

`mnist_1d_hypersearch.py`: Grid-search for initial hyperparameters across different transfer methods and a shifts in range `[0,30]`

`mnist_1d_with_pooling_hypersearch.py`: Same as above, but with a student network that includes a pooling layer.

`mnist_1d_shift.py`: For the hyperparameters we found in the grid-search, we train models across training setting with all possible shift settings.

`mnist_1d_with_pooling_shift.py`: Same as above, but with a student network that includes a pooling layer.

### 2D MNIST: Translation Equivariance

`mnist_2d_cnn_linear.py`: Comparison of different functional transfer methods on centered and translated MNIST.

`mnist_2d_resnet_vit.py`: Same as above, but transferring between a ResNet18 and a small VIT.

`mnist_2d_cnn_linear_loss_ablation.py`: Orbit transfer loss ablation.

### 2D MNIST: Rotation Equivariance

`mnist_2d_rotation.py`: Transferring from a rotation equivariant teacher to an MLP. 

## :bug: Report bugs 

In case you find a bug, please create an issue or contact any of the contributors.
