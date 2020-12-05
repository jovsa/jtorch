# jTorch
An autograd engine and a neural net library implemented for personal exploration and learning.
The design is inspired by [minitorch](https://minitorch.github.io/) and [pytorch](https://pytorch.org/).

## Capabilities:
* Dynamic graph construction
    * Variable tracking
    * Gradient calculations
    * [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
    * [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

* Tensor support
    * No dependency on external tensor libraries (eg. [numpy](https://numpy.org/))
    * Allow for [tensor broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)

* GPU support
    * Parallel computation on a single GPU
    * Fusing operations on tensors
    * [CUDA](https://developer.nvidia.com/cuda-toolkit) support

* Basic Neural Network building block support
    * Convolution
    * Pooling
    * Softmax
    * Basic training harness

### Tests

All tests are located in `tests/`. To run all tests:
```
pytest tests/
```
> NOTE: Machine needs to have a GPU to run CUDA tests. Expect these tests to fail if you only have a CPU on board.



