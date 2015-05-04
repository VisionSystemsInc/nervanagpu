# NervanaGPU library

## Introduction

**nervanagpu** is a Python module for deep learning. It includes,

- matrix-multiply (GEMM), convolution, and pooling kernels optimized using a custom [assembler](https://github.com/NervanaSystems/maxas),
- element-wise and broadcast operations that automatically compound into efficient kernels,
- a simple but powerful array class, leveraging and with code partially borrowed from pycuda<sup>[1](#refs)</sup>,
- layer classes for building networks for benchmarking,
- full assembler source to encourage contributions from the community.

#### Design goals

**nervanagpu** grew out of a tool Nervana uses for internal hardware efforts. It's been repackaged for use by the community. The goals for **nervanagpu** are to provide,

- near **theoretical peak performance**,
- numpy functionality for **ease-of-use**,
- convolution kernel features and arguments identical to cuDNN<sup>[2](#refs)</sup>,
- integration into [neon](https://github.com/NervanaSystems/neon), Nervana's full-featured deep learning library,
- a tool for algorithmic explorations using alternative numerical formats,
- a seemless transition path to Nervana hardware,
- ease of integration into other deep learning frameworks.

Only NVIDIA Maxwell and future architectures are supported. Older architectures are not well-suited for assembler level optimizations used here.

#### Numerical formats

Supported numerical formats currently include,

- **fp32**: standard 32-bit floating point,
- **fp16**: 16-bit floating point memory format with underlying operations in 32 bits.
- **int8** and **uint8**: in elementwise and as input to the first convolutional layer.

with more to come (eg. like [this](https://github.com/NervanaSystems/nervana-lib-gpu-performance-preview)).

#### Extra features

Our kernels have some additional useful features:

- 3D convolutions and 4D pooling (including output feature map dim)
- optional ReLu is builtin to GEMM and convolution operations,
- stochastic rounding support for **fp16**<sup>[3](#refs)</sup>,
- instrumented to return statistics useful for avoiding numerical issues (coming soon),
- support for matrix sizes common in deep learning, significantly out performing cuBLAS

Small optimizations like these can result in significant speed and performance improvements.

## Usage

**nervanagpu** includes a factory class `NervanaGPU` and a numpy-like array class `GPUTensor`. Memory layout for tensors and gemm ops is **row-ordered**. Below are examples on how they are used.

### Matrix multiplication example

Here is full example of doing a basic GEMM operation using 16-bit float:

```python
import numpy as np
import pycuda.autoinit
from nervanagpu import NervanaGPU

# initialize factory class
ng = NervanaGPU(stochastic_round=False)

m, n, k  = 10, 20, 10
dtype = np.float16

# create matrices on host
cpuA = np.random.randn(k,m)
cpuB = np.random.randn(k,n)

# transfer to device
devA = ng.array(cpuA, dtype=dtype)
devB = ng.array(cpuB, dtype=dtype)
devC = ng.empty((m,n), dtype=dtype)

# do GEMM operation
ng.dot(devA.T, devB, devC, relu=False)

# get from device
cpuC = devC.get()
```

### Element-wise operations

**nervanagpu** compiles tensor arithmetic expressions into efficient CUDA kernels which are lazily evaluated upon assignment. For example, computing variance along an axis consists of a set of element-wise, reduction and broadcast operations that compiles to a single kernel:

```python
# import and initialize NervanaGPU, transfer matrix from cpu to dev as above

devC[:] = ng.mean(ng.square(devA - ng.mean(devA, axis=1)), axis=1)

```

Batch normalization can be done by computing mean and variance across the batch (n) dimension and automatically taking advantage of broadcasting to subtract and divide the original data.

```python
# import and initialize NervanaGPU as above

eps  = .001
A    = ng.empty((128, 32), dtype=np.float16)
A[:] = ng.rand() # generate uniform random on device between 0 and 1

# Normalize batch data by batch mean and variance, 
A[:] = ng.reciprocal(ng.sqrt(ng.var(A, axis=1) + eps)) * (A - ng.mean(A, axis=1)) 

```
The last expression above is automatically collapsed into a single gpu kernel. The two mean(A,axis=1) expressions are automatically simplified into one. To be able to broadcast a reduction to a subsequent operation the reduction op must appear prior to the broadcast op in the [*postfix*](http://en.wikipedia.org/wiki/Reverse_Polish_notation) version of the expression. Hence, the reciprocal operation instead of division.

## Building

**nervanagpu** comes with full assembler code for kernels. To build the kernels, install [**maxas**](https://github.com/NervanaSystems/maxas), Nervana's assembler for NVIDIA Maxwell. The module can then be built by running:

    make kernels      # build the kernels
    make python       # build python bindings
    make test         # run nosetests
    make doc          # build sphinx docs

A simple `make` will perform the first two steps.

Documentation and tests are currently sparse. Please contribute.

## Performance

**nervanagpu** comes with a set of benchmark scripts under `nervanagpu/benchmarks`. Also included are scripts to validate kernel results against cuBLAS and cuDNN.

Here is a sample run of `benchmarks/convnet-benchmarks.py` using the networks listed on Soumith Chintala's [benchmarking page](https://github.com/soumith/convnet-benchmarks).  Run on a single TitanX with default clocks and power limit:

    ---------------------------------------------
    Alexnet (dtype=float16, N=128)  Results:
    ---------------------------------------------
    Avg(10) fprop:   29.498 msecs 6043.698 gflops
    Avg(10) bprop:   66.562 msecs 5356.689 gflops
    Avg(10) total:   96.059 msecs 5567.654 gflops
    ---------------------------------------------
    Alexnet (dtype=float32, N=128)  Results:
    ---------------------------------------------
    Avg(10) fprop:   31.251 msecs 5704.698 gflops
    Avg(10) bprop:   77.567 msecs 4596.660 gflops
    Avg(10) total:  108.818 msecs 4914.869 gflops

    ---------------------------------------------
    Overfeat (dtype=float16, N=128)  Results:
    ---------------------------------------------
    Avg(10) fprop:  116.723 msecs 6134.994 gflops
    Avg(10) bprop:  242.084 msecs 5916.054 gflops
    Avg(10) total:  358.807 msecs 5987.277 gflops
    ---------------------------------------------
    Overfeat (dtype=float32, N=128)  Results:
    ---------------------------------------------
    Avg(10) fprop:  124.569 msecs 5748.559 gflops
    Avg(10) bprop:  278.408 msecs 5144.192 gflops
    Avg(10) total:  402.977 msecs 5331.015 gflops

    ---------------------------------------------
    VGG (dtype=float16, N=64)  Results:
    ---------------------------------------------
    Avg(10) fprop:  162.186 msecs 5978.348 gflops
    Avg(10) bprop:  357.850 msecs 5419.051 gflops
    Avg(10) total:  520.036 msecs 5593.481 gflops
    ---------------------------------------------
    VGG (dtype=float32, N=64)  Results:
    ---------------------------------------------
    Avg(10) fprop:  170.822 msecs 5676.112 gflops
    Avg(10) bprop:  438.031 msecs 4427.108 gflops
    Avg(10) total:  608.853 msecs 4777.533 gflops


#### Acknowledgements

Thanks to Erich Elsen and Bryan Catanzaro of Baidu, Matthieu Courbariaux and Frédéric Bastien of the Bengio lab, Vincent Vanhoucke of Google, and Soumith Chintala of Facebook for feedback on early versions of this library. We'd also like to thank NVIDIA for generously providing us with several TitanXs for benchmarking.


#### References <a name="refs"></a>

1. Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov, Ahmed Fasih.
[*PyCUDA and PyOpenCL: A scripting-based approach to GPU run-time code generation*](http://arxiv.org/abs/0911.3456)
Parallel Computing, Volume 38, Issue 3, March 2012, Pages 157-174.

2. Chetlur, Sharan, Cliff Woolley, Philippe Vandermersch, Jonathan
Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer.
[*cuDNN: Efficient primitives for deep learning.*](http://arxiv.org/abs/1410.0759)
arXiv preprint arXiv:1410.0759 (2014).

3. Gupta, Suyog, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. [*Deep Learning with Limited Numerical Precision.*](http://arxiv.org/abs/1502.02551) arXiv preprint arXiv:1502.02551 (2015).

