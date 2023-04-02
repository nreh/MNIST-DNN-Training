# MNIST-DNN-Training
Training a Deep Neural Network (DNN) on the MNIST dataset.

## 🥅 Goal

Create minimum implementation of backpropagation algorithm to train a DNN as both an educational resource and a starting
point for implementing a training algorithm.

## ⭐ Quick Start

Before building project, initialize and update submodules,
```
git submodule init
git submodule update
```

To build project, in root project folder,
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

and then to run,
```
./runme --help
```

To display possible runtime arguments,

```
Create minimum implementation of backpropagation algorithm to train a DNN as both an educational resource and a starting point for implementing a training algorithm.
Usage: MNIST-DNN-Training [OPTIONS]

Options:
  -h,--help                         Print this help message and exit
  --training_data TEXT REQUIRED     Path to training data file
  --training_labels TEXT REQUIRED   Path to training labels file
  --test_data TEXT REQUIRED         Path to test data file
  --test_labels TEXT REQUIRED       Path to test labels file
  -v,--verbose [0]                  Print out debug information as well
  --no-logging{false} [1]           Disable logging by passing the --no-logging flag
```

⚠️ These instructions were tested on Ubuntu environment. When building on Windows or some other operating system, the compiled binary might be in a different folder and so the exact commands and folder structure might be different.

To start training, pass the proper paths for training/test data as well as any flags,

```
./runme --training_data ..\training_data\bin\train-images.idx3-ubyte \
        --training_labels ..\training_data\bin\train-labels.idx1-ubyte \
        --test_data ..\training_data\bin\test-images.idx3-ubyte \
        --test_labels ..\training_data\bin\test-labels.idx1-ubyte
```

You can also pass the `-v` flag to enable verbose debugging

## 🚫 Issues

- At the moment, training only really works for 1 hidden layer. Adding more layer results in terrible training convergence.

- Only one cost function, and not one ideal for classification.

## 📃 Future Goals

- Implement AVX intrinsics for vectorized dot product calculation. While compilers like gcc and clang can usually optimize this automatically, my experience when building with different compilers has been inconsistent and suggests that there can be major differences in speed depending on which compiler is used.

- Implement ADAM optimizer for better training

- Implement multithreaded calculation (Possibly using OpenMP)

- Use smart pointers rather than raw pointers

- In the future I plan on creating some UI for training and analysing the network rather than doing it through terminal.
Numerous variables and program structure decisions are made with this in mind.