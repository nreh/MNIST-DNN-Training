# MNIST-DNN-Training
Training an Deep Neural Network (DNN) on the MNIST dataset.

## Goal

Create minimum implementation of backpropagation algorithm to train a DNN as both an educational resource and a starting
point for implementing a training algorithm.

## Quick Start

To build project,
```
mkdir build
cd build
cmake ..
make
```

and then to run,
```
./runme
```

## Issues

- At the moment, training only really works for 1 hidden layer. Adding more layer results in terrible training convergence.

- Only one cost function, and not one ideal for classification.

## Future Goals

- Implement ADAM optimizer for better training

- Implement multithreaded calculation

- Use smart pointers rather than raw pointers

- In the future I plan on creating some UI for training and analysing the network rather than doing it through terminal.
Numerous variables and program structure decisions are made with this in mind.