# nn

A simple from-scratch implementation of an MNIST-classifier neural net in Zig, reaching 96.5% accuracy.

Built because I wanted to learn Zig and how neural nets work at the same time.

# Overview

The neural network library is located in `src/lib`, the MNIST usage example is under `src/mnist`.
The MNIST example has a 784 -> 32 -> 10 architecture and uses sigmoid activations.

# Running

You'll need Zig 0.16.0.

```sh
# Download the MNIST CSV files into assets/
./download_mnist.sh
# Build the project
zig build
# Run the tests
zig build test
# Run the MNIST classifier
zig build run -Doptimize=ReleaseSafe
# Benchmark the zero-cost abstraction linear algebra library
zig build bench -Doptimize=ReleaseSafe
```

# Features

The library implements a simple neural network. It features

- Feedforward and Backpropagation
- SGD with mini batches
- Numerical gradient checking
- Xavier/He parameter initialization
- A zero-cost abstraction linear algebra library

# Limitations

- The net uses Sigmoid and MSE rather than Cross Entropy Loss and Softmax
- The optimizer is basic SGD without momentum or Adam
- The project has an educational focus. My goal wasn't to achieve state-of-the-art accuracy or use a complex architecture, but to learn the basics of neural nets and how they work under the hood.

# Training Runs

| Seed | Epochs | Batch size | Hidden size | Learning rate | Test accuracy |
| ---: | -----: | ---------: | ----------: | ------------: | ------------: |
|   67 |     20 |         32 |          32 |            10 |        96.52% |
<img width="209" height="79" alt="Output from the training run after 20 epochs showing a test accuracy of 96.52%" src="https://github.com/user-attachments/assets/f562c5c8-a859-43bd-b4ed-8c090c11c564" />


# License

This project is licensed under the GNU General Public License v3.0 or later.
