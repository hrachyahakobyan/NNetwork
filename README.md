# NNetwork
A C++ implementation of feed-forward multi-layer neural network. 
Uses Eigen C++ matrix library for matrix operations. 

# NNetwork

A feed-forward neural network class with adjustable user-defined number of layers and neurons in each layer.
Currently supporting two types of activation functions 
* Sigmoid
* Softmax

The program supports 2 types of cost functions (yet)
* Quadratic cost
* Cross-entropy

Learning is based on the stochastic gradient method with adjustable batch size and learning rate.

The configuration properties are changed for debug and release version. 
To achieve maximum performance, release version is tuned to optimize the code using compiler optimization, SSE2 extended instruction set and OpenMP. Debuggin is turned off for these version.
