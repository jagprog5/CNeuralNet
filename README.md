![Demo](https://github.com/jagprog5/CNeuralNet/blob/master/_demo.gif)

<sup>Gif was edited for brevity. The program still usually completes in less than 1 minute.</sup>

# C Neural Network

* [Classifies the MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)
* 28x28 inputs, 32 nodes in hidden layer, and 10 output nodes
* Properties of this network:
  * Sigmoid activation on hidden layer
  * Softmax activation on output layer
  * Cross entropy loss and stochastic gradient descent for training
  * Typically has an error rate of 7-8%

Use the commands:
* `make run` to compile and run
* `make run_reduced` to compile and run with reduced output
* `make clean` will remove the object files and executable.
