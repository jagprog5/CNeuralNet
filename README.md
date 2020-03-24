![Demo](https://github.com/jagprog5/CNeuralNet/blob/master/_demo.gif)

<sup>Gif was edited for brevity. The program still usually completes in less than 1 minute.</sup>

# C Neural Network

[Classifies the MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)

The program is general and allows for arbitrarily sized networks (within reason).

Properties of this network:
  * 28x28 inputs, 32 nodes in hidden layer, and 10 output nodes
  * Sigmoid activation for hidden layer
  * Softmax activation for output layer
  * Cross entropy loss and stochastic gradient descent for training
  * Typically has an error rate of 7-8% for this database

In the gif:
* "Label" shows the image's correct corresponding digit, as given in the database
* "Probs" shows the neural networks output as probabilities for each digit
* "Final" shows the network's output with the highest probability, its final guess

Use the commands:
* `make run` to compile and run
* `make run_reduced` to compile and run with reduced output
* `make clean` will remove the object files and executable.
