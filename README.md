![](https://github.com/jagprog5/CNeuralNet/blob/master/reasources/_demo.gif)

# C Neural Network

Use `make run` to compile and run. Needs the [ncurses](https://www.cyberciti.biz/faq/linux-install-ncurses-library-headers-on-debian-ubuntu-centos-fedora/) library.

[Classifies elements in the MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)

Properties of this network:
  * 28x28 inputs, 32 nodes in hidden layer, 10 output nodes
  * Sigmoid activation for hidden layer
  * Softmax activation for output layer
  * Stochastic gradient descent for training
  * Typically error rate of 7-8% for this database

Screen Views:
 * *No-output* is fast
   * Doesn't update the screen, but...
   * Trains in ~2 seconds. BLAZINGLY fast
 * *Set-view* shows the current image in the training set, and the network's output
	 * "Probs" shows the network's output for the image
	 * "Label" shows the image's correct corresponding digit, as given in the dataset
	 * "GOOD" & "BAD" indicates if the highest probability output matches the label
 * *Field-view* shows the receptive field for the output neurons

After training, the test set is loaded, and the error rate is calculated.

