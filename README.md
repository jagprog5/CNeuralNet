![](https://github.com/jagprog5/CNeuralNet/blob/master/reasources/_demo.gif)

# C Neural Network

Use `make run` to compile and run. Needs the [ncurses](https://www.cyberciti.biz/faq/linux-install-ncurses-library-headers-on-debian-ubuntu-centos-fedora/) library.

Or, run in a docker container (138MB image).

[Classifies the MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)

Properties of this network:
  * 28x28 inputs, 32 nodes in hidden layer, 10 output nodes
  * Sigmoid activation for hidden layer
  * Softmax activation for output layer
  * Cross entropy loss and stochastic gradient descent for training
  * Typically error rate of 7-8% for this database

Controls:
 * The LEFT and RIGHT arrow keys toggle the screen view
	 * *No output* is fast, and trains the network is ~30 seconds
	 * *Set view* shows the current image in the training set, and the network's output
		 * "Probs" shows the network's output for the image
		 * "Label" shows the image's correct corresponding digit, as given in the dataset
		 * "GOOD" & "BAD" indicates if the highest probability output matches the label
	 * *Field view* shows the receptive field for the output neurons
 * The UP and DOWN arrow keys are used to navigate
	 * In set view, this moves through the dataset
	 * In field view, this moves through the output neurons
 * Space starts and stops training
 * Backspace resets to the beginning of training

After training, the test set is loaded, and the error rate is calculated.

Make sure the terminal window size is large enough.
If running in WSL, run in tmux to avoid rendering issues.

