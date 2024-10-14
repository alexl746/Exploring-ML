Both programs in this folder detect digits from the MNIST dataset, the one labelled "feed forward neural net to detect digit.py" takes advantage of PyTorch so a lot of the neural network is abstracted from the program and is handled by the library, especially the loss and gradient descent methods to tweak the weights and biases of the neural network. This one has a higher accuracy of around 95%, and also outputs some example digits from the MNIST dataset.

The one labelled "detecting digits.py" doesn't use any machine learning libraries and has the maths coded into the program (except from dot product which is done through numpy). This gets an accuracy of around 80% but only does 500 iterations whereas the PyTorch program does 1200 so the difference in accuracy could be due to PyTorch being better or due to the lack of data given to the non-PyTorch program.

Both use a neural network, forward/back propagation and gradient descent to tweak the weight and bias parameters.
