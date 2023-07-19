import numpy as np

class MyClassifier:
    def one_hot_encoding(self):
        """
        Converts the true labels to one-hot encoded form.

        Args:
            Y (numpy.ndarray): True labels.

        Returns:
            numpy.ndarray: One-hot encoded labels.
        """
        #creates a matrix of zeros initially for all the data
        Y = self.Y
        matrix = np.zeros((Y.size, 2))

        # Iterates over the indices and assign 1 to the corresponding element in each row
        for i, y in enumerate(Y):
            matrix[i, y] = 1

        # Return the transposed matrix
        return matrix.T

    def __init__(self, X, alpha, iterations, Y):
        """
        initialises a neural network with the specified number of features, hidden size, and output size.

        Args:
            total_features (int): total number of features in the input data.
            hidden_size (int): Num of hidden layers
            output_size (int): Num of neurons in the output layer.
        """
        self.X = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        # self.X = pd.read_csv(X)  # we are reading the entire contents via pandas.
        self.alpha = alpha
        self.iterations = iterations
        self.dim = np.shape(self.X)

        self.Y = self.X[:, 0]

        self.one_hot = self.one_hot_encoding()
        self.gradient_decent()

    def initalise_parameters(self):
        # takes in dimension of dataset X and 2 output neurons
        weights1 = np.random.randn(2, 55)
        # 2 neurons in the hidden layer and 2 output neurons
        weights2 = np.random.randn(2, 2)

        # single col vector, but each of the two neurons need a unique val
        bias1 = np.random.randn(2, 1)
        bias2 = np.random.rand(2, 1)
        # generates a random for the dims specified.
        # number between the specific ranges of -0.5 and 0/5
        # this ensures symmetry.
        return weights1, weights2, bias1, bias2

    def rectified_linear_activation_funct(self, Z1):
        """
        applies the Rectified Linear Unit (ReLU) activation function element-wise to the input Z.

        Args:
            Z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Result of applying the ReLU activation function to Z
        """
        return np.maximum(Z1, 0)

    def softmax_activation_funct(self, Z2):
        """
        applies the softmax activation function to the input Z

        Args:
            Z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Result of applying the softmax activation function to Z
        """
        exp_Z = np.exp(Z)
        # return  the exponential of the input / the total sum of the exponential input.
        return np.exp(Z2) / sum(np.exp(Z2))

    def forward_propogation(self, weights1, weight2, bias1, bias2, X):
        """
        performs forward propagation through the neural network

        args:
            X (numpy.ndarray): Input data

        returns:
            tuple: Tuple containing the intermediate results of forward propagation
                   (Z1, A1, Z2, A2)
        """
        Z1 = np.dot(weights1, X.T) + bias1
        layer1_out = self.rectified_linear_activation_funct(Z1)

        # weights * output for activation function + bias
        Z2 = np.dot(weight2, layer1_out) + bias2
        layer2_out = self.softmax_activation_funct(Z2)

        return Z1, Z2, layer1_out, layer2_out

    def ReLU_deriv(self, Z):
        """
        computes the derivative of the ReLU activation function with respect to the input Z

        Args:
            z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Derivative of the ReLU activation function with respect to Z
        """
        return Z > 0

    def backward_propogation(self, Z1, Z2, layer1_out, layer2_out, weights2):
        """
        Performs backward propagation to compute the gradients of the parameters.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            Z1 (numpy.ndarray):  output of forward propagation (hidden layer pre-activation)
            A1 (numpy.ndarray): output result of forward propagation (hidden layer after activation)
            Z2 (numpy.ndarray): result of forward propagation (output layer pre-activation)
            A2 (numpy.ndarray):  result of forward propagation (output layer after activation) final

        Returns:
            tuple: Tuple containing the gradients of the parameters:
                   (dW1, db1, dW2, db2)
        """
        one_hot_format = self.one_hot
        m, n = self.dim

        # layer 2 first as we are calculating in reverse
        diffZ2 = layer2_out - one_hot_format

        dW2 = 1 / m * diffZ2.dot(layer1_out.T)
        db2 = 1 / m * np.sum(diffZ2)

        diffZ1 = weights2.T.dot(diffZ2) * self.ReLU_deriv(Z1)

        dW1 = 1 / m * diffZ1.dot(self.X)
        db1 = 1 / m * np.sum(diffZ1)

        return dW2, db2, dW1, db1

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        """
        Updates the weights and biases using gradient descent.
        """
        
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    def get_predictions(self, A2):
        """
        Returns the predictions based on the output of the neural network.

        Args:
        A2 (ndarray): Output of the second layer after activation.

        Returns:
        ndarray: Predicted class labels.
        """
        return np.argmax(A2, 0)
        # find the largest value in each col in matrix A2, (this will be the prediction, the highest probability it is in this class)

    def get_accuracy(self, predictions, Y):
        """
        Calculates the accuracy of the predictions.

        Args:
        predictions (ndarray): Predicted class labels.
        Y (ndarray): True class labels.

        Returns:
        float: Accuracy of the predictions.
        """
        print(predictions, "\n", Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_decent(self):
        """
        Performs gradient descent to train the neural network.
        """
        # note to self (if you assign the weights and biases as self, then you wouldn't need to return them)
        weights1, weights2, bias1, bias2 = self.initalise_parameters()
        for i in range(self.iterations):
            Z1, Z2, layer1_out, layer2_out = self.forward_propogation(weights1, weights2, bias1, bias2, self.X)
            # now perform forward and backward propagation.
            dW2, db2, dW1, db1 = self.backward_propogation(Z1, Z2, layer1_out, layer2_out, weights2)
            weights1, bias1, weight2, bias2 = self.update_params(weights1, bias1, weights2, bias2, dW1, db1, dW2, db2,
                                                                 self.alpha)

            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(layer2_out)
                print(self.get_accuracy(predictions, self.Y))

        return weights1, bias1, weights2, bias2


s = MyClassifier("data/training_spam.csv", 0.1, 100, "data/testing_spam.csv")
