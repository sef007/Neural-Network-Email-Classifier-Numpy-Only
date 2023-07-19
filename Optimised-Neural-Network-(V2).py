import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm


class MyClassifier:
    def __init__(self, X_train, X_val, alpha, iterations):
        """
        Initialises the MyClassifier object.

        Args:
        X_train (ndarray): Training data.
        X_val (ndarray): Validation data.
        alpha (float): Learning rate.
        iterations (int): Number of iterations for gradient descent.
        """
        self.X_train = X_train
        self.X_val = X_val
        self.alpha = alpha
        self.iterations = iterations
        self.dim = np.shape(self.X_train)

        self.Y_train = self.X_train[:, 0]
        self.Y_val = self.X_val[:, 0]

        self.weights1, self.weights2, self.bias1, self.bias2 = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialises the weights and biases for the neural network.

        Returns:
        weights1 (ndarray): Initial weights for the first layer.
        weights2 (ndarray): Initial weights for the second layer.
        bias1 (ndarray): Initial biases for the first layer.
        bias2 (ndarray): Initial biases for the second layer.
        """
        weights1 = np.random.randn(2, 55)
        weights2 = np.random.randn(2, 2)
        bias1 = np.random.randn(2, 1)
        bias2 = np.random.randn(2, 1)
        return weights1, weights2, bias1, bias2

    def rectified_linear_activation_funct(self, Z1):
        """
        Applies the rectified linear activation function to the input.

        Args:
        Z1 (ndarray): Input to the activation function.

        Returns:
        ndarray: Output after applying the activation function.
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
        return np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    def forward_propagation(self, weights1, weights2, bias1, bias2, X):
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

        Z2 = np.dot(weights2, layer1_out) + bias2
        layer2_out = self.softmax_activation_funct(Z2)

        return Z1, Z2, layer1_out, layer2_out

    def one_hot_encoding(self):
        """
        Converts the true labels to one-hot encoded form.

        Args:
            Y (numpy.ndarray): True labels.

        Returns:
            numpy.ndarray: One-hot encoded labels.
        """
        Y = self.Y_train
        num_classes = np.max(Y) + 1
        encoded = np.zeros((num_classes, Y.size))
        encoded[Y, np.arange(Y.size)] = 1
        return encoded

    def ReLU_deriv(self, Z):
        """
        calculates the derivative of the rectified linear activation function

        Args:
        Z (ndarray): Input to the activation function.

        Returns:
        ndarray: Derivative of the activation function.
        """
        return Z > 0

    def backward_propagation(self, Z1, Z2, layer1_out, layer2_out, weights2):
        """
        performs backward propagation through the neural network

        Args:
        Z1 (ndarray): Output of the first layer before activation.
        Z2 (ndarray): Output of the second layer before activation.
        layer1_out (ndarray): Output of the first layer after activation.
        layer2_out (ndarray): Output of the second layer after activation.
        weights2 (ndarray): Weights for the second layer.

        Returns:
        dW2 (ndarray): Gradient of weights2.
        db2 (ndarray): Gradient of bias2.
        dW1 (ndarray): Gradient of weights1.
        db1 (ndarray): Gradient of bias1.
        """

        one_hot_format = self.one_hot_encoding()
        m, n = self.dim

        diffZ2 = layer2_out - one_hot_format

        dW2 = 1 / m * diffZ2.dot(layer1_out.T)
        test = 1 / m * np.sum(diffZ2, axis=1)
        # keepdims true ensures that the dimensions are the same.
        db2 = 1 / m * np.sum(diffZ2, axis=1, keepdims=True)

        diffZ1 = weights2.T.dot(diffZ2) * self.ReLU_deriv(Z1)

        dW1 = 1 / m * diffZ1.dot(self.X_train)
        db1 = 1 / m * np.sum(diffZ1, axis=1, keepdims=True)

        return dW2, db2, dW1, db1

    def update_parameters(self, weights1, bias1, weights2, bias2, dW1, db1, dW2, db2, alpha):
        """
        Updates the weights and biases using gradient descent.
        """
        weights1 = weights1 - alpha * dW1
        bias1 = bias1 - alpha * db1
        weights2 = weights2 - alpha * dW2
        bias2 = bias2 - alpha * db2
        return weights1, bias1, weights2, bias2

    def get_predictions(self, A2):
        """
        Returns the predictions based on the output of the neural network.

        Args:
        A2 (ndarray): Output of the second layer after activation.

        Returns:
        ndarray: Predicted class labels.
        """
        return np.argmax(A2, axis=0)
        # Find the largest value in each col in matrix A2, this will be the prediction, the highest probability it is
        # in this class

    def get_accuracy(self, predictions, Y):
        """
        calculates the accuracy of the predictions.

        Args:
        predictions (ndarray): Predicted class labels
        Y (ndarray): True class labels.

        Returns:
        float: Accuracy of the predictions.
        """
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self):
        """
        performs gradient descent to train the neural network
        """
        weights1, weights2, bias1, bias2 = self.initialize_parameters()
        for i in range(self.iterations):
            Z1, Z2, layer1_out, layer2_out = self.forward_propagation(weights1, weights2, bias1, bias2, self.X_train)
            dW2, db2, dW1, db1 = self.backward_propagation(Z1, Z2, layer1_out, layer2_out, weights2)
            weights1, bias1, weights2, bias2 = self.update_parameters(weights1, bias1, weights2, bias2, dW1, db1, dW2, db2,
                                                                     self.alpha)

            if i % 10 == 0:
                predictions = self.get_predictions(layer2_out)
                acc = self.get_accuracy(predictions, self.Y_train)
                print("Iteration:", i, "Train Accuracy:", acc)

        # monitor the performance of the model on the validation set during training.
        _, _, _, val_output = self.forward_propagation(weights1, weights2, bias1, bias2, self.X_val)
        val_predictions = self.get_predictions(val_output)
        val_acc = self.get_accuracy(val_predictions, self.Y_val)
        print("Validation Accuracy:", val_acc)
        return weights1, bias1, weights2, bias2, val_acc


#  load the data
data = pd.read_csv("data/training_spam.csv")
data = shuffle(data, random_state=42).values

# Split the data into training, validation, and testing sets
# 20% Is being used for testing and 80% for training
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) # the dataset on the left will be used to prevent overfitting.
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# hyperparameter search
best_alpha = 0.1
best_iterations = 500
best_val_acc = 0.0

alpha_range = [0.1, 0.15, 0.2]
iterations_range = [500, 600, 700]

# The below starts the progress bar.
pbar = tqdm(total=len(alpha_range) * len(iterations_range))
for alpha in alpha_range:
    for iterations in iterations_range:
        avg_val_acc = 0.0
        runs = 10

        for _ in range(runs):
            # Create an instance of MyClassifier
            s = MyClassifier(train_data, val_data, alpha, iterations)
            weights1, bias1, weights2, bias2, val_acc = s.gradient_descent()
            avg_val_acc += val_acc

        avg_val_acc /= runs

        if avg_val_acc > best_val_acc:
            best_alpha = alpha
            best_iterations = iterations
            best_val_acc = avg_val_acc

        pbar.update(1)

# close the progress bar
pbar.close()

# train the final model with the best hyperparameters
print("BEST ALPHA: ", best_alpha)
print("BEST ITERATIONS ", best_iterations)
s = MyClassifier(train_data, val_data, best_alpha, best_iterations)
weights1, bias1, weights2, bias2, _ = s.gradient_descent()

# evaluate on the test set to ensure no over overfitting
_, _, _, test_output = s.forward_propagation(weights1, weights2, bias1, bias2, test_data)
test_predictions = s.get_predictions(test_output)
test_accuracy = s.get_accuracy(test_predictions, test_data[:, 0])
print("Test Accuracy:", test_accuracy)
