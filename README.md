# README

# **Neural-Network-Scratch-HAM-SPAM-(V1)**

## **Description:**

This program implements a neural network classifier from scratch to classify emails as either HAM or SPAM. It uses a custom implementation of a feedforward neural network with a rectified linear activation function (ReLU) in the hidden layer and a softmax activation function in the output layer. The model is trained using gradient descent with backpropagation.

## **Key Features:**

- One-hot encoding of target labels to prepare them for classification.
- Random initialisation of weights and biases.
- Rectified linear activation function (ReLU) in the hidden layer.
- Softmax activation function in the output layer.
- Forward propagation to compute the output of the neural network.
- Backward propagation to compute the gradients of the weights and biases.
- Update of parameters using gradient descent.
- Prediction of classes based on the highest probability in the output layer.
- Calculation of accuracy to evaluate the model's performance.

## **Usage:**

1. Instantiate the **`MyClassifier`** class by providing the input data, learning rate (alpha), number of iterations, and target labels.
2. The program initialises the parameters, performs forward and backward propagation, and updates the parameters iteratively.
3. During training, the program displays the iteration number and the training accuracy every 10 iterations.
4. After training, the program evaluates the model on a test set and prints the test accuracy.

# **Optimised-Neural-Network-(V2)**

## **Description:**

This program implements an optimised version of a neural network classifier for email classification (HAM or SPAM). It uses the **`sklearn`** library for data preprocessing and model evaluation, improving efficiency and performance compared to the previous version (V1).

mention hyper parameter search space 

## **Key Features:**

- Random initialisation of weights and biases.
- Rectified linear activation function (ReLU) in the hidden layer.
- Softmax activation function in the output layer.
- Forward propagation to compute the output of the neural network.
- One-hot encoding of target labels to prepare them for classification.
- Backward propagation to compute the gradients of the weights and biases.
- Update of parameters using gradient descent.
- Prediction of classes based on the highest probability in the output layer.
- Calculation of accuracy to evaluate the model's performance.
- Splitting the data into training, validation, and testing sets using **`train_test_split`**.
- Hyperparameter search to find the best combination of learning rate (alpha) and number of iterations.
- Progress bar display during the hyperparameter search process.

## **Usage:**

1. Load the input data from a CSV file using **`pd.read_csv`** and shuffle the data using **`shuffle`** function.
2. Split the data into training, validation, and testing sets using **`train_test_split`** function.
3. Define a range of values for learning rate (alpha) and number of iterations to search for the best hyperparameters.
4. Instantiate the **`MyClassifier`** class by providing the training and validation data, learning rate (alpha), and number of iterations.
5. The program initialises the parameters, performs forward and backward propagation, and updates the parameters iteratively. It displays the training accuracy every 10 iterations.
6. After training, the program evaluates the model on the test set and prints the test accuracy.