from __future__ import print_function
import numpy as np


# Load MNIST dataset
# Load MNIST dataset
def load_data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    x_train, x_test = preprocess_data(x_train, x_test)

    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# Preprocess the data
def preprocess_data(x_train, x_test):
    # Flatten images
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Normalize pixel values to between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test


# Convert class vectors to binary class matrices
def to_categorical(y, num_classes):
    m = len(y)
    y_one_hot = np.zeros((m, num_classes))
    for i in range(m):
        y_one_hot[i, y[i]] = 1
    return y_one_hot
# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Forward propagation
def forward_propagation(X, parameters):
    # Retrieve parameters
    W1, b1, W2, b2, W3, b3 = parameters

    # Hidden layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Hidden layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # Output layer
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3


# Compute loss
def compute_loss(A3, Y):
    m = Y.shape[0]
    loss = -1/m * np.sum(Y * np.log(A3 + 1e-8))
    return loss


# Backward propagation
def backward_propagation(X, Y, parameters, caches):
    m = X.shape[0]
    W1, b1, W2, b2, W3, b3 = parameters
    Z1, A1, Z2, A2, Z3, A3 = caches

    dZ3 = (A3 - Y) / m
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = (dW1, db1, dW2, db2, dW3, db3)

    return gradients


# Update parameters
def update_parameters(parameters, gradients, learning_rate):
    W1, b1, W2, b2, W3, b3 = parameters
    dW1, db1, dW2, db2, dW3, db3 = gradients

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    parameters = (W1, b1, W2, b2, W3, b3)

    return parameters


# Train the model
# Train the model
def train_model(x_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=128, hidden_layer_size=512):
    # Get data shape and number of classes
    m, n = x_train.shape
    num_classes = len(np.unique(y_train))

    # Initialize parameters
    W1 = np.random.randn(n, hidden_layer_size) * np.sqrt(2.0 / n)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = np.random.randn(hidden_layer_size, hidden_layer_size) * np.sqrt(2.0 / hidden_layer_size)
    b2 = np.zeros((1, hidden_layer_size))
    W3 = np.random.randn(hidden_layer_size, num_classes) * np.sqrt(2.0 / hidden_layer_size)
    b3 = np.zeros((1, num_classes))

    parameters = (W1, b1, W2, b2, W3, b3)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0  # Initialize total loss for this epoch

        # Shuffle training data
        permutation = np.random.permutation(m)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, m, batch_size):
            # Get mini-batch
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Forward propagation
            caches = forward_propagation(x_batch, parameters)
            Z1, A1, Z2, A2, Z3, A3 = caches

            # Compute loss
            loss = compute_loss(A3, y_batch)

            # Accumulate total loss for this epoch
            total_loss += loss

            # Backward propagation
            gradients = backward_propagation(x_batch, y_batch, parameters, caches)

            # Update parameters
            parameters = update_parameters(parameters, gradients, learning_rate)

        # Compute average loss for this epoch
        average_loss = total_loss / (m // batch_size)
        print("Epoch {}: Loss = {}".format(epoch + 1, average_loss))

    return parameters


# Predict
def predict(x_test, parameters):
    caches = forward_propagation(x_test, parameters)
    _, _, _, _, _, A3 = caches
    predictions = np.argmax(A3, axis=1)
    return predictions


# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Train model
parameters = train_model(x_train, y_train, num_epochs=20)

# Predict on test set
predictions = predict(x_test, parameters)

# Evaluate accuracy
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print("Test Accuracy:", accuracy)
