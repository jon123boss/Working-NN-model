import numpy as np

# Setting up hyperparameters  
input_size = 2
hidden_count = 10
hidden_size = 5
output_count = 1
learning_rate = 0.01


# Activation functions and their derivatives
def ReLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Example usage with dummy data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])

# Weights and biases initialization  
weights = []
biases = []

# Input to Input layer  
weights.append(np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, hidden_count)))
biases.append(np.zeros(hidden_count))

# Hidden layers  
for _ in range(hidden_size):
    weights.append(np.random.normal(0, np.sqrt(2.0 / hidden_count), (hidden_count, hidden_count)))
    biases.append(np.zeros(hidden_count))

# Last hidden layer to output  
W_out = np.random.normal(0, np.sqrt(2.0 / hidden_count), (hidden_count, output_count))
b_out = np.zeros(output_count)


def compute_loss(y_true, y_pred):
    epsilon = 1e-15  # small value to prevent log(0)  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip to prevent log(0) or log(1 - 0)  
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Forward propagation
def forward_propagation(x_data):
    z_store = []
    a_store = [x_data]
    a = x_data
    for weight_layer, bias_layer in zip(weights, biases):
        z = np.dot(a, weight_layer) + bias_layer
        z_store.append(z)
        a = ReLU(z)
        a_store.append(a)
    z = np.dot(a, W_out) + b_out
    a = sigmoid(z)
    return a, z_store, a_store


# Backward propagation
def backward_propagation(y_true, y_pred, z_store, a_store):
    m = X.shape[0]
    delta_out = y_pred - y_true
    dW_out = np.dot(a_store[-1].T, delta_out) / m
    db_out = np.mean(delta_out, axis=0)

    # Update output layer weights and biases  
    global W_out, b_out
    W_out -= learning_rate * dW_out
    b_out -= learning_rate * db_out

    delta = np.dot(delta_out, W_out.T) * ReLU_derivative(z_store[-1])
    for i in range(len(weights) - 1, 0, -1):
        dW = np.dot(a_store[i].T, delta) / m
        db = np.mean(delta, axis=0)
        weights[i] -= learning_rate * dW
        biases[i] -= learning_rate * db
        delta = np.dot(delta, weights[i].T) * ReLU_derivative(z_store[i - 1])
    dW = np.dot(X.T, delta) / m
    db = np.mean(delta, axis=0)
    weights[0] -= learning_rate * dW
    biases[0] -= learning_rate * db

    return compute_loss(y_true, y_pred)


# Training loop
for epoch in range(50000):
    y_pred, z_store, a_store = forward_propagation(X)
    loss = backward_propagation(y_true, y_pred, z_store, a_store)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss:.4f}")

    # Testing loop
while True:
    a = int(input("Enter value for A (0 or 1): "))
    b = int(input("Enter value for B (0 or 1): "))
    prediction, _, _ = forward_propagation(np.array([[a, b]]))
    print(f"Prediction for A={a}, B={b}: {prediction[0][0]:.4f}")