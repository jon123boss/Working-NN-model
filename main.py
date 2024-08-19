import numpy as np
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("Titanic train.csv")
df_test = pd.read_csv("Titanic test.csv")

# Preprocess the data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())  # Use direct assignment instead of inplace to avoid the FutureWarning

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y_true = df['Survived'].values.reshape(-1, 1)  # Reshape to make y_true a column vector

# Preprocess the testing data
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())  # Use direct assignment instead of inplace to avoid the FutureWarning

X_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values

# Setting up hyperparameters
input_size = 6
hidden_count = 5
hidden_size = 3
output_count = 1
learning_rate = 0.001  # Reduced learning rate

# Activation functions and their derivatives
def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    # Clip the input values to prevent overflow
    x = np.clip(x, -500, 500)  # Adjust clipping range as needed
    return 1 / (1 + np.exp(-x))

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


# Weights and biases initialization
weights = []
biases = []

# Input to first hidden layer
weights.append(np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, hidden_count)))
biases.append(np.zeros(hidden_count))

# Hidden layers
for _ in range(hidden_size - 1):
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


# Backward propagation with gradient clipping
def backward_propagation(y_true, y_pred, z_store, a_store):
    m = X.shape[0]
    delta_out = y_pred - y_true
    dW_out = np.dot(a_store[-1].T, delta_out) / m
    db_out = np.mean(delta_out, axis=0)

    # Gradient clipping
    dW_out = np.clip(dW_out, -1, 1)
    db_out = np.clip(db_out, -1, 1)

    # Update output layer weights and biases
    global W_out, b_out
    W_out -= learning_rate * dW_out
    b_out -= learning_rate * db_out

    delta = np.dot(delta_out, W_out.T) * ReLU_derivative(z_store[-1])
    for i in range(len(weights) - 1, 0, -1):
        dW = np.dot(a_store[i].T, delta) / m
        db = np.mean(delta, axis=0)

        # Gradient clipping
        dW = np.clip(dW, -1, 1)
        db = np.clip(db, -1, 1)

        weights[i] -= learning_rate * dW
        biases[i] -= learning_rate * db
        delta = np.dot(delta, weights[i].T) * ReLU_derivative(z_store[i - 1])

    dW = np.dot(a_store[0].T, delta) / m
    db = np.mean(delta, axis=0)

    # Gradient clipping
    dW = np.clip(dW, -1, 1)
    db = np.clip(db, -1, 1)

    weights[0] -= learning_rate * dW
    biases[0] -= learning_rate * db

    return compute_loss(y_true, y_pred)


# Check initial loss before training
y_pred, _, _ = forward_propagation(X)
initial_loss = compute_loss(y_true, y_pred)
print(f"Initial Loss: {initial_loss:.4f}")

# Training loop
for epoch in range(100000):
    y_pred, z_store, a_store = forward_propagation(X)
    loss = backward_propagation(y_true, y_pred, z_store, a_store)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch/thousand: {(epoch + 1)/1000}, Loss: {loss:.4f}")

y_test_pred, _, _ = forward_propagation(X_test)
y_test_pred = np.round(y_test_pred)

# Create a DataFrame with the test results
output_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_test_pred.flatten()})

# Save the predictions to a CSV file
output_df.to_csv('Titanic_predictions.csv', index=False)

ct = bool(input("Continue training(0/1):"))
while ct is True:
    for epoch in range(1000000):
        y_pred, z_store, a_store = forward_propagation(X)
        loss = backward_propagation(y_true, y_pred, z_store, a_store)
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch/thousand: {(epoch + 1)/1000}, Loss: {loss:.4f}")
    ct = bool(input("Continue training(0/1):"))

y_test_pred, _, _ = forward_propagation(X_test)
y_test_pred = np.round(y_test_pred)

# Create a DataFrame with the test results
output_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_test_pred.flatten()})

# Save the predictions to a CSV file
output_df.to_csv('Titanic_predictions.csv', index=False)