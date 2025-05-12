import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rmse_array_for_test = []
rmse_array_for_train = []

# Sigmoid activation function
def sigmoid(x):
    lamb = 0.8
    return 1 / (1 + np.exp(-x * lamb))

# Derivative of Sigmaoit activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Mean squared error loss for end of the feed forward
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivative of Mean squared error loss for the back propagation
def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Normalization of data
def normalization(first_column_max_value, first_column_min_value, second_column_max_value, second_column_min_value, first_column_value, second_column_value):
    data = []
    for i in range(len(first_column_value)):
        X1 = (first_column_value[i] - first_column_min_value) / (first_column_max_value - first_column_min_value)
        X2 = (second_column_value[i] - second_column_min_value) / (second_column_max_value - second_column_min_value)
        data.append([X1, X2])
    return data

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    # RMSE score calculation
    def RMSE(self, testX, testY):
        output = self.forward_for_RMSE(testX)
        error1 = np.sqrt(np.mean((testY[:,0] - output[:,0]) ** 2))
        error2 = np.sqrt(np.mean((testY[:,1] - output[:,1]) ** 2))
        return (error1 + error2) / 2
        
    # Feed forward
    def forward(self, X):
        self.input = X
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        
        self.output = sigmoid(self.output_input)
        
        return self.output
    
    # To calculate RMSE in each epoch this function is used.
    # Only difference between forward is it does not updates weights
    def forward_for_RMSE(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = sigmoid(output_input)
        return output

    # Back propagation
    def backward(self, X, y):

        # Backward pass
        output_error = mse_loss_derivative(y, self.output) * sigmoid_derivative(self.output_input)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_input)

        # Gradients for weights and biases
        weights_hidden_output_gradient = np.dot(self.hidden_output.T, output_error)
        bias_output_gradient = np.sum(output_error, axis=0, keepdims=True)
        weights_input_hidden_gradient = np.dot(X.T, hidden_error)
        bias_hidden_gradient = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * weights_hidden_output_gradient
        self.bias_output -= self.learning_rate * bias_output_gradient
        self.weights_input_hidden -= self.learning_rate * weights_input_hidden_gradient
        self.bias_hidden -= self.learning_rate * bias_hidden_gradient

    # Training
    def train(self, TrainX, TrainY, TestX, TestY, epochs):
        for epoch in range(epochs):
            for i in range(len(TrainX)):

                # Reshape the inputs for begin to feed forward
                tx = TrainX[i].reshape(1, -1)
                ty = TrainY[i].reshape(1, -1)
                
                # Perform forward pass
                output = self.forward(tx)
                
                # Calculate loss
                loss = mse_loss(ty, output)
                
                # Perform backward pass
                self.backward(tx, ty)

            # Compute RMSE with forward
            rmse_array_for_test.append(self.RMSE(TestX, TestY))
            rmse_array_for_train.append(self.RMSE(TrainX, TrainY))

            # Logging and early stopping after each 100 epoch
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss}")
                print("self.weights_input_hidden = ", self.weights_input_hidden.tolist())
                print("self.weights_hidden_output = ", np.transpose(self.weights_hidden_output).tolist())
                print("self.bias_hidden = ", self.bias_hidden.tolist())
                print("self.bias_output =", self.bias_output.tolist())
                print("RMSE:", rmse_array_for_test[-1])
            if epoch > 1:
                if rmse_array_for_test[-1] >= rmse_array_for_test[-2]:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    print("weights_input_hidden", self.weights_input_hidden)
                    print("weights_hidden_output", np.transpose(self.weights_hidden_output))
                    print("bias_hidden", self.bias_hidden)
                    print("bias_output", self.bias_output)
                    print("RMSE:", rmse_array_for_test[-1])
                    print("OVERFIT!!!!")
                    break

# Example usage
if __name__ == "__main__":

    # Read data
    file_path = 'ce889_dataCollection.csv'
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)

    # Shuffle the data
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Read each column
    first_column = shuffled_df.iloc[:, 0].tolist()
    second_column = shuffled_df.iloc[:, 1].tolist()
    output1_column = shuffled_df.iloc[:, 2].tolist()
    output2_column = shuffled_df.iloc[:, 3].tolist()

    # Find min and max of inputs to normalize
    first_column_max_value = max(first_column)
    second_column_max_value = max(second_column)
    first_column_min_value = min(first_column)
    second_column_min_value = min(second_column)

    # Find min and max of outputs to normalize
    output1_column_min_value = min(output1_column)
    output2_column_min_value = min(output2_column)
    output1_column_max_value = max(output1_column)
    output2_column_max_value = max(output2_column)

    # Normalized inputs
    input_layer = normalization(first_column_max_value, first_column_min_value, second_column_max_value, second_column_min_value, first_column, second_column)

    # Normalize outputs
    expected_outputs = normalization(output1_column_max_value, output1_column_min_value, output2_column_max_value, output2_column_min_value, output1_column, output2_column)

    # Convert to numpy arrays
    X = np.array(input_layer)
    y = np.array(expected_outputs)

    # split the data
    split_index = int(0.70 * len(shuffled_df))
    trainX = X[:split_index]  # 70% for training
    trainY = y[:split_index]  # 70% for training
    testX = X[split_index:]  # 30% for testing
    testY = y[split_index:]  # 30% for testing

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=2, learning_rate=0.8)
    nn.train(trainX, trainY, testX, testY, epochs=100)

    # Prediction data for test
    y_predicted = nn.forward(testX)

    # Test the network
    print("Predictions:")
    print(y_predicted)
    print("expected_outputs:")
    print(testY)
    print(mse_loss(testY[:,0], y_predicted[:,0]))
    print(mse_loss(testY[:,1], y_predicted[:,1]))

    # show the RMSE graph for training and testing at the end of the training
    def show_rmse():
        plt.figure(figsize=(8, 6))
        plt.plot(rmse_array_for_test, color="blue")
        plt.plot(rmse_array_for_train, color="red")

        plt.title("Plot of My Function", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("f(x)", fontsize=14)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.grid(True)

        plt.show()

    # Plot prediction and actuals of X and y 
    def plot_predictions_vs_actual(predictions_column, actual_column):
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_column[::100], label='Predictions', linestyle='--', linewidth=1)
        plt.plot(actual_column[::100], label='Actual', linestyle='-', linewidth=1)
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title('Predictions vs Actual Data')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    show_rmse()
    plot_predictions_vs_actual(y_predicted[:,0],testY[:,0])
    plot_predictions_vs_actual(y_predicted[:,1],testY[:,1])
    


