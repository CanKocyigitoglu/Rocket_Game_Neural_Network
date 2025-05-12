import pandas as pd
import math
import numpy as np

file_path = 'ce889_dataCollection.csv'

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

        # read csv for min max values of each column
        df = pd.read_csv(file_path)

        # initialize columns
        first_column = df.iloc[:, 0].tolist()
        second_column = df.iloc[:, 1].tolist()
        output1_column = df.iloc[:, 2].tolist()
        output2_column = df.iloc[:, 3].tolist()

        # Find max and min of inputs for normalization
        self.first_column_max_value = max(first_column)
        self.second_column_max_value = max(second_column)
        self.first_column_min_value = min(first_column)
        self.second_column_min_value = min(second_column)

        # Find max and min of inputs for denormalization
        self.output1_column_min_value = min(output1_column)
        self.output2_column_min_value = min(output2_column)
        self.output1_column_max_value = max(output1_column)
        self.output2_column_max_value = max(output2_column)

        # Give the trained weights to Neural Network
        self.weights_input_hidden =  [[-12.674228746750183, 40.41137718781614, -37.60412687367112, 0.4361284082796585, 6.567722860906312], [2.425978899158578, 1.768729949596471, 2.9332180582393756, -13.618316880394065, 3.090292001151955]]
        self.weights_hidden_output =  [[-0.7301613016236757, -1.397216555161206, -1.5051822268755197, -0.9059766916053494, -1.0444756001138162], [4.826231504516578, 0.9704966735657774, -0.9126111386624778, 0.008482798486692835, -2.6428807000645884]]
        self.bias_hidden =  [[-3.6498804490406584, -22.773303196006697, 16.856456108501295, -0.5395729004251948, -9.898000098729351]]
        self.bias_output = [[0.663248076014699, 0.1550492541779815]]
    
    def predict(self, input_row):

        # Sigmoid activation function and its derivative
        def sigmoid(x, lamb):
            return 1 / (1 + np.exp(-x * lamb))
        
        # Feedforward
        def forward(self, X):
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            self.hidden_output = sigmoid(hidden_input, 0.8)
            self.output_input = np.dot(self.hidden_output, np.transpose(self.weights_hidden_output)) + self.bias_output
            self.output = sigmoid(self.output_input, 0.8)
            return self.output
        
        # Input_row preprocessing
        values = input_row.split(',')

        input_X = float(values[0])
        input_Y = float(values[1])

        # Normalization
        normalized_X = (input_X - self.first_column_min_value) / (self.first_column_max_value - self.first_column_min_value)
        normalized_Y = (input_Y - self.second_column_min_value) / (self.second_column_max_value - self.second_column_min_value)
        input_layer = [normalized_X, normalized_Y]

        # Prediction
        o = forward(self, input_layer)

        # Denormalization
        normal_output1 = o[0][0]
        normal_output2 = o[0][1]

        denormalized_output1 = (self.output1_column_max_value - self.output1_column_min_value) * o[0][0] + self.output1_column_min_value
        denormalized_output2 = (self.output2_column_max_value - self.output2_column_min_value) * o[0][1] + self.output2_column_min_value
        
        # y and x gives true prediction
        return [denormalized_output2, denormalized_output1] 

