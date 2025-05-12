import pandas as pd
import matplotlib.pyplot as plt

# Function to read data and plot the first two columns
def plot_columns_separately(file_path, x, y):
    # Read the data from the file
    data = pd.read_csv(file_path)

    # Extract the first two columns (assuming they're numeric)
    x_values = data.iloc[:, x]  # First column
    y_values = data.iloc[:, y]  # Second column

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first column on the first subplot
    ax1.plot(x_values, label='X Values', color='b')
    ax1.set_title('First Column (X Values)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('X Value')
    ax1.grid(True)
    ax1.legend()

    # Plot the second column on the second subplot
    ax2.plot(y_values, label='Y Values', color='r')
    ax2.set_title('Second Column (Y Values)')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Y Value')
    ax2.grid(True)
    ax2.legend()

    # Display the plots
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def plot_first_two_columns(file_path, x, y):
    # Read the data from the file
    data = pd.read_csv(file_path)

    # Extract the first two columns (assuming they're numeric)
    x_values = data.iloc[:, x]  # First column
    y_values = data.iloc[:, y]  # Second column


    # Plot the extracted columns
    plt.plot(x_values, y_values, 'o', markersize=0.5)  # Scatter plot of first two columns
    plt.xlabel('X axis')  # Label for the X-axis
    plt.ylabel('Y axis')  # Label for the Y-axis
    plt.title('Plot of First Two Columns')  # Title of the plot
    plt.grid(True)  # Add grid
    plt.show()  # Display the plot

def count_values(file_path, x, y):
    # Read the data from the file
    data = pd.read_csv(file_path)

    # Extract the first two columns (assuming they're numeric)
    x_values = data.iloc[:, x]  # First column
    y_values = data.iloc[:, y]  # Second column

    # Count values greater than 400
    greater_than_400 = ((x_values > 400).sum(), (y_values > 400).sum())

    # Count values between 0 and 400 (exclusive)
    less_than_400 = ((x_values <= 400) & (x_values > 0)).sum(), ((y_values <= 400) & (y_values > 0)).sum()

    # Count values less than -400
    less_than_n_400 = ((x_values < -400).sum(), (y_values < -400).sum())

    # Count values between -400 and 0 (exclusive)
    greater_than_n_400 = ((x_values >= -400) & (x_values < 0)).sum(), ((y_values >= -400) & (y_values < 0)).sum()

    # Print the results
    print(f"First Column:")
    print(f"  Values > 400: {greater_than_400[0]}")
    print(f"  Values between 0 and 400: {less_than_400[0]}")
    print(f"  Values < -400: {less_than_n_400[0]}")
    print(f"  Values between -400 and 0: {greater_than_n_400[0]}")
    
    print(f"\nSecond Column:")
    print(f"  Values > 400: {greater_than_400[1]}")
    print(f"  Values between 0 and 400: {less_than_400[1]}")
    print(f"  Values < -400: {less_than_n_400[1]}")
    print(f"  Values between -400 and 0: {greater_than_n_400[1]}")
    
    # print(f"\nSecond Column:")
    # print(f"  Values > 0: {greater_than_zero[1]}")
    # print(f"  Values < 0: {less_than_zero[1]}")
    # print(f"  Values = 0: {equal_to_zero[1]}")

# Example usage:
# Replace 'your_data.csv' with the path to your CSV file
file_path = 'ce889_dataCollection.csv'
count_values(file_path, 0, 1)
count_values(file_path, 2, 3)
plot_columns_separately(file_path,0,1)
plot_columns_separately(file_path,2,3)
plot_first_two_columns(file_path,0,1)
plot_first_two_columns(file_path,2,3)
