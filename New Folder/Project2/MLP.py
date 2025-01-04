import pandas as pd
import matplotlib.pyplot as plt

# Method to read the CSV file
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

# Method to plot the learning curve
def plot_learning_curve(data):
    plt.plot(data['Epoch'], data['Error'], label='Training Error')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main method to handle the process
def main():
    # Read data from the CSV file
    data = read_data('errors_xor.csv')

    # Plot the learning curve
    plot_learning_curve(data)

# Run the main method
if __name__ == '__main__':
    main()
