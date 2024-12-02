import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('errors.csv')

# Plot the learning curve
plt.plot(data['Epoch'], data['Error'], label='Training Error')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()