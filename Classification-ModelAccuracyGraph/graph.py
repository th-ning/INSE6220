import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('model_evaluation_results.csv', nrows=7)

# Extract Model names and Accuracy data
models = df.iloc[:, 0]
accuracies = df.iloc[:, 1]

# Reverse the order of models and accuracies
models = models[::-1]
accuracies = accuracies[::-1]

# Create the plot
fig, ax = plt.subplots()
ax.barh(models, accuracies, color='darkblue')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Model')
ax.set_title('Model Accuracy')

# Annotate the accuracy values on the plot
for i, v in enumerate(accuracies):
    ax.text(v, i, f'{v:.4f}', va='center', ha='left')

# Adjust the plot layout to prevent the annotations from being clipped
plt.subplots_adjust(left=0.15, right=0.85)

# Display the plot
plt.show()
