import numpy as np
import os

# Define the paths to the files
labels_path = os.path.abspath("solutions/private_solution_easy.npy")
predictions_path = os.path.abspath("predictions/predictions_easy.npy")

# Check if the files exist
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found: {labels_path}")
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

# Load the data
labels = np.load(labels_path)
predictions = np.load(predictions_path)

# Calculate the score
correct = 0
for i in range(len(labels)):
    correct += (predictions[i] == labels[i]).sum().item()

print(f'score: {100 * correct / len(labels)}%')
