import pandas as pd
import numpy as np
from pycaret.classification import *
import os
import matplotlib.pyplot as plt
import shap

# Read the data
original_data_path = 'P01_Merged_All_final_matrix.csv'
clustered_data_path = 'P01_Merged_All_final_matrix_peaks_clustered.csv'

original_data = pd.read_csv(original_data_path, index_col=0)
clustered_data = pd.read_csv(clustered_data_path, index_col=0, nrows=201)

# Split the clustered_data into train and test
train_data = clustered_data.sample(frac=0.8, random_state=786)
test_data = clustered_data.drop(train_data.index)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(train_data.shape))
print('Unseen Data For Predictions: ' + str(test_data.shape))

# Save train and test data to CSV files
train_data.to_csv('train_data.csv', index=True)
test_data.to_csv('test_data.csv', index=True)

# Set up the classification model
clf = setup(data=train_data, target='cluster', train_size=0.7, session_id=123, use_gpu=True)

# Create and tune models
best_model = compare_models()

# Evaluate the best model
evaluate_model(best_model)

# Calculate SHAP values and test data
explainer = shap.Explainer(best_model)
test_X = get_config('X_test')
shap_values = explainer(test_X)

# Save the SHAP Summary Plot to an image file
shap.summary_plot(shap_values, test_X, feature_names=test_X.columns, show=False)
plt.savefig('shap_summary_plot_best_model.png', bbox_inches='tight')
plt.close()

# Make predictions on the original data (first 201 rows)
original_data_to_predict = original_data.iloc[:201]
original_data_predictions = predict_model(best_model, data=original_data_to_predict)
print(original_data_predictions.columns)

# Create a new DataFrame with the same index as original_data
predicted_classes = pd.DataFrame(index=original_data_to_predict.index)

# Add the 'Label' column from original_data_predictions to the new DataFrame
predicted_classes['class'] = original_data_predictions['prediction_label']

# Concatenate original_data with predicted_classes
original_data_with_predictions = pd.concat([original_data, predicted_classes], axis=1)

# Save the DataFrame with predictions to a CSV file
original_data_with_predictions.to_csv('original_data_with_predictions.csv', index=True)
