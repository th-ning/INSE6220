import pandas as pd
import numpy as np
from pycaret.classification import *
import os
import glob
import matplotlib.pyplot as plt
import shap


def plot_classification(data, labels, save_path):
    plt.figure()
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', s=30, edgecolor='k')
    plt.colorbar(scatter)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Set the input and output directories
input_dir_original = r'C:\Users\123456\Documents\PCA project\FinalMatrix(All)'
input_dir_clustered = r'C:\Users\123456\Documents\PCA project\ClusterPeaks（GFP)（FinalAll)'
output_dir = r'C:\Users\123456\Documents\PCA project\Classification(All_PCA_GFP)'

# Loop through each folder
for folder_name in os.listdir(input_dir_original):
    # Read the original data from the current folder
    folder_path_original = os.path.join(input_dir_original, folder_name)
    original_data_files = glob.glob(os.path.join(folder_path_original, '*.csv'))
    original_data = pd.concat([pd.read_csv(file, index_col=0) for file in original_data_files])

    # Read the clustered data from the corresponding folder
    folder_path_clustered = os.path.join(input_dir_clustered, folder_name)
    clustered_data_files = glob.glob(os.path.join(folder_path_clustered, '*.csv'))
    clustered_data = pd.concat([pd.read_csv(file, index_col=0) for file in clustered_data_files])

    # Split the clustered_data into train and test
    train_data = clustered_data.sample(frac=0.8, random_state=786)
    test_data = clustered_data.drop(train_data.index)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(train_data.shape))
    print('Unseen Data For Predictions: ' + str(test_data.shape))

    # Create the output folder if it doesn't exist
    folder_path_output = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path_output, exist_ok=True)

    # Save train and test data to CSV files
    train_data.to_csv(os.path.join(folder_path_output, 'train_data.csv'), index=True)
    test_data.to_csv(os.path.join(folder_path_output, 'test_data.csv'), index=True)

    # Set up the classification model
    clf = setup(data=clustered_data, target='cluster', train_size=0.7, session_id=123, use_gpu=True)

    # Create and tune models
    best_model = compare_models()

    # Evaluate the best model
    evaluate_model(best_model)

    # Generate the confusion matrix plot for LinearDiscriminantAnalysis
    '''plot_model(best_model, plot='confusion_matrix', use_train_data=False)
    fig = plt.gcf()
    fig.savefig(os.path.join(folder_path_output, 'confusion_matrix_lda.png'), bbox_inches='tight')
    plt.close()'''

    # Perform model evaluation
    model_results = pull()
    model_results.to_csv(os.path.join(folder_path_output, 'model_evaluation_results.csv'), index=False)

    # Calculate SHAP values and test data
    explainer = shap.Explainer(best_model)
    test_X = get_config('X_test')
    shap_values = explainer(test_X)

    # Save the SHAP Summary Plot to an image file
    '''shap.summary_plot(shap_values, test_X, feature_names=test_X.columns, show=False)
    plt.savefig(os.path.join(folder_path_output, 'shap_summary_plot_best_model.png'), bbox_inches='tight')
    plt.close()'''

    # Make predictions on the original data
    original_data_predictions = predict_model(best_model, data=original_data)
    print(original_data_predictions.columns)

    # Create a new DataFrame with the same index as original_data
    predicted_classes = pd.DataFrame(index=original_data.index)

    # Add the 'Label' column from original_data_predictions to the new DataFrame
    predicted_classes['class'] = original_data_predictions['prediction_label']

    # Concatenate original_data with predicted_classes
    original_data_with_predictions = pd.concat([original_data, predicted_classes], axis=1)

    # Save the DataFrame with predictions to a CSV file
    output_filename = f"{folder_name}_All_PCA_GFP_Classification.csv"
    original_data_with_predictions.to_csv(os.path.join(folder_path_output, output_filename), index=True)

    # Plot and save the classification plot
    plot_classification(original_data_with_predictions.iloc[:, :2].values,
                        original_data_with_predictions['class'].values,
                        os.path.join(folder_path_output, f"{folder_name}_classification_plot.png"))

