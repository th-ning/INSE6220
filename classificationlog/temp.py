import pandas as pd

data = {
    'Model': ['lightgbm', 'xgboost', 'knn', 'nb', 'qda', 'lda', 'lr'],
    'F1': [0.9977, 0.9968, 0.9745, 0.9329, 0.9291, 0.9238, 0.4333],
    'Kappa': [0.9960, 0.9944, 0.9551, 0.8820, 0.8752, 0.8623, 0.0000],
    'MCC': [0.9960, 0.9944, 0.9552, 0.8821, 0.8753, 0.8706, 0.0000],
    'TT (Sec)': [5.137, 4.482, 4.683, 3.424, 3.434, 3.598, 3.479]
}

df = pd.DataFrame(data)

description = [
    'Session id',
    'Target',
    'Target type',
    'Original data shape',
    'Transformed data shape',
    'Transformed train set shape',
    'Transformed test set shape',
    'Numeric features',
    'Preprocess',
    'Imputation type',
    'Numeric imputation',
    'Categorical imputation',
    'Fold Generator',
    'Fold Number',
    'CPU Jobs',
    'Use GPU',
    'Log Experiment',
    'Experiment Name',
    'USI'
]

value = [
    '123',
    'cluster',
    'Multiclass',
    '(154510, 22)',
    '(154510, 22)',
    '(108157, 22)',
    '(46353, 22)',
    '21',
    'True',
    'simple',
    'mean',
    'mode',
    'StratifiedKFold',
    '10',
    '-1',
    'True',
    'False',
    'clf-default-name',
    '5afb'
]

df_description = pd.DataFrame({'Description': description, 'Value': value})

# Save the dataframes as CSV files
df.to_csv('model_metrics.csv', index=False)
df_description.to_csv('experiment_details.csv', index=False)
