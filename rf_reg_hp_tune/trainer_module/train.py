import datetime
import argparse
import os
import sys
import pandas as pd
import hypertune
from datetime import datetime
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# Create the argument parser for each parameter plus the job directory
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument(
    '--job-dir',  # Handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
    )
parser.add_argument(
    '--n_estimators',  # Specified in the config file
    help='The number of trees in the forest',
    default=500,
    type=int
    )
args = parser.parse_args()

# Get gcs bucket information from args.job_dir argument
# Example: job_dir = 'gs://BUCKET_ID/scikit_learn_job_dir/1'
job_dir =  args.job_dir.replace('gs://', '')  # Remove the 'gs://'
# Get the Bucket Id
bucket_id = job_dir.split('/')[0]
# Get the path
bucket_path = job_dir.lstrip('{}/'.format(bucket_id))  # Example: 'scikit_learn_job_dir/1'

# Load data from bucket to filesystem
# Public bucket holding the auto mpg data
bucket = storage.Client().bucket(bucket_id)

# Path to the data inside the public bucket
blob_features = bucket.blob('data/data_train_features.csv')
blob_labels = bucket.blob('data/data_train_labels.csv')

# Download the data
blob_features.download_to_filename('data_train_features.csv')
blob_labels.download_to_filename('data_train_labels.csv')

# Load data from .csv (previously downloaded from gcs bucket)
#with open('./data_train_features.csv', 'r') as df_train_features:
features = pd.read_csv('data_train_features.csv')

#with open('./data_train_labels.csv', 'r') as df_train_labels:
labels = pd.read_csv('data_train_labels.csv')

# Preprocess data (one-hot-encoding of categorical features)
features_onehot = pd.get_dummies(features)

# Prepare feature dataset for sklearn (2D/1D arrays as input)
features_onehot = features_onehot.values
labels = labels.values.ravel()

# Train/Test split for hyperparamter tuning process
features_onehot_train, features_onehot_test, labels_train, labels_test = train_test_split(features_onehot, labels, test_size=0.20)

# Define the model
rf = RandomForestRegressor(
    n_estimators = args.n_estimators, 
    random_state = 42)

# Train the model
rf.fit(features_onehot_train, labels_train)

# Predict observations
labels_pred = rf.predict(features_onehot_test)

# Calculate the mean accuracy on the given test data and labels.
mse = mean_squared_error(labels_test, labels_pred)

# Calling the hypertune library and setting our metric
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='mean_squared_error',
    metric_value=mse,
    global_step=1000
    )

# Export the model to a file
model_filename = 'model.joblib'
joblib.dump(rf, model_filename)

# Upload the model to GCS
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob('{}/{}'.format(
    bucket_path,
    model_filename))
blob.upload_from_filename(model_filename)
