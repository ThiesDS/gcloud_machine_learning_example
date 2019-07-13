import os
import datetime
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/sventhies/service_accounts/sven_uses_google_vision_api.json"

# Define bucket name
bucket_name = 'zd-ai-platform-performanceattribution-training-bucket'
# Build job_dir from bucket name
job_dir =  bucket_name + str('/job_dir')
# Get the Bucket Id
bucket_id = job_dir.split('/')[0]
# Get the path
bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

# Load data from bucket to filesystem
bucket = storage.Client().bucket(bucket_id)

# Path to the data inside the public bucket
blob_features = bucket.blob('data/data_train_features.csv')
blob_labels = bucket.blob('data/data_train_labels.csv')

# Download the data
blob_features.download_to_filename('data_train_features.csv')
blob_labels.download_to_filename('data_train_labels.csv')

# Load data from .csv (previously downloaded from gcs bucket)
features = pd.read_csv('data_train_features.csv')
labels = pd.read_csv('data_train_labels.csv')

# Preprocess data (one-hot-encoding of categorical features)
features_onehot = pd.get_dummies(features)

# Prepare feature dataset for sklearn (2D/1D arrays as input)
features_onehot = features_onehot.values
labels = labels.values.ravel()

# Train/Test split for hyperparamter tuning process
features_onehot_train, features_onehot_test, labels_train, labels_test = train_test_split(features_onehot, labels, test_size=0.20)

# Define the model
rf = RandomForestRegressor()

# Define grid to search over
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 300, num = 2)],
               'max_depth': [int(x) for x in np.linspace(10, 20, num = 2)]}

# Perform randomized search with cross validation to tune hyperparameters
rf_randomCV = RandomizedSearchCV(
	estimator = rf, 
	param_distributions = random_grid, 
	scoring = 'neg_mean_squared_error',
	n_iter = 30, 
	cv = 3, 
	random_state=42, 
	n_jobs = -1)

# Train the model
rf_randomCV.fit(features_onehot_train, labels_train)

# Predict observations
labels_pred = rf_randomCV.predict(features_onehot_test)

# Calculate the mean accuracy on the given test data and labels.
mse = mean_squared_error(labels_test, labels_pred)

# Export the model to a file
model_filename = 'model.joblib'
joblib.dump(rf_randomCV, model_filename)

# Upload the model to GCS
bucket_path=job_dir+datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
#bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob('{}/{}'.format(
    bucket_path,
    model_filename))
blob.upload_from_filename(model_filename)
