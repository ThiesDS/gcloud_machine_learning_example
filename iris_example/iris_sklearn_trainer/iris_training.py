import datetime
import os
import subprocess
import sys
import pandas as pd
from sklearn import svm
import joblib

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'gcloud_machine_learning_example_bucket'

# Download iris data from gcs bucket
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ml-engine/iris'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_data_filename),
                       iris_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_target_filename),
                       iris_target_filename], stderr=sys.stdout)


# Load data into pandas, then use `.values` to get NumPy arrays
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values

# Convert one-column 2D array into 1D array for use with scikit-learn
iris_target = iris_target.reshape((iris_target.size,))


# Train the model
classifier = svm.SVC(gamma='auto', verbose=True)
classifier.fit(iris_data, iris_target)

# Export the classifier to a file
model_filename = 'model.joblib'
joblib.dump(classifier, model_filename)


# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)