import pandas as pd
from sklearn import svm
import joblib

# Define names of datasets
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'

# Load data into pandas, then use `.values` to get NumPy arrays
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values

# Convert one-column 2D array into 1D array for use with scikit-learn
iris_target = iris_target.reshape((iris_target.size,))

# Load model
svm_model = joblib.load('gcloud_trained_models/model.joblib')
svm_model

# Predict first three instances (caution: test data equal training data!)
list(svm_model.predict(iris_data[:3]))  