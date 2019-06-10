#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


## 0. Parameter

# Folder and conda environment name
FOLDER_NAME="gcloud_machine_learning_example"

# Bucket name
BUCKET_NAME="gcloud_machine_learning_example_bucket"

# Region
REGION="europe-west1"

# gcloud config account
CONFIG_ACCOUNT="private"


## 1. Setup

# Create conda environment
conda create -n $FOLDER_NAME python 

# Activate conda environment
conda activate $FOLDER_NAME

# Install required packages and frameworks
pip install numpy scipy pandas scikit-learn joblib

# Set gcloud accout
gcloud config configurations activate $CONFIG_ACCOUNT


## 2. Create GCS Bucket

# Create bucket
gsutil mb -l $REGION gs://$BUCKET_NAME


## 3. Build python training module

# Create file
touch iris_training.py

# Fill isis_training.py with the following steps
# (Example: https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/sklearn/iris_training.py)
# - Load data (e.g. from gcs bucket) into pandas
# - train model (e.g. using scikit-learn)
# - Export model as model.joblib file
# - Load model in separat folder to to gcs bucket that you created above


# 4. Create training application package

# Create empty folder
mkdir iris_sklearn_trainer

# Create init file
touch iris_sklearn_trainer/__init__.py

# Move python training module to this folder
mv iris_training.py iris_sklearn_trainer/


# 5. Training the model

# Set environmental variables
BUCKET_NAME=$BUCKET_NAME
JOB_NAME="iris_scikit_learn_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_NAME/scikit_learn_job_dir
TRAINING_PACKAGE_PATH="./iris_sklearn_trainer/"
MAIN_TRAINER_MODULE="iris_sklearn_trainer.iris_training"
REGION=$REGION
RUNTIME_VERSION=1.12
PYTHON_VERSION=2.7
SCALE_TIER=BASIC

# Execute local training job
gcloud ai-platform local train \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE

# Send training job to cloud
gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER


# 7. Download model to local filesystem

# Get object from gcs bucket
gsutil cp gs://$BUCKET_NAME/iris_*/model.joblib .

# Create python script to use gcloud trained model object and make predictions
touch iris_predict_local.py




