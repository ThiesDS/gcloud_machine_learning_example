#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Customizable environmental variables
BUCKET_NAME="zd-ai-platform-performanceattribution-training-bucket"
JOB_NAME="rf_training$(date +"%Y%m%d_%H%M%S")"
SCALE_TIER=BASIC



# Fixed environment variables (DO NOT CHANGE)
JOB_DIR=gs://$BUCKET_NAME/job_dir
TRAINING_PACKAGE_PATH="./trainer_module/"
MAIN_TRAINER_MODULE="trainer_module.train"
REGION="europe-west1"
RUNTIME_VERSION=1.12
PYTHON_VERSION=3.5

# Send training job to the cloud
gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER