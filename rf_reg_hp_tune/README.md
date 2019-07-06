## Template for training a random forest regressor with hyperparameter tuning on gcloud ai-platform

### Usage

1. Pull repository
2. Create gcs-bucket with training data (structure see below)
3. Execute `sh submit_job BUCKET_NAME SCALE_TIER`

### Required structure of gcs-bucket

|- data \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- data_train_features.csv \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- data_train_labels.csv