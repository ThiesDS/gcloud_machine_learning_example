## Teplate for training a random forest regressor with hyperparameter tuning on gcloud ai-platform

### Usage

1. Pull repository
2. Create gcs-bucket with training data (structure see below)
3. Edit environmental variables in `submit_job.sh`:
	* Set `BUCKET_NAME` variable 
	* Set `SCALE_TIER` variable
4. execute with `sh submit_job.sh`


### Required structure of gcs-bucket

|- data
	|- data_train_features.csv
	|- data_train_labels.csv