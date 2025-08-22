#!/bin/bash

# Set your GCP project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_NAME="gs://${PROJECT_ID}-fear-greed-models"

# Create GCS bucket
gsutil mb -l ${REGION} ${BUCKET_NAME}

# Upload model files
gsutil cp ../models/xgboost_model.pkl ${BUCKET_NAME}/
gsutil cp ../models/scaler.pkl ${BUCKET_NAME}/
gsutil cp ../models/feature_cols.pkl ${BUCKET_NAME}/
gsutil cp ../src/predict.py ${BUCKET_NAME}/

# Upload training data for monitoring
gsutil cp ../data/training_data_with_predictions.csv ${BUCKET_NAME}/

echo "Files uploaded to GCS bucket: ${BUCKET_NAME}"
echo "Next steps:"
echo "1. Run python upload_model.py"
echo "2. Run python deploy_endpoint.py"
echo "3. Run python setup_monitoring.py"
