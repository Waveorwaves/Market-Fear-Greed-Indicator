# 🚀 Fear & Greed ML Deployment Guide

This guide provides step-by-step instructions for deploying the Fear & Greed ML model to Google Cloud Vertex AI.

## 📋 Prerequisites

### Required Tools
- **Python 3.9+**
- **Google Cloud CLI** ([Install Guide](https://cloud.google.com/sdk/docs/install))
- **Git**

### Required GCP Services
- Vertex AI API
- Cloud Storage API
- IAM API

## 🔧 Step 1: Environment Setup

### Install Google Cloud CLI
```bash
# Windows: Download installer from https://cloud.google.com/sdk/docs/install
# Mac: brew install google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# Verify installation
gcloud --version
```

### Initialize GCP Configuration
```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (replace with your actual project)
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Set default region
gcloud config set compute/region us-central1

# Verify configuration
gcloud config list
```

### Enable Required APIs
```bash
# Enable necessary Google Cloud APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable iam.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled --filter="name:(aiplatform OR storage)"
```

## 🔐 Step 2: Authentication Setup

### Create Service Account
```bash
# Create service account for ML operations
gcloud iam service-accounts create fear-greed-ml-sa \
    --display-name="Fear & Greed ML Service Account" \
    --description="Service account for ML model deployment"

# Get your project ID
PROJECT_ID=$(gcloud config get-value project)
```

### Grant Required Permissions
```bash
# Grant Vertex AI permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fear-greed-ml-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.admin"

# Grant Cloud Storage permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fear-greed-ml-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Grant service account user role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fear-greed-ml-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

### Create Service Account Key
```bash
# Create and download service account key
gcloud iam service-accounts keys create fear-greed-ml-key.json \
    --iam-account=fear-greed-ml-sa@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable for authentication
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/fear-greed-ml-key.json"

# Verify authentication
gcloud auth list
```

## 📦 Step 3: Prepare Model Files

### Required Files Checklist
Ensure you have these files in the correct directories:

```
models/
├── xgboost_model.pkl     # Trained XGBoost model
├── scaler.pkl            # Feature standardization scaler  
└── feature_cols.pkl      # Feature column definitions

data/
├── fear_greed_dataset.csv                  # Training dataset
└── training_data_with_predictions.csv      # Baseline for monitoring

config/
└── monitoring_schema.json                  # Feature schema (116 features)
```

### Copy Files to Correct Locations
```bash
# Copy model files (you need to do this manually)
# From parent directory:
cp model.pkl models/xgboost_model.pkl
cp scaler.pkl models/
cp feature_cols.pkl models/

# Copy data files
cp fear_greed_dataset.csv data/
cp training_data_with_predictions.csv data/

# Copy configuration
cp monitoring_schema.json config/
```

## ☁️ Step 4: Cloud Storage Setup

### Create GCS Bucket
```bash
# Set variables
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="${PROJECT_ID}-fear-greed-models"
REGION="us-central1"

# Create bucket
gsutil mb -l $REGION gs://$BUCKET_NAME

# Verify bucket creation
gsutil ls gs://$BUCKET_NAME
```

### Upload Model Artifacts
```bash
# Run the deployment script
bash deployment/deploy.sh

# Or manually upload files:
gsutil cp models/xgboost_model.pkl gs://$BUCKET_NAME/
gsutil cp models/scaler.pkl gs://$BUCKET_NAME/
gsutil cp models/feature_cols.pkl gs://$BUCKET_NAME/
gsutil cp src/predict.py gs://$BUCKET_NAME/
gsutil cp data/training_data_with_predictions.csv gs://$BUCKET_NAME/

# Verify uploads
gsutil ls -la gs://$BUCKET_NAME/
```

## 🤖 Step 5: Deploy Model to Vertex AI

### Upload Model to Model Registry
```bash
# Update PROJECT_ID in the upload script
sed -i "s/your-project-id/$PROJECT_ID/g" deployment/upload_model.py

# Run model upload
python deployment/upload_model.py
```

**Expected Output:**
```
✅ Model uploaded successfully!
Model ID: projects/your-project-id/locations/us-central1/models/1234567890
Model name: fear-greed-xgboost-model
```

### Deploy Model to Endpoint
```bash
# Run endpoint deployment
python deployment/deploy_endpoint.py
```

**Expected Output:**
```
✅ Model deployed successfully!
Endpoint ID: projects/your-project-id/locations/us-central1/endpoints/9876543210
Endpoint URL: https://us-central1-aiplatform.googleapis.com/v1/projects/...
```

## 📊 Step 6: Setup Model Monitoring

### Configure Monitoring
```bash
# Update email in monitoring script
sed -i "s/your-email@example.com/your-actual-email@domain.com/g" deployment/setup_monitoring.py

# Run monitoring setup
python deployment/setup_monitoring.py
```

**Expected Output:**
```
✅ Model monitoring job created!
Monitoring Job ID: projects/your-project-id/locations/us-central1/modelDeploymentMonitoringJobs/...
📊 View monitoring dashboard at: https://console.cloud.google.com/vertex-ai/model-registry
```

## 🧪 Step 7: Test Deployment

### Test Local Flask API (Optional)
```bash
# Start local Flask server
cd src/
python app.py

# Test in another terminal
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @../tests/test_normal_market.json
```

### Test Vertex AI Endpoint
```bash
# Update endpoint URL in test script
# Edit tests/test_endpoint.py and replace YOUR_PROJECT and YOUR_ENDPOINT_ID

# Run endpoint tests
python tests/test_endpoint.py
```

**Expected Output:**
```
🧪 Testing Fear & Greed ML Endpoint
==================================================

📊 Testing: Normal Market
------------------------------
Status Code: 200
✅ Prediction successful: [18.45]

📊 Testing: High Volatility
------------------------------
Status Code: 200
✅ Prediction successful: [28.32]

📊 Testing: Market Crash
------------------------------
Status Code: 200
✅ Prediction successful: [42.18]

📈 Test Results Summary
==================================================
Normal Market: ✅ PASS
High Volatility: ✅ PASS
Market Crash: ✅ PASS

Overall: 3/3 tests passed
```

## 📈 Step 8: Verify Deployment

### Check Model Registry
```bash
# List deployed models
gcloud ai models list --region=us-central1

# Get model details
gcloud ai models describe MODEL_ID --region=us-central1
```

### Check Endpoints
```bash
# List endpoints
gcloud ai endpoints list --region=us-central1

# Get endpoint details
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Check Monitoring Jobs
```bash
# List monitoring jobs
gcloud ai model-deployment-monitoring-jobs list --region=us-central1

# Get monitoring job details
gcloud ai model-deployment-monitoring-jobs describe JOB_ID --region=us-central1
```

## 🎯 Step 9: Make Predictions

### Using gcloud CLI
```bash
# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Get endpoint URL
ENDPOINT_URL=$(gcloud ai endpoints list --region=us-central1 --format="value(predictHttpUri)")

# Make prediction
curl -X POST $ENDPOINT_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d @tests/test_normal_market.json
```

### Using Python Script
```python
from google.cloud import aiplatform
import json

# Initialize Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

# Get endpoint
endpoint = aiplatform.Endpoint("projects/your-project-id/locations/us-central1/endpoints/YOUR_ENDPOINT_ID")

# Load test data
with open('tests/test_normal_market.json', 'r') as f:
    test_data = json.load(f)

# Make prediction
response = endpoint.predict(instances=test_data['instances'])
print(f"Prediction: {response.predictions}")
```

## 🔍 Step 10: Monitor and Maintain

### View Monitoring Dashboard
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Vertex AI > Model Registry**
3. Select your model: `fear-greed-xgboost-model`
4. Click on **Monitoring** tab
5. View drift detection, skew analysis, and alerts

### Set Up Alerts
The monitoring system will automatically send email alerts when:
- Data drift exceeds 1% threshold
- Prediction skew exceeds 1% threshold
- Model performance degrades significantly

### View Logs
```bash
# View endpoint logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" --limit=50

# View model monitoring logs
gcloud logging read "resource.type=aiplatform.googleapis.com/ModelDeploymentMonitoringJob" --limit=50
```

## 💰 Cost Management

### Estimated Monthly Costs
- **Vertex AI Endpoint**: $360-720 (n1-standard-2, 24/7)
- **Model Monitoring**: $72-144 (hourly monitoring)
- **Cloud Storage**: $2-5 (model artifacts)
- **Predictions**: $0.001 per prediction

### Cost Optimization Tips
```bash
# Scale down during off-hours
gcloud ai endpoints update ENDPOINT_ID \
  --region=us-central1 \
  --min-replica-count=0 \
  --max-replica-count=1

# Set billing alerts
# Go to: https://console.cloud.google.com/billing
# Create budgets with alerts at $50, $100, $200
```

## 🚨 Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:fear-greed-ml-sa"
```

#### Model Upload Failures
```bash
# Check bucket contents
gsutil ls -la gs://$BUCKET_NAME/

# Verify file sizes
gsutil ls -l gs://$BUCKET_NAME/xgboost_model.pkl

# Check upload permissions
gsutil iam get gs://$BUCKET_NAME
```

#### Prediction Errors
```bash
# Check endpoint status
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1

# View prediction logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" \
  --filter="severity>=ERROR" --limit=20
```

#### Feature Count Mismatches
- Ensure you're using the correct `feature_cols.pkl` file
- Verify test data has all 114 required features
- Check feature names match exactly (case-sensitive)

### Getting Help

1. **Check logs first**: Most issues are logged in Cloud Logging
2. **Verify permissions**: Ensure service account has required roles
3. **Test locally**: Use Flask API to debug model issues
4. **Check quotas**: Verify you haven't exceeded GCP quotas

## ✅ Success Checklist

- [ ] Google Cloud CLI installed and configured
- [ ] Project and APIs enabled
- [ ] Service account created with proper permissions
- [ ] Model files uploaded to GCS bucket
- [ ] Model uploaded to Vertex AI Model Registry
- [ ] Endpoint created and model deployed
- [ ] Model monitoring configured
- [ ] Test predictions working
- [ ] Monitoring dashboard accessible
- [ ] Email alerts configured

## 🎉 Next Steps

1. **Integration**: Connect the endpoint to your trading system
2. **Automation**: Set up CI/CD pipeline for model updates
3. **Scaling**: Configure auto-scaling based on traffic
4. **Monitoring**: Set up custom metrics and dashboards
5. **Retraining**: Implement automated model retraining pipeline

---

**🚀 Congratulations! Your Fear & Greed ML model is now deployed and ready for production use!**

For ongoing support, refer to the [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs).
