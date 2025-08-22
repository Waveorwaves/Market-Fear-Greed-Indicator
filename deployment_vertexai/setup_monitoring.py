import os
from google.cloud import aiplatform

# Initialize Vertex AI
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
REGION = "us-central1"
BUCKET_NAME = f"gs://{PROJECT_ID}-fear-greed-models"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

# Read endpoint ID from file
try:
    with open('../config/endpoint_id.txt', 'r') as f:
        endpoint_id = f.read().strip()
    print(f"Setting up monitoring for endpoint: {endpoint_id}")
except FileNotFoundError:
    print("❌ Endpoint ID not found. Run deploy_endpoint.py first.")
    exit(1)

print(f"Creating model monitoring job...")

# Create monitoring job
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="fear-greed-monitoring-job",
    endpoint=endpoint_id,
    monitoring_config={
        "model_monitoring_objective_configs": [
            {
                "training_dataset": {
                    "dataset": f"{BUCKET_NAME}/training_data_with_predictions.csv",
                    "gcs_source": {"uris": [f"{BUCKET_NAME}/training_data_with_predictions.csv"]}
                },
                "training_prediction_skew_detection_config": {
                    "skew_thresholds": {
                        "default_skew_threshold": 0.01
                    }
                },
                "prediction_drift_detection_config": {
                    "drift_thresholds": {
                        "default_drift_threshold": 0.01
                    }
                }
            }
        ],
        "monitoring_interval": 3600,  # 1 hour
        "model_monitoring_alert_config": {
            "email_alert_config": {
                "user_emails": ["your-email@example.com"]  # Replace with your email
            }
        }
    }
)

print(f"✅ Model monitoring job created!")
print(f"Monitoring Job ID: {monitoring_job.resource_name}")
print(f"📊 View monitoring dashboard at:")
print(f"https://console.cloud.google.com/vertex-ai/model-registry")

# Save monitoring job ID
with open('../config/monitoring_job_id.txt', 'w') as f:
    f.write(monitoring_job.resource_name)
