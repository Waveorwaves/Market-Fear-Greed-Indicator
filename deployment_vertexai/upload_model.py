import os
from google.cloud import aiplatform

# Initialize Vertex AI
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
REGION = "us-central1"
BUCKET_NAME = f"gs://{PROJECT_ID}-fear-greed-models"

print(f"Uploading model to Vertex AI...")
print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET_NAME
)

# Upload model to Vertex AI
model = aiplatform.Model.upload(
    display_name="fear-greed-xgboost-model",
    artifact_uri=BUCKET_NAME,
    serving_container_image_uri="us-docker.pkg.dev/cloud-aiplatform/prediction/xgboost-cpu.1-7:latest",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_ports=[8080]
)

print(f"✅ Model uploaded successfully!")
print(f"Model ID: {model.resource_name}")
print(f"Model name: {model.display_name}")

# Save model ID for later use
with open('../config/model_id.txt', 'w') as f:
    f.write(model.resource_name)
