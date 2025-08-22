import os
from google.cloud import aiplatform

# Initialize Vertex AI
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
REGION = "us-central1"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

# Read model ID from file
try:
    with open('../config/model_id.txt', 'r') as f:
        model_id = f.read().strip()
    print(f"Using model ID: {model_id}")
except FileNotFoundError:
    print("❌ Model ID not found. Run upload_model.py first.")
    exit(1)

# Create Model object from resource name
model = aiplatform.Model(model_id)

print(f"Creating endpoint...")

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="fear-greed-prediction-endpoint",
    project=PROJECT_ID,
    location=REGION
)

print(f"Deploying model to endpoint...")

# Deploy model
deployed_model = endpoint.deploy(
    model=model,
    deployed_model_display_name="fear-greed-deployed-model",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)

print(f"✅ Model deployed successfully!")
print(f"Endpoint ID: {endpoint.resource_name}")
print(f"Endpoint URL: {endpoint.predict}")

# Save endpoint ID for monitoring setup
with open('../config/endpoint_id.txt', 'w') as f:
    f.write(endpoint.resource_name)
