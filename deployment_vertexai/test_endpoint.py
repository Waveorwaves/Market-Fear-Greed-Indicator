import requests
import json
import os
from google.auth.transport.requests import Request
from google.oauth2 import service_account

def get_access_token():
    """Get access token for GCP authentication"""
    try:
        # Try to get token from gcloud
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Fallback to service account if available
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'fear-greed-ml-key.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        return credentials.token
    except:
        print("❌ Could not get access token. Please run: gcloud auth login")
        return None

def test_endpoint(endpoint_url, test_data):
    """Test the Vertex AI endpoint with sample data"""
    
    access_token = get_access_token()
    if not access_token:
        return False
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            print(f"✅ Prediction successful: {predictions}")
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing endpoint: {str(e)}")
        return False

def main():
    """Main testing function"""
    
    # Replace with your actual endpoint URL
    ENDPOINT_URL = "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT/locations/us-central1/endpoints/YOUR_ENDPOINT_ID:predict"
    
    print("🧪 Testing Fear & Greed ML Endpoint")
    print("=" * 50)
    
    # Load test cases
    test_cases = [
        {
            "name": "Normal Market",
            "file": "test_normal_market.json"
        },
        {
            "name": "High Volatility", 
            "file": "test_high_volatility.json"
        },
        {
            "name": "Market Crash",
            "file": "test_market_crash.json"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        print("-" * 30)
        
        try:
            with open(test_case['file'], 'r') as f:
                test_data = json.load(f)
            
            success = test_endpoint(ENDPOINT_URL, test_data)
            results.append({
                'name': test_case['name'],
                'success': success
            })
            
        except FileNotFoundError:
            print(f"⚠️ Test file {test_case['file']} not found")
            results.append({
                'name': test_case['name'],
                'success': False
            })
    
    # Summary
    print(f"\n📈 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{result['name']}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
