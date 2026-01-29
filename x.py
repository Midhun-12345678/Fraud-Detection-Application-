import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Sample transaction features (30 features from credit card dataset)
sample_features = [
    -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443,
    -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507,
    0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348,
    -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
    0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
    -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731,
    0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215,
    149.62, 0
]

print("=" * 50)
print("Testing Fraud Detection API")
print("=" * 50)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 2: Predict with JSON
print("\n2. Testing /predict-json endpoint...")
payload = {"features": sample_features}
response = requests.post(f"{BASE_URL}/predict-json", json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 3: Predict with Form data
print("\n3. Testing /predict endpoint (form data)...")
data = {"features": json.dumps(sample_features)}
response = requests.post(f"{BASE_URL}/predict", data=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 4: Dashboard
print("\n4. Testing /dashboard endpoint...")
response = requests.get(f"{BASE_URL}/dashboard")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 5: Transaction History
print("\n5. Testing /history endpoint...")
response = requests.get(f"{BASE_URL}/history")
print(f"Status: {response.status_code}")
print(f"Number of transactions: {len(response.json())}")
if response.json():
    print(f"Latest transaction: {response.json()[0]}")

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)