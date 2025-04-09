import requests
from requests.auth import HTTPBasicAuth
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API credentials from environment variables
ASSETS_API_URL = os.getenv('API_URL', 'https://eurac.goantares.uno/public/api/v3/Assets')
ACTIVITIES_API_URL = os.getenv('ACTIVITIES_API_URL', 'https://eurac.goantares.uno/public/api/v1/Activities')
AUTH_KEY = os.getenv('AUTH_KEY')
AUTH_SECRET = os.getenv('AUTH_SECRET')

def test_activities_api():
    print("Testing Activities API...")
    
    headers = {"accept": "application/json"}
    auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)
    params = {"$top": 1}

    try:
        response = requests.get(ACTIVITIES_API_URL, headers=headers, params=params, auth=auth)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response Body: {json.dumps(response.json() if response.status_code == 200 else response.text, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_assets_api():
    print("\nTesting Assets API...")
    
    headers = {"accept": "application/json"}
    auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)
    params = {"$top": 1}

    try:
        response = requests.get(ASSETS_API_URL, headers=headers, params=params, auth=auth)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response Body: {json.dumps(response.json() if response.status_code == 200 else response.text, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_activities_api()
    test_assets_api() 