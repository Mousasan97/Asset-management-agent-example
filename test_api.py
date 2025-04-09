import requests
from requests.auth import HTTPBasicAuth
import json

def test_activities_api():
    print("Testing Activities API...")
    API_URL = "https://eurac.goantares.uno/public/api/v1/Activities"
    AUTH_KEY = "b069c270-9622-478d-8bb2-6778a3d953f6"
    AUTH_SECRET = "b57853f8-346b-4cf0-923d-30020705a784"

    headers = {"accept": "application/json"}
    auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)
    params = {"$top": 1}

    try:
        response = requests.get(API_URL, headers=headers, params=params, auth=auth)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response Body: {json.dumps(response.json() if response.status_code == 200 else response.text, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_assets_api():
    print("\nTesting Assets API...")
    API_URL = "https://eurac.goantares.uno/public/api/v3/Assets"
    AUTH_KEY = "b069c270-9622-478d-8bb2-6778a3d953f6"
    AUTH_SECRET = "b57853f8-346b-4cf0-923d-30020705a784"

    headers = {"accept": "application/json"}
    auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)
    params = {"$top": 1}

    try:
        response = requests.get(API_URL, headers=headers, params=params, auth=auth)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response Body: {json.dumps(response.json() if response.status_code == 200 else response.text, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_activities_api()
    test_assets_api() 