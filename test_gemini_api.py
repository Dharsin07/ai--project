import os
import requests
import json

# Load API key from .env
GEMINI_API_KEY = None
try:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                GEMINI_API_KEY = line.split('=', 1)[1].strip()
                break
except Exception as e:
    print(f"Error reading .env file: {e}")

print(f"🔑 API Key loaded: {GEMINI_API_KEY[:20]}...{GEMINI_API_KEY[-10:] if GEMINI_API_KEY else 'None'}")

def test_gemini_api():
    """Test the Gemini API connection"""
    if not GEMINI_API_KEY:
        print("❌ No API key found")
        return False
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": "Hello! Can you respond with just 'API is working' to test the connection?"
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100,
        }
    }
    
    try:
        print("🔄 Testing API connection...")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📋 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📄 Response JSON: {json.dumps(result, indent=2)}")
            
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ API Response: {content}")
                return True
            else:
                print("❌ No content in response")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"❌ Error Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini API Connection")
    print("=" * 50)
    
    success = test_gemini_api()
    
    if success:
        print("\n✅ Gemini API is working correctly!")
    else:
        print("\n❌ Gemini API test failed!")
        print("\n🔧 Possible issues:")
        print("1. API key is invalid")
        print("2. API key has no quota")
        print("3. Network connectivity issues")
        print("4. API service is down")
