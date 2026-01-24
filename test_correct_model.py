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

def test_working_models():
    """Test the working models from the list"""
    if not GEMINI_API_KEY:
        print("❌ No API key found")
        return False
    
    # Test with models that should work for text generation
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro", 
        "gemini-flash-latest",
        "gemini-pro-latest"
    ]
    
    for model_name in models_to_try:
        print(f"\n🔄 Testing with model: {model_name}")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
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
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    print(f"✅ API Response: {content}")
                    print(f"🎉 SUCCESS: Model {model_name} works!")
                    return model_name, True
                else:
                    print("❌ No content in response")
                    print(f"📄 Response: {json.dumps(result, indent=2)}")
            else:
                print(f"❌ API Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"❌ Error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                except:
                    print(f"❌ Error Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    return None, False

if __name__ == "__main__":
    print("🧪 Testing Correct Gemini Models")
    print("=" * 50)
    
    working_model, success = test_working_models()
    
    if success:
        print(f"\n✅ Gemini API is working with model: {working_model}")
        print(f"\n🔧 Update your server to use model: {working_model}")
    else:
        print("\n❌ All Gemini API models failed!")
