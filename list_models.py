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

def list_available_models():
    """List all available models for the API key"""
    if not GEMINI_API_KEY:
        print("❌ No API key found")
        return False
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        print("🔄 Fetching available models...")
        response = requests.get(url, timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📄 Available models:")
            
            if 'models' in result:
                for model in result['models']:
                    name = model.get('name', 'Unknown')
                    display_name = model.get('displayName', 'Unknown')
                    description = model.get('description', 'No description')
                    
                    # Extract model name from full path
                    model_name = name.split('/')[-1] if '/' in name else name
                    
                    print(f"  🤖 {model_name}")
                    print(f"     📝 {display_name}")
                    print(f"     📄 {description}")
                    print()
                
                return True
            else:
                print("❌ No models found in response")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"❌ Error: {error_data.get('error', {}).get('message', 'Unknown error')}")
            except:
                print(f"❌ Error Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Listing Available Gemini Models")
    print("=" * 50)
    
    success = list_available_models()
    
    if success:
        print("\n✅ Successfully retrieved available models!")
    else:
        print("\n❌ Failed to retrieve models!")
        print("\n🔧 Possible issues:")
        print("1. API key is invalid")
        print("2. API key has no permissions")
        print("3. Network connectivity issues")
        print("4. API service is down")
