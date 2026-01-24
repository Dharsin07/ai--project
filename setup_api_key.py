# Simple script to help you set up your Gemini API key

import os

def setup_api_key():
    print("🔑 Gemini API Key Setup")
    print("=" * 40)
    print()
    print("1. Get your API key from: https://makersuite.google.com/app/apikey")
    print("2. Copy the API key")
    print("3. Paste it below when prompted")
    print()
    
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    
    print("✅ API key saved to .env file!")
    print()
    print("📝 Next steps:")
    print("1. Restart the server: py gemini_server.py")
    print("2. The server will now use real Gemini API responses")
    print("3. Upload a PDF and ask questions to get AI-powered answers")
    
    return True

if __name__ == "__main__":
    setup_api_key()
