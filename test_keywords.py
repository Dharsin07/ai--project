import requests
import json

response = requests.post('http://127.0.0.1:8001/research', json={
    "topic": "machine learning", 
    "summary_type": "short", 
    "source_count": 2
})

if response.status_code == 200:
    data = response.json()
    print("Keywords:", data.get('keywords', []))
    print("Topic:", data.get('topic'))
    print("Summary preview:", data.get('summary', '')[:100] + "...")
else:
    print("Error:", response.text)
