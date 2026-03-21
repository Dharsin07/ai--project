"""
Start server with proper error handling
"""

import subprocess
import time
import os

def start_server():
    """Start the enhanced server with error handling"""
    
    print("🚀 STARTING ENHANCED RAG SERVER")
    print("=" * 50)
    
    server_file = "self_contained_enhanced_server.py"
    server_path = os.path.join(os.getcwd(), server_file)
    
    print(f"📄 Server file: {server_path}")
    print("🔧 Starting server...")
    
    try:
        # Start the server process
        process = subprocess.Popen(
            ["python", server_file],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully!")
            print("🌐 Server is running on: http://localhost:8001")
            print("📋 Ready for PDF, text, and image uploads")
            
            # Show server output
            try:
                while True:
                    output = process.stdout.readline()
                    if output:
                        print(f"📊 Server: {output.strip()}")
                    if process.poll() is not None:
                        break
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
        else:
            # Show error if server failed to start
            stderr_output = process.stderr.read()
            print(f"❌ Server failed to start!")
            print(f"📄 Error output: {stderr_output}")
            
            # Try to identify common issues
            if "ModuleNotFoundError" in stderr_output:
                print("🔧 Fix: Install missing modules")
                print("   pip install fastapi uvicorn python-multipart")
            elif "Permission denied" in stderr_output:
                print("🔧 Fix: Check file permissions")
            elif "Address already in use" in stderr_output:
                print("🔧 Fix: Port 8001 is already in use")
                print("   Close other applications using this port")
            else:
                print("🔧 Check the error message above for specific fix")
    
    except FileNotFoundError:
        print(f"❌ Server file not found: {server_path}")
        print("🔧 Make sure you're in the correct directory")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    start_server()
