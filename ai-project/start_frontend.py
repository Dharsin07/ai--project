#!/usr/bin/env python3
"""
Simple Frontend Server for RAG Application
Serves the frontend HTML/JS/CSS files to avoid file:// protocol issues
"""

import os
import http.server
import socketserver
import webbrowser
import sys
from pathlib import Path

def start_frontend_server(port=3000):
    """Start a simple HTTP server for the frontend"""
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Find available Python command
    python_cmd = None
    for cmd in ["py", "python", "python3"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            python_cmd = cmd
            break
        except:
            continue
    
    if not python_cmd:
        print("❌ Python not found. Please install Python.")
        sys.exit(1)
    
    print("🚀 Starting Frontend Server...")
    print("=" * 50)
    print(f"📁 Serving files from: {project_dir}")
    print(f"🌐 Frontend URL: http://localhost:{port}")
    print(f"🔗 Backend URL: http://localhost:8002")
    print("=" * 50)
    print("📝 Press Ctrl+C to stop the server")
    print()
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"✅ Frontend server running on port {port}")
            
            # Auto-open browser after a short delay
            import threading
            import time
            
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f"http://localhost:{port}")
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Please try a different port.")
            print(f"💡 Try: py start_frontend.py --port 3001")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start frontend server for RAG application")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the frontend server on")
    
    args = parser.parse_args()
    
    start_frontend_server(args.port)
