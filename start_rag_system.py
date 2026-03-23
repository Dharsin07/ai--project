#!/usr/bin/env python3
"""
Complete RAG System Startup Script
Starts both backend and frontend servers simultaneously
"""

import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path
import signal
import os

class RAGSystemLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_dir = Path(__file__).parent
        self.running = True
        
    def start_backend(self):
        """Start the RAG backend server"""
        print("🔧 Starting Backend Server...")
        
        backend_script = self.project_dir / "simple_rag_server.py"
        
        try:
            # Start backend process
            self.backend_process = subprocess.Popen(
                ["py", str(backend_script)],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor backend output
            def monitor_backend():
                while self.running and self.backend_process:
                    output = self.backend_process.stdout.readline()
                    if output:
                        print(f"[Backend] {output.strip()}")
                    elif self.backend_process.poll() is not None:
                        break
            
            backend_thread = threading.Thread(target=monitor_backend)
            backend_thread.daemon = True
            backend_thread.start()
            
            # Wait for backend to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print("✅ Backend server started successfully on http://localhost:8002")
                return True
            else:
                print("❌ Backend server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend server"""
        print("🎨 Starting Frontend Server...")
        
        frontend_script = self.project_dir / "start_frontend.py"
        
        try:
            # Start frontend process
            self.frontend_process = subprocess.Popen(
                ["py", str(frontend_script), "--port", "3000"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor frontend output
            def monitor_frontend():
                while self.running and self.frontend_process:
                    output = self.frontend_process.stdout.readline()
                    if output:
                        print(f"[Frontend] {output.strip()}")
                    elif self.frontend_process.poll() is not None:
                        break
            
            frontend_thread = threading.Thread(target=monitor_frontend)
            frontend_thread.daemon = True
            frontend_thread.start()
            
            # Wait for frontend to start
            time.sleep(2)
            
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully on http://localhost:3000")
                return True
            else:
                print("❌ Frontend server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def open_browser(self):
        """Open browser after both servers are ready"""
        time.sleep(3)
        if self.running:
            print("🌐 Opening browser...")
            webbrowser.open("http://localhost:3000")
    
    def shutdown(self, signum=None, frame=None):
        """Gracefully shutdown all processes"""
        print("\n🛑 Shutting down RAG system...")
        self.running = False
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend server stopped")
            except:
                try:
                    self.backend_process.kill()
                    print("✅ Backend server force killed")
                except:
                    pass
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend server stopped")
            except:
                try:
                    self.frontend_process.kill()
                    print("✅ Frontend server force killed")
                except:
                    pass
        
        print("👋 RAG system shutdown complete")
        sys.exit(0)
    
    def run(self):
        """Run the complete RAG system"""
        print("🚀 Starting Complete RAG System...")
        print("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # Check if required files exist
        backend_script = self.project_dir / "simple_rag_server.py"
        frontend_script = self.project_dir / "start_frontend.py"
        
        if not backend_script.exists():
            print(f"❌ Backend script not found: {backend_script}")
            return False
        
        if not frontend_script.exists():
            print(f"❌ Frontend script not found: {frontend_script}")
            return False
        
        # Start backend
        if not self.start_backend():
            return False
        
        # Start frontend
        if not self.start_frontend():
            self.shutdown()
            return False
        
        print("=" * 60)
        print("🎉 RAG System is now running!")
        print("📍 Frontend: http://localhost:3000")
        print("📍 Backend:  http://localhost:8002")
        print("📍 Health Check: http://localhost:8002/health")
        print("=" * 60)
        print("📝 Press Ctrl+C to stop both servers")
        print()
        
        # Open browser in separate thread
        browser_thread = threading.Thread(target=self.open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend server crashed!")
                    self.shutdown()
                    break
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend server crashed!")
                    self.shutdown()
                    break
                    
        except KeyboardInterrupt:
            self.shutdown()
        
        return True

if __name__ == "__main__":
    launcher = RAGSystemLauncher()
    success = launcher.run()
    
    if not success:
        print("❌ Failed to start RAG system")
        sys.exit(1)
    else:
        print("✅ RAG system stopped successfully")
