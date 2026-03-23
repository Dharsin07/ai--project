# RAG System Setup Guide

## Quick Start (Recommended)

### Option 1: One-Click Start (Windows)
1. Double-click `START_RAG_SYSTEM.bat`
2. Wait for both servers to start
3. Browser will automatically open to `http://localhost:3000`

### Option 2: Command Line Start
```bash
python start_rag_system.py
```

## Manual Start (Advanced)

### Step 1: Start Backend Server
```bash
python production_rag_server.py
```
Backend will run on: `http://localhost:8002`

### Step 2: Start Frontend Server (in separate terminal)
```bash
python start_frontend.py
```
Frontend will run on: `http://localhost:3000`

### Step 3: Open Browser
Navigate to: `http://localhost:3000`

## Fixing Connection Issues

### Problem: ERR_CONNECTION_REFUSED
**Solution**: Ensure backend is running on port 8002
```bash
# Check if backend is running
curl http://localhost:8002/health

# If not running, start it:
python production_rag_server.py
```

### Problem: CORS Errors
**Solution**: Backend CORS is already configured. If you still get CORS errors:
1. Check backend is running on the correct port (8002)
2. Clear browser cache and hard refresh (Ctrl+F5)
3. Ensure you're accessing frontend via `http://localhost:3000` not `file://`

### Problem: File Upload Not Working
**Solution**: 
1. Make sure you're using the frontend server (`http://localhost:3000`)
2. Don't open the HTML file directly (`file://`)
3. Check backend health: `http://localhost:8002/health`

## File Upload Support

The system supports:
- **PDF files**: `.pdf`
- **Text files**: `.txt`
- **Image files**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (with OCR)

## API Endpoints

### Backend (http://localhost:8002)
- `POST /rag/upload` - Upload document
- `POST /rag/query` - Query uploaded document
- `GET /health` - Health check
- `GET /status` - System status
- `DELETE /rag/document` - Delete current document

### Frontend (http://localhost:3000)
- Serves the web interface
- Handles file uploads and queries

## Troubleshooting

### Port Already in Use
If you get "Port already in use" errors:

**Option 1**: Kill existing processes
```bash
# Windows
netstat -ano | findstr :8002
taskkill /PID <PID> /F

netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Option 2**: Use different ports
```bash
# Backend on port 8003
python production_rag_server.py  # Edit the port in the file

# Frontend on port 3001
python start_frontend.py --port 3001
```

### Backend Not Responding
1. Check Python dependencies: `pip install -r requirements.txt`
2. Check backend logs for errors
3. Verify health endpoint: `http://localhost:8002/health`

### Frontend Not Loading
1. Ensure frontend server is running
2. Check you're using `http://localhost:3000` not `file://`
3. Clear browser cache

## Development Notes

### File Structure
```
project/
├── production_rag_server.py     # Main backend server
├── start_frontend.py           # Frontend server
├── start_rag_system.py         # Complete system launcher
├── START_RAG_SYSTEM.bat        # Windows batch file
├── index.html                  # Frontend HTML
├── script.js                   # Frontend JavaScript
├── style.css                   # Frontend CSS
└── requirements.txt            # Python dependencies
```

### Configuration
- **Backend Port**: 8002 (configurable in `production_rag_server.py`)
- **Frontend Port**: 3000 (configurable via command line)
- **CORS**: Configured for localhost development

### Security Notes
- This setup is for development only
- CORS allows all origins (`*`) for development
- In production, restrict CORS to specific domains
- Use HTTPS in production

## Support

If you encounter issues:
1. Check both servers are running
2. Verify ports are not blocked by firewall
3. Check browser console for errors
4. Ensure all Python dependencies are installed

## Next Steps

Once the system is running:
1. Upload a PDF document using the web interface
2. Wait for processing to complete
3. Ask questions about your document
4. View AI-generated answers with source citations
