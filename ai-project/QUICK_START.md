# 🚀 QUICK START GUIDE

## ✅ Your RAG System is Working!

Both servers are running and tested successfully.

## 🌐 Access Your Application

**Open your browser and go to:**
```
http://localhost:3000
```

## 📄 How to Use

1. **Upload a Document**
   - Click "Upload PDF for RAG"
   - Select a PDF, TXT, or image file
   - Wait for "Upload Successful" message

2. **Ask Questions**
   - Use the "Ask About Your Document" section
   - Type your question about the uploaded document
   - Click "Ask Document"

3. **View Results**
   - See AI-generated answers with confidence scores
   - View source chunks used for the answer
   - Copy or download results

## 🔧 If You Still Get Errors

### Option 1: Use the Easy Launcher
```bash
# Double-click this file
START_RAG_SYSTEM.bat
```

### Option 2: Manual Start
```bash
# Terminal 1 - Backend
py production_rag_server.py

# Terminal 2 - Frontend
py start_frontend.py
```

### Option 3: Test the System
```bash
py test_system.py
```

## 📋 Supported File Types

- **PDF files**: `.pdf`
- **Text files**: `.txt` 
- **Image files**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (with OCR)

## 🛠️ Troubleshooting

### "Connection Refused" Error
- Make sure both servers are running
- Check backend health: http://localhost:8002/health
- Restart servers if needed

### "File Upload Not Working"
- Use http://localhost:3000 (NOT file://)
- Clear browser cache (Ctrl+F5)
- Check file size and format

### "CORS Error"
- Ensure both servers are running
- Use the frontend server (port 3000)
- Don't open HTML directly

## 🎯 Key URLs

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8002
- **Health Check**: http://localhost:8002/health
- **System Status**: http://localhost:8002/status

## 📞 Need Help?

If you still have issues:
1. Run `py test_system.py` to diagnose
2. Check both terminals for error messages
3. Make sure ports 3000 and 8002 are not blocked

---

**🎉 Your RAG system is ready to use!**
