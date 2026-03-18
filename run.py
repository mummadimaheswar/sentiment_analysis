"""
Run the Sentiment Intelligence web application.

Usage:
    python run.py
    
Then open http://127.0.0.1:8000 in your browser.
"""
import uvicorn
from app.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    print("\n🧠 Sentiment Intelligence — starting server …")
    print(f"   Open  http://{HOST}:{PORT}  in your browser\n")
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
    )
