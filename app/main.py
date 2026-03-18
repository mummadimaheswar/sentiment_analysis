"""
FastAPI application entry point.
"""
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routes.chat import router as chat_router

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(
    title="Sentiment Intelligence",
    description="Multimodal Customer Sentiment Intelligence System",
    version="1.0.0",
)

# ── API routes ─────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api")

# ── Serve static files (CSS, JS, assets) ──────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Serve the SPA (single-page app) ───────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))
