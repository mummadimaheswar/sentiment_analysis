"""
Application configuration & constants.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Model names (HuggingFace Hub) ─────────────────────────────────────────
TEXT_MODEL = os.getenv(
    "TEXT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
AUDIO_MODEL = os.getenv(
    "AUDIO_MODEL", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL", "openai/clip-vit-base-patch32"
)

# ── Server ─────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

# ── Allowed upload extensions ──────────────────────────────────────────────
ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
MAX_UPLOAD_MB = 25
