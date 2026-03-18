"""
Chat / Analysis API routes.
"""
from __future__ import annotations

import os
import uuid
import time
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import UPLOAD_DIR, ALLOWED_AUDIO, ALLOWED_IMAGE, MAX_UPLOAD_MB
from app.models.fusion import fuse

router = APIRouter()

# ── Lazy-loaded singletons ─────────────────────────────────────────────────
_text_analyser = None
_audio_analyser = None
_image_analyser = None


def _get_text_analyser():
    global _text_analyser
    if _text_analyser is None:
        from app.models.text_analyser import TextSentimentAnalyser
        _text_analyser = TextSentimentAnalyser()
    return _text_analyser


def _get_audio_analyser():
    global _audio_analyser
    if _audio_analyser is None:
        from app.models.audio_analyser import AudioEmotionAnalyser
        _audio_analyser = AudioEmotionAnalyser()
    return _audio_analyser


def _get_image_analyser():
    global _image_analyser
    if _image_analyser is None:
        from app.models.image_analyser import ImageSentimentAnalyser
        _image_analyser = ImageSentimentAnalyser()
    return _image_analyser


def _save_upload(file: UploadFile, allowed_exts: set[str]) -> Path:
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            400, f"Unsupported file type '{ext}'. Allowed: {allowed_exts}"
        )
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    contents = file.file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    dest.write_bytes(contents)
    return dest


# ── Health check ───────────────────────────────────────────────────────────
@router.get("/health")
def health():
    return {"status": "ok"}


# ── Main analysis endpoint ─────────────────────────────────────────────────
@router.post("/analyse")
async def analyse(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Accepts any combination of:
      - text  (form field)
      - audio (file upload – .wav / .mp3 / .flac)
      - image (file upload – .jpg / .png / .webp)
    Returns a unified multimodal sentiment report.
    """
    if not text and not audio and not image:
        raise HTTPException(400, "Provide at least one input (text, audio, or image).")

    results: dict[str, dict] = {}
    audio_path: Optional[Path] = None
    image_path: Optional[Path] = None
    start = time.time()

    try:
        # ── Text ───────────────────────────────────────────────────────
        if text and text.strip():
            analyser = _get_text_analyser()
            results["text"] = analyser.analyse(text.strip())

        # ── Audio ──────────────────────────────────────────────────────
        if audio and audio.filename:
            audio_path = _save_upload(audio, ALLOWED_AUDIO)
            analyser = _get_audio_analyser()
            results["audio"] = analyser.analyse(str(audio_path))

        # ── Image ──────────────────────────────────────────────────────
        if image and image.filename:
            image_path = _save_upload(image, ALLOWED_IMAGE)
            analyser = _get_image_analyser()
            results["image"] = analyser.analyse(str(image_path))

        if not results:
            raise HTTPException(400, "No valid input provided.")

        report = fuse(results)
        report["processing_time_ms"] = round((time.time() - start) * 1000)
        return report

    finally:
        # Clean up uploaded files
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)


# ── Pre-warm: load text model on first request ────────────────────────────
@router.post("/warmup")
def warmup():
    """Pre-load the text model so the first real request is fast."""
    _get_text_analyser()
    return {"status": "text model loaded"}
