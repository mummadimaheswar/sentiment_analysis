"""
Pydantic schemas for API requests / responses.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ModalityResult(BaseModel):
    sentiment: str = "Neutral"
    emotion: str = "Unknown"
    confidence: float = 0.0
    details: dict = {}


class AnalysisResponse(BaseModel):
    """Unified response returned by the /analyse endpoint."""
    customer_emotion: str = "Unknown"
    sentiment: str = "Neutral"
    main_issue: str = "N/A"
    confidence: float = 0.0
    modalities_used: list[str] = []
    modality_results: dict[str, ModalityResult] = {}
    suggestions: list[str] = []


class ChatMessage(BaseModel):
    role: str = "user"            # "user" | "assistant"
    content: str = ""
    analysis: Optional[AnalysisResponse] = None
    has_audio: bool = False
    has_image: bool = False
    audio_name: Optional[str] = None
    image_name: Optional[str] = None
