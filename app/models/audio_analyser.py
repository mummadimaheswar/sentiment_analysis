"""
Audio Emotion Analyser — Wav2Vec2 fine-tuned on speech emotions.
"""
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from app.config import AUDIO_MODEL


class AudioEmotionAnalyser:
    LABEL_MAP = {
        "angry":     ("Anger",    "Negative"),
        "calm":      ("Calm",     "Neutral"),
        "disgust":   ("Disgust",  "Negative"),
        "fearful":   ("Fear",     "Negative"),
        "happy":     ("Joy",      "Positive"),
        "neutral":   ("Neutral",  "Neutral"),
        "sad":       ("Sadness",  "Negative"),
        "surprised": ("Surprise", "Neutral"),
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("  [Audio] Loading Wav2Vec2 emotion model …")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL
        ).to(device)
        self.model.eval()

    def analyse(self, audio_path: str, sr: int = 16_000) -> dict:
        speech, _ = librosa.load(audio_path, sr=sr)
        inputs = self.feature_extractor(
            speech, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        raw_label = self.model.config.id2label[idx].lower()
        emotion, sentiment = self.LABEL_MAP.get(
            raw_label, (raw_label.title(), "Neutral")
        )
        confidence = float(probs[idx]) * 100

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "details": {
                "all_scores": {
                    self.model.config.id2label[i]: f"{p:.2%}"
                    for i, p in enumerate(probs)
                }
            },
        }
