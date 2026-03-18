"""
Text Sentiment Analyser — RoBERTa fine-tuned on sentiment.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import TEXT_MODEL


class TextSentimentAnalyser:
    LABEL_MAP = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

    EMOTION_KEYWORDS = {
        "frustration": [
            "frustrated", "annoying", "terrible", "worst", "useless",
            "broken", "waste", "horrible", "awful", "unacceptable",
            "delay", "late", "slow", "waiting", "never arrived",
        ],
        "anger": [
            "angry", "furious", "outraged", "ridiculous", "scam",
            "rip off", "disgusted", "hate", "unforgivable",
        ],
        "joy": [
            "love", "amazing", "excellent", "fantastic", "great",
            "wonderful", "happy", "delighted", "perfect", "best",
        ],
        "sadness": [
            "sad", "disappointed", "let down", "unfortunate",
            "depressing", "unhappy", "regret",
        ],
        "surprise": [
            "shocked", "unexpected", "surprised", "unbelievable",
            "wow", "incredible",
        ],
    }

    ISSUE_KEYWORDS = {
        "Delivery delay": [
            "late delivery", "delay", "not arrived", "hasn't arrived",
            "still waiting", "lost package", "missing package",
            "wrong address", "never received", "shipping delay",
            "dispatch", "waiting", "haven't arrived", "still hasn't arrived",
        ],
        "Product quality": [
            "quality", "broken", "defective", "damaged",
            "malfunction", "cheap", "flimsy",
        ],
        "Customer service": [
            "support", "service", "agent", "representative",
            "response", "help", "rude", "unhelpful",
        ],
        "Billing issue": [
            "charge", "billing", "refund", "payment",
            "overcharged", "invoice", "money",
        ],
        "App/Website bug": [
            "app", "website", "crash", "bug", "error",
            "glitch", "login", "loading",
        ],
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("  [Text] Loading RoBERTa sentiment model …")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            TEXT_MODEL
        ).to(device)
        self.model.eval()

    @staticmethod
    def _keyword_match(text: str, keyword_map: dict) -> str:
        text_lower = text.lower()
        best, best_count = "Unknown", 0
        for label, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best, best_count = label, count
        return best.title() if best_count > 0 else "Unknown"

    def analyse(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        raw_label = self.model.config.id2label[idx].lower()
        sentiment = self.LABEL_MAP.get(raw_label, raw_label.title())
        confidence = float(probs[idx]) * 100

        emotion = self._keyword_match(text, self.EMOTION_KEYWORDS)
        issue = self._keyword_match(text, self.ISSUE_KEYWORDS)

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "details": {
                "detected_issue": issue,
                "all_scores": {
                    self.model.config.id2label[i]: f"{p:.2%}"
                    for i, p in enumerate(probs)
                },
            },
        }
