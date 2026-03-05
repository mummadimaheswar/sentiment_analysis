"""
Image Sentiment Analyser — CLIP zero-shot classification.
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from app.config import IMAGE_MODEL


class ImageSentimentAnalyser:

    SENTIMENT_PROMPTS = [
        "a photo showing a very happy and satisfied customer",
        "a photo showing a neutral customer experience",
        "a photo showing an angry and frustrated customer",
    ]
    PROMPT_LABELS = ["Positive", "Neutral", "Negative"]
    PROMPT_EMOTIONS = ["Joy", "Neutral", "Frustration"]

    ISSUE_PROMPTS = [
        "a photo of a damaged or broken product",
        "a photo of a late or missing delivery package",
        "a photo of a billing or payment error on a screen",
        "a photo of poor customer service interaction",
        "a photo of a normal product in good condition",
    ]
    ISSUE_LABELS = [
        "Product quality", "Delivery delay", "Billing issue",
        "Customer service", "No issue detected",
    ]

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("  [Image] Loading CLIP model …")
        self.processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
        self.model = CLIPModel.from_pretrained(IMAGE_MODEL).to(device)
        self.model.eval()

    def _zero_shot(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=prompts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits_per_image.cpu().numpy()[0]
        return np.exp(logits) / np.exp(logits).sum()

    def analyse(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")

        sent_probs = self._zero_shot(image, self.SENTIMENT_PROMPTS)
        best_idx = int(np.argmax(sent_probs))
        sentiment = self.PROMPT_LABELS[best_idx]
        emotion = self.PROMPT_EMOTIONS[best_idx]
        confidence = float(sent_probs[best_idx]) * 100

        issue_probs = self._zero_shot(image, self.ISSUE_PROMPTS)
        issue_idx = int(np.argmax(issue_probs))
        issue = self.ISSUE_LABELS[issue_idx]

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "details": {
                "detected_issue": issue,
                "sentiment_scores": {
                    l: f"{p:.2%}"
                    for l, p in zip(self.PROMPT_LABELS, sent_probs)
                },
                "issue_scores": {
                    l: f"{p:.2%}"
                    for l, p in zip(self.ISSUE_LABELS, issue_probs)
                },
            },
        }
