"""
Multimodal Customer Sentiment Intelligence System
===================================================
Analyzes text, audio, and images from customer feedback using:
  - BERT/RoBERTa  → text sentiment
  - Wav2Vec2      → speech emotion detection
  - CLIP / ViT    → image sentiment
Combines NLP + Speech + Computer Vision for unified sentiment output.
"""

import os
import warnings
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np

# NLP
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
    pipeline,
)

# Audio
import librosa

# Image
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Unified result returned by each modality analyser."""
    sentiment: str = "Neutral"          # Positive / Negative / Neutral
    emotion: str = "Unknown"            # Frustration, Joy, Anger …
    confidence: float = 0.0             # 0‑100 %
    details: dict = field(default_factory=dict)


@dataclass
class MultimodalReport:
    """Final fused report across all modalities."""
    customer_emotion: str = "Unknown"
    sentiment: str = "Neutral"
    main_issue: str = "N/A"
    confidence: float = 0.0
    modality_results: dict = field(default_factory=dict)

    def display(self) -> str:
        sep = "=" * 52
        lines = [
            sep,
            "  Multimodal Customer Sentiment Intelligence Report",
            sep,
            f"  Sentiment        : {self.sentiment}",
            f"  Confidence       : {self.confidence:.0f}%",
            sep,
        ]
        for modality, result in self.modality_results.items():
            lines.append(f"\n  [{modality.upper()}]")
            lines.append(f"    Sentiment  : {result.sentiment}")
            lines.append(f"    Emotion    : {result.emotion}")
            lines.append(f"    Confidence : {result.confidence:.1f}%")
            if result.details:
                for k, v in result.details.items():
                    lines.append(f"    {k}: {v}")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. TEXT SENTIMENT ANALYSER  (RoBERTa fine‑tuned on sentiment)
# ---------------------------------------------------------------------------

class TextSentimentAnalyser:
    """Uses cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa)."""

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Map model labels → human labels
    LABEL_MAP = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
    }

    EMOTION_KEYWORDS = {
        "frustration": ["frustrated", "annoying", "terrible", "worst", "useless",
                        "broken", "waste", "horrible", "awful", "unacceptable",
                        "delay", "late", "slow", "waiting", "never arrived"],
        "anger": ["angry", "furious", "outraged", "ridiculous", "scam",
                  "rip off", "disgusted", "hate", "unforgivable"],
        "joy": ["love", "amazing", "excellent", "fantastic", "great",
                "wonderful", "happy", "delighted", "perfect", "best"],
        "sadness": ["sad", "disappointed", "let down", "unfortunate",
                    "depressing", "unhappy", "regret"],
        "surprise": ["shocked", "unexpected", "surprised", "unbelievable",
                     "wow", "incredible"],
    }

    ISSUE_KEYWORDS = {
        "Delivery delay": ["late delivery", "delay", "not arrived",
                           "hasn't arrived", "still waiting", "lost package",
                           "missing package", "wrong address", "never received",
                           "shipping delay", "dispatch", "waiting", "haven't arrived",
                           "still hasn't arrived"],
        "Product quality": ["quality", "broken", "defective", "damaged",
                            "malfunction", "cheap", "flimsy"],
        "Customer service": ["support", "service", "agent", "representative",
                             "response", "help", "rude", "unhelpful"],
        "Billing issue": ["charge", "billing", "refund", "payment",
                          "overcharged", "invoice", "money"],
        "App/Website bug": ["app", "website", "crash", "bug", "error",
                            "glitch", "login", "loading"],
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("  [Text] Loading RoBERTa sentiment model …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(device)
        self.model.eval()

    # --- helpers --------------------------------------------------------
    @staticmethod
    def _keyword_match(text: str, keyword_map: dict) -> str:
        text_lower = text.lower()
        best, best_count = "Unknown", 0
        for label, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best, best_count = label, count
        return best.title() if best_count > 0 else "Unknown"

    # --- public ---------------------------------------------------------
    def analyse(self, text: str) -> SentimentResult:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        raw_label = self.model.config.id2label[idx].lower()
        sentiment = self.LABEL_MAP.get(raw_label, raw_label.title())
        confidence = float(probs[idx]) * 100

        emotion = self._keyword_match(text, self.EMOTION_KEYWORDS)
        issue = self._keyword_match(text, self.ISSUE_KEYWORDS)

        return SentimentResult(
            sentiment=sentiment,
            emotion=emotion,
            confidence=confidence,
            details={"detected_issue": issue,
                     "all_scores": {self.model.config.id2label[i]: f"{p:.2%}"
                                    for i, p in enumerate(probs)}},
        )


# ---------------------------------------------------------------------------
# 2. AUDIO EMOTION ANALYSER  (Wav2Vec2 fine‑tuned on speech emotions)
# ---------------------------------------------------------------------------

class AudioEmotionAnalyser:
    """Uses ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition."""

    MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    LABEL_MAP = {
        "angry": ("Anger", "Negative"),
        "calm": ("Calm", "Neutral"),
        "disgust": ("Disgust", "Negative"),
        "fearful": ("Fear", "Negative"),
        "happy": ("Joy", "Positive"),
        "neutral": ("Neutral", "Neutral"),
        "sad": ("Sadness", "Negative"),
        "surprised": ("Surprise", "Neutral"),
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("  [Audio] Loading Wav2Vec2 emotion model …")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.MODEL_NAME
        )
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(device)
        self.model.eval()

    def analyse(self, audio_path: str, sr: int = 16_000) -> SentimentResult:
        speech, _ = librosa.load(audio_path, sr=sr)
        inputs = self.feature_extractor(speech, sampling_rate=sr,
                                        return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        raw_label = self.model.config.id2label[idx].lower()
        emotion, sentiment = self.LABEL_MAP.get(raw_label,
                                                 (raw_label.title(), "Neutral"))
        confidence = float(probs[idx]) * 100

        return SentimentResult(
            sentiment=sentiment,
            emotion=emotion,
            confidence=confidence,
            details={"all_scores": {self.model.config.id2label[i]: f"{p:.2%}"
                                    for i, p in enumerate(probs)}},
        )


# ---------------------------------------------------------------------------
# 3. IMAGE SENTIMENT ANALYSER  (CLIP zero‑shot)
# ---------------------------------------------------------------------------

class ImageSentimentAnalyser:
    """Uses openai/clip-vit-base-patch32 for zero‑shot image sentiment."""

    MODEL_NAME = "openai/clip-vit-base-patch32"

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
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()

    def _zero_shot(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, images=image,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits_per_image.cpu().numpy()[0]
        return np.exp(logits) / np.exp(logits).sum()   # softmax

    def analyse(self, image_path: str) -> SentimentResult:
        image = Image.open(image_path).convert("RGB")

        # Sentiment classification
        sent_probs = self._zero_shot(image, self.SENTIMENT_PROMPTS)
        best_idx = int(np.argmax(sent_probs))
        sentiment = self.PROMPT_LABELS[best_idx]
        emotion = self.PROMPT_EMOTIONS[best_idx]
        confidence = float(sent_probs[best_idx]) * 100

        # Issue classification
        issue_probs = self._zero_shot(image, self.ISSUE_PROMPTS)
        issue_idx = int(np.argmax(issue_probs))
        issue = self.ISSUE_LABELS[issue_idx]

        return SentimentResult(
            sentiment=sentiment,
            emotion=emotion,
            confidence=confidence,
            details={
                "detected_issue": issue,
                "sentiment_scores": {l: f"{p:.2%}"
                                     for l, p in zip(self.PROMPT_LABELS, sent_probs)},
                "issue_scores": {l: f"{p:.2%}"
                                 for l, p in zip(self.ISSUE_LABELS, issue_probs)},
            },
        )


# ---------------------------------------------------------------------------
# 4. MULTIMODAL FUSION ENGINE
# ---------------------------------------------------------------------------

class MultimodalFusionEngine:
    """Fuses results from text, audio, and image analysers with weighted voting."""

    # Weights per modality (text is typically most informative for issues)
    DEFAULT_WEIGHTS = {"text": 0.50, "audio": 0.25, "image": 0.25}

    SENTIMENT_PRIORITY = {"Negative": 2, "Neutral": 1, "Positive": 0}

    def fuse(self, results: dict[str, SentimentResult],
             weights: Optional[dict[str, float]] = None) -> MultimodalReport:
        weights = weights or self.DEFAULT_WEIGHTS

        # --- Weighted sentiment vote ---
        sentiment_scores: dict[str, float] = {}
        confidence_sum = 0.0
        total_weight = 0.0

        for modality, result in results.items():
            w = weights.get(modality, 0.25)
            sentiment_scores[result.sentiment] = (
                sentiment_scores.get(result.sentiment, 0.0) + w
            )
            confidence_sum += result.confidence * w
            total_weight += w

        # Pick sentiment with highest vote; break ties toward Negative
        final_sentiment = max(
            sentiment_scores,
            key=lambda s: (sentiment_scores[s],
                           self.SENTIMENT_PRIORITY.get(s, 0)),
        )
        fused_confidence = confidence_sum / total_weight if total_weight else 0

        # --- Emotion: prefer text emotion if available, else majority ---
        emotions = [r.emotion for r in results.values() if r.emotion != "Unknown"]
        final_emotion = max(set(emotions), key=emotions.count) if emotions else "Unknown"

        # --- Issue: prefer text‑detected issue ---
        main_issue = "N/A"
        if "text" in results:
            main_issue = results["text"].details.get("detected_issue", "N/A")
        if main_issue in ("N/A", "Unknown") and "image" in results:
            main_issue = results["image"].details.get("detected_issue", "N/A")

        return MultimodalReport(
            customer_emotion=final_emotion,
            sentiment=final_sentiment,
            main_issue=main_issue,
            confidence=fused_confidence,
            modality_results=results,
        )


# ---------------------------------------------------------------------------
# 5. ORCHESTRATOR – ties everything together
# ---------------------------------------------------------------------------

class SentimentIntelligenceSystem:
    """
    High‑level API.  Supply any combination of text / audio_path / image_path
    and get a unified MultimodalReport.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*52}")
        print("  Initialising Multimodal Sentiment Intelligence")
        print(f"  Device: {self.device}")
        print(f"{'='*52}\n")

        self.text_analyser: Optional[TextSentimentAnalyser] = None
        self.audio_analyser: Optional[AudioEmotionAnalyser] = None
        self.image_analyser: Optional[ImageSentimentAnalyser] = None
        self.fusion = MultimodalFusionEngine()

    # Lazy loading – only load models that are actually needed
    def _ensure_text(self):
        if self.text_analyser is None:
            self.text_analyser = TextSentimentAnalyser(self.device)

    def _ensure_audio(self):
        if self.audio_analyser is None:
            self.audio_analyser = AudioEmotionAnalyser(self.device)

    def _ensure_image(self):
        if self.image_analyser is None:
            self.image_analyser = ImageSentimentAnalyser(self.device)

    def analyse(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> MultimodalReport:
        results: dict[str, SentimentResult] = {}

        if text:
            self._ensure_text()
            print("\n  Analysing text …")
            results["text"] = self.text_analyser.analyse(text)

        if audio_path:
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            self._ensure_audio()
            print("  Analysing audio …")
            results["audio"] = self.audio_analyser.analyse(audio_path)

        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            self._ensure_image()
            print("  Analysing image …")
            results["image"] = self.image_analyser.analyse(image_path)

        if not results:
            raise ValueError("Provide at least one input: text, audio_path, or image_path.")

        print("  Fusing modality results …\n")
        return self.fusion.fuse(results)


# ---------------------------------------------------------------------------
# 6. DEMO / CLI
# ---------------------------------------------------------------------------

def demo_text_only():
    """Quick demo using only text input."""
    system = SentimentIntelligenceSystem()

    reviews = [
        "I've been waiting 3 weeks for my delivery and it still hasn't arrived. "
        "This is absolutely unacceptable! Your customer support was no help at all.",

        "The product quality is amazing, I love everything about it! "
        "Delivery was fast and the packaging was perfect.",

        "The app keeps crashing every time I try to make a payment. "
        "I'm frustrated and considering switching to a competitor.",
    ]

    for i, review in enumerate(reviews, 1):
        print(f"\n{'#'*52}")
        print(f"  REVIEW {i}")
        print(f"{'#'*52}")
        print(f'  "{review[:90]}…"' if len(review) > 90 else f'  "{review}"')
        report = system.analyse(text=review)
        print(report.display())


def main():
    """
    Entry point. Run with:
        python sentimentanaysis.py                     # text‑only demo
        python sentimentanaysis.py --text "..."        # custom text
        python sentimentanaysis.py --audio call.wav    # audio file
        python sentimentanaysis.py --image photo.jpg   # image file
    All flags can be combined for true multimodal analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Multimodal Customer Sentiment Intelligence System"
    )
    parser.add_argument("--text", type=str, default=None,
                        help="Customer review text to analyse")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (.wav) of customer call")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (.jpg/.png) from social media")
    parser.add_argument("--demo", action="store_true",
                        help="Run built‑in text demo with sample reviews")

    args = parser.parse_args()

    # If nothing specified, run demo
    if not any([args.text, args.audio, args.image]) or args.demo:
        demo_text_only()
        return

    system = SentimentIntelligenceSystem()
    report = system.analyse(
        text=args.text,
        audio_path=args.audio,
        image_path=args.image,
    )
    print(report.display())


if __name__ == "__main__":
    main()
