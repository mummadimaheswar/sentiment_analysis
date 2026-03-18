"""
Multimodal Fusion Engine — merges results from all modalities.
"""
from __future__ import annotations
from typing import Optional


SENTIMENT_PRIORITY = {"Negative": 2, "Neutral": 1, "Positive": 0}
DEFAULT_WEIGHTS = {"text": 0.50, "audio": 0.25, "image": 0.25}

# Auto-generated actionable suggestions per issue
SUGGESTIONS = {
    "Delivery delay": [
        "Apologise for the delay and provide a tracking update.",
        "Offer expedited re-shipping or a discount on the next order.",
        "Escalate to logistics team for investigation.",
    ],
    "Product quality": [
        "Initiate a replacement or full refund immediately.",
        "Forward case to the quality-assurance team.",
        "Send a personalised apology with a goodwill voucher.",
    ],
    "Customer service": [
        "Review the interaction and coach the agent involved.",
        "Follow up personally with the customer.",
        "Offer a service-recovery credit.",
    ],
    "Billing issue": [
        "Verify the charge and issue a correction / refund.",
        "Send an updated invoice or receipt.",
        "Escalate to finance team for review.",
    ],
    "App/Website bug": [
        "Log a high-priority bug ticket for engineering.",
        "Provide a workaround or alternative way to complete the action.",
        "Notify the customer once the fix is deployed.",
    ],
}


def fuse(
    results: dict[str, dict],
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """Fuse modality dicts into a single unified report dict."""
    weights = weights or DEFAULT_WEIGHTS

    sentiment_scores: dict[str, float] = {}
    confidence_sum = 0.0
    total_weight = 0.0

    for modality, result in results.items():
        w = weights.get(modality, 0.25)
        s = result["sentiment"]
        sentiment_scores[s] = sentiment_scores.get(s, 0.0) + w
        confidence_sum += result["confidence"] * w
        total_weight += w

    final_sentiment = max(
        sentiment_scores,
        key=lambda s: (sentiment_scores[s], SENTIMENT_PRIORITY.get(s, 0)),
    )
    fused_confidence = confidence_sum / total_weight if total_weight else 0.0

    emotions = [r["emotion"] for r in results.values() if r["emotion"] != "Unknown"]
    final_emotion = (
        max(set(emotions), key=emotions.count) if emotions else "Unknown"
    )

    main_issue = "N/A"
    if "text" in results:
        main_issue = results["text"].get("details", {}).get("detected_issue", "N/A")
    if main_issue in ("N/A", "Unknown") and "image" in results:
        main_issue = results["image"].get("details", {}).get("detected_issue", "N/A")

    suggestions = SUGGESTIONS.get(main_issue, [])

    return {
        "customer_emotion": final_emotion,
        "sentiment": final_sentiment,
        "main_issue": main_issue,
        "confidence": round(fused_confidence, 2),
        "modalities_used": list(results.keys()),
        "modality_results": results,
        "suggestions": suggestions,
    }
