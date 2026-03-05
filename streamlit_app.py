"""
🧠 Multimodal Customer Sentiment Intelligence — Streamlit App
==============================================================
A ChatGPT / Gemini-style chatbot for analysing customer sentiment
across text, audio, and images.

Run:
    streamlit run streamlit_app.py
"""

import os
import io
import time
import tempfile
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark ChatGPT / Gemini theme ──────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0f0f0f;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #2a2a2a;
    --text-primary: #e8e8e8;
    --text-secondary: #a0a0a0;
    --text-muted: #6b6b6b;
    --accent: #6c5ce7;
    --positive: #00b894;
    --negative: #e74c3c;
    --neutral: #fdcb6e;
    --border: #2a2a2a;
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: 1px solid #2a2a2a;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8e8e8 !important;
}

/* Hide default Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none !important;}

/* Chat input */
.stChatInput > div {
    background-color: #1e1e1e !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
}

.stChatInput textarea {
    color: #e8e8e8 !important;
}

/* Chat messages */
.stChatMessage {
    background-color: transparent !important;
    border-bottom: 1px solid #1a1a1a;
    padding: 20px 0 !important;
}

/* ── Analysis Card ─────────────────────────────────────────────── */
.analysis-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    overflow: hidden;
    margin: 12px 0;
}

.analysis-header {
    padding: 18px 24px;
    border-bottom: 1px solid #2a2a2a;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.analysis-header h3 {
    margin: 0;
    font-size: 15px;
    color: #e8e8e8;
}

.badge {
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-positive { background: rgba(0,184,148,0.15); color: #00b894; }
.badge-negative { background: rgba(231,76,60,0.15); color: #e74c3c; }
.badge-neutral  { background: rgba(253,203,110,0.15); color: #fdcb6e; }

.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #2a2a2a;
}

.metric-cell {
    background: #1a1a1a;
    padding: 18px 24px;
}

.metric-label {
    font-size: 11px;
    color: #6b6b6b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
    font-weight: 600;
}

.metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #e8e8e8;
}

.metric-value.positive { color: #00b894; }
.metric-value.negative { color: #e74c3c; }
.metric-value.neutral  { color: #fdcb6e; }

/* Confidence bar */
.conf-bar-bg {
    height: 6px;
    background: #333;
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s ease;
}
.conf-bar-fill.positive { background: linear-gradient(90deg, #00b894, #55efc4); }
.conf-bar-fill.negative { background: linear-gradient(90deg, #e74c3c, #ff6b6b); }
.conf-bar-fill.neutral  { background: linear-gradient(90deg, #fdcb6e, #ffeaa7); }

/* Detail rows */
.detail-section {
    border-top: 1px solid #2a2a2a;
    padding: 16px 24px;
}

.detail-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid #222;
}
.detail-row:last-child { border-bottom: none; }
.detail-key { color: #a0a0a0; }
.detail-val { color: #e8e8e8; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* Suggestions */
.suggestions {
    border-top: 1px solid #2a2a2a;
    padding: 16px 24px;
}
.suggestions h4 {
    font-size: 12px;
    color: #6b6b6b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 10px 0;
}
.suggestion-item {
    padding: 6px 0;
    font-size: 13px;
    color: #a0a0a0;
    line-height: 1.5;
}
.suggestion-item::before {
    content: "→ ";
    color: #6c5ce7;
    font-weight: 700;
}

/* Modality tab labels */
.modality-label {
    display: inline-block;
    padding: 4px 10px;
    background: #2a2a2a;
    border-radius: 6px;
    font-size: 11px;
    color: #a0a0a0;
    margin-right: 6px;
    font-weight: 600;
}

/* Welcome card */
.welcome-card {
    text-align: center;
    padding: 48px 24px;
}
.welcome-icon {
    width: 72px; height: 72px;
    margin: 0 auto 20px;
    background: linear-gradient(135deg, #6c5ce7, #a29bfe, #fd79a8);
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    box-shadow: 0 8px 32px rgba(108,92,231,0.25);
}

/* Streamlit expander */
div[data-testid="stExpander"] {
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    background: #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy-loaded model singletons ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_text_model():
    from app.models.text_analyser import TextSentimentAnalyser
    return TextSentimentAnalyser()

@st.cache_resource(show_spinner=False)
def load_audio_model():
    from app.models.audio_analyser import AudioEmotionAnalyser
    return AudioEmotionAnalyser()

@st.cache_resource(show_spinner=False)
def load_image_model():
    from app.models.image_analyser import ImageSentimentAnalyser
    return ImageSentimentAnalyser()


# ── Fusion ─────────────────────────────────────────────────────────────────
SUGGESTIONS = {
    "Delivery Delay": [
        "Apologise for the delay and provide a tracking update.",
        "Offer expedited re-shipping or a discount on the next order.",
        "Escalate to logistics team for investigation.",
    ],
    "Product Quality": [
        "Initiate a replacement or full refund immediately.",
        "Forward case to the quality-assurance team.",
        "Send a personalised apology with a goodwill voucher.",
    ],
    "Customer Service": [
        "Review the interaction and coach the agent involved.",
        "Follow up personally with the customer.",
        "Offer a service-recovery credit.",
    ],
    "Billing Issue": [
        "Verify the charge and issue a correction / refund.",
        "Send an updated invoice or receipt.",
        "Escalate to finance team for review.",
    ],
    "App/Website Bug": [
        "Log a high-priority bug ticket for engineering.",
        "Provide a workaround or alternative way to complete the action.",
        "Notify the customer once the fix is deployed.",
    ],
}


def fuse_results(results: dict) -> dict:
    """Weighted fusion of modality results."""
    weights = {"text": 0.50, "audio": 0.25, "image": 0.25}
    priority = {"Negative": 2, "Neutral": 1, "Positive": 0}

    sentiment_scores, conf_sum, total_w = {}, 0.0, 0.0
    for mod, r in results.items():
        w = weights.get(mod, 0.25)
        s = r["sentiment"]
        sentiment_scores[s] = sentiment_scores.get(s, 0.0) + w
        conf_sum += r["confidence"] * w
        total_w += w

    sentiment = max(sentiment_scores, key=lambda s: (sentiment_scores[s], priority.get(s, 0)))
    confidence = conf_sum / total_w if total_w else 0.0

    emotions = [r["emotion"] for r in results.values() if r["emotion"] != "Unknown"]
    emotion = max(set(emotions), key=emotions.count) if emotions else "Unknown"

    issue = "N/A"
    if "text" in results:
        issue = results["text"].get("details", {}).get("detected_issue", "N/A")
    if issue in ("N/A", "Unknown") and "image" in results:
        issue = results["image"].get("details", {}).get("detected_issue", "N/A")

    return {
        "customer_emotion": emotion,
        "sentiment": sentiment,
        "main_issue": issue,
        "confidence": round(confidence, 2),
        "modalities_used": list(results.keys()),
        "modality_results": results,
        "suggestions": SUGGESTIONS.get(issue, []),
    }


# ── Render analysis card as HTML ──────────────────────────────────────────
def render_analysis_card(a: dict) -> str:
    s = a["sentiment"].lower()
    badge = f'<span class="badge badge-{s}">{a["sentiment"]}</span>'

    # Metrics grid
    metrics = f"""
    <div class="metrics-grid">
      <div class="metric-cell">
        <div class="metric-label">Customer Emotion</div>
        <div class="metric-value">{a["customer_emotion"]}</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Sentiment</div>
        <div class="metric-value {s}">{a["sentiment"]}</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Main Issue</div>
        <div class="metric-value">{a["main_issue"]}</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Confidence</div>
        <div class="metric-value">{a["confidence"]:.0f}%</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill {s}" style="width:{a['confidence']}%"></div></div>
      </div>
    </div>"""

    # Per-modality detail sections
    detail_html = ""
    for mod in a.get("modalities_used", []):
        r = a["modality_results"][mod]
        rows = f"""
        <div class="detail-row"><span class="detail-key">Sentiment</span><span class="detail-val">{r['sentiment']}</span></div>
        <div class="detail-row"><span class="detail-key">Emotion</span><span class="detail-val">{r['emotion']}</span></div>
        <div class="detail-row"><span class="detail-key">Confidence</span><span class="detail-val">{r['confidence']:.1f}%</span></div>"""
        for k, v in r.get("details", {}).items():
            val = str(v) if not isinstance(v, dict) else ", ".join(f"{dk}: {dv}" for dk, dv in v.items())
            rows += f'<div class="detail-row"><span class="detail-key">{k}</span><span class="detail-val">{val}</span></div>'
        detail_html += f"""
        <div class="detail-section">
          <span class="modality-label">{mod.upper()}</span>
          {rows}
        </div>"""

    # Suggestions
    sugg_html = ""
    if a.get("suggestions"):
        items = "".join(f'<div class="suggestion-item">{s}</div>' for s in a["suggestions"])
        sugg_html = f'<div class="suggestions"><h4>💡 Recommended Actions</h4>{items}</div>'

    return f"""
    <div class="analysis-card">
      <div class="analysis-header">
        <h3>📊 Analysis Report</h3>
        {badge}
      </div>
      {metrics}
      {detail_html}
      {sugg_html}
    </div>"""


# ── Run analysis ──────────────────────────────────────────────────────────
def run_analysis(text: str | None, audio_file, image_file) -> dict:
    results = {}

    if text and text.strip():
        with st.spinner("🔤 Analysing text with RoBERTa…"):
            analyser = load_text_model()
            results["text"] = analyser.analyse(text.strip())

    if audio_file is not None:
        with st.spinner("🎙️ Analysing audio with Wav2Vec2…"):
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            try:
                analyser = load_audio_model()
                results["audio"] = analyser.analyse(tmp_path)
            finally:
                os.unlink(tmp_path)

    if image_file is not None:
        with st.spinner("🖼️ Analysing image with CLIP…"):
            suffix = os.path.splitext(image_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_file.read())
                tmp_path = tmp.name
            try:
                analyser = load_image_model()
                results["image"] = analyser.analyse(tmp_path)
            finally:
                os.unlink(tmp_path)

    if not results:
        return None

    report = fuse_results(results)
    return report


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Sentiment AI")
    st.caption("Multimodal Intelligence")

    st.divider()

    if st.button("＋  New Analysis", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.uploads = {"audio": None, "image": None}
        st.rerun()

    st.divider()

    # File upload section
    st.markdown("### 📎 Attachments")
    audio_upload = st.file_uploader(
        "Audio file", type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="audio_uploader", label_visibility="collapsed",
        help="Upload a customer call recording (.wav, .mp3)"
    )
    image_upload = st.file_uploader(
        "Image file", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        key="image_uploader", label_visibility="collapsed",
        help="Upload a social media image (.jpg, .png)"
    )

    st.divider()

    # Quick examples
    st.markdown("### ⚡ Quick Examples")
    examples = {
        "📦 Delivery complaint": "I've been waiting 3 weeks for my order and it still hasn't arrived. This is unacceptable! Your support was no help at all.",
        "⭐ Positive review": "The product quality is amazing, I love everything about it! Fast delivery and perfect packaging.",
        "🐛 App bug report": "The app keeps crashing every time I try to make a payment. I'm frustrated and considering switching.",
        "💳 Billing issue": "I was overcharged $50 on my last invoice and nobody in support can help me. This is ridiculous!",
    }
    for label, prompt in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_example = prompt
            st.rerun()

    st.divider()

    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px">'
        '<div style="width:8px;height:8px;border-radius:50%;background:#00b894;animation:pulse 2s infinite"></div>'
        '<span style="font-size:12px;color:#6b6b6b">RoBERTa · Wav2Vec2 · CLIP</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_example" not in st.session_state:
    st.session_state.pending_example = None


# ══════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════

# ── Welcome screen (when no messages) ─────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
      <div class="welcome-icon">🧠</div>
      <h2 style="color:#e8e8e8; font-size:28px; margin-bottom:8px">Sentiment Intelligence</h2>
      <p style="color:#a0a0a0; font-size:15px; max-width:520px; margin:0 auto; line-height:1.6">
        Analyse customer feedback across <b>text</b>, <b>audio</b>, and <b>images</b>.
        I combine <b>RoBERTa</b>, <b>Wav2Vec2</b>, and <b>CLIP</b> to detect emotions,
        sentiment, and issues in real time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("💬  **Type** a customer review in the chat box below")
        st.info("🎙️  **Upload** an audio file via the sidebar")
    with col2:
        st.info("🖼️  **Attach** an image via the sidebar")
        st.info("⚡  Try a **Quick Example** from the sidebar")

# ── Display chat history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        # Show file attachment chips
        if msg.get("audio_name"):
            st.caption(f"🎙️ {msg['audio_name']}")
        if msg.get("image_name"):
            st.caption(f"🖼️ {msg['image_name']}")
        # Show analysis card
        if msg.get("analysis_html"):
            st.markdown(msg["analysis_html"], unsafe_allow_html=True)


# ── Handle pending quick-example ──────────────────────────────────
if st.session_state.pending_example:
    prompt = st.session_state.pending_example
    st.session_state.pending_example = None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Run analysis
    with st.chat_message("assistant", avatar="🧠"):
        report = run_analysis(prompt, None, None)
        if report:
            summary = _build_summary(report) if False else ""  # noqa – handled below
            s_parts = []
            s_parts.append(f"I detected **{report['customer_emotion']}** emotion with **{report['sentiment']}** sentiment ({report['confidence']:.0f}% confidence).")
            if report["main_issue"] != "N/A":
                s_parts.append(f"The main issue appears to be: **{report['main_issue']}**.")
            mods = ", ".join(m.title() for m in report["modalities_used"])
            s_parts.append(f"Modalities analysed: {mods}.")
            summary = " ".join(s_parts)

            st.markdown(summary)
            card_html = render_analysis_card(report)
            st.markdown(card_html, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": summary,
                "analysis_html": card_html,
            })
        else:
            err = "Please provide at least one input (text, audio, or image)."
            st.warning(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

    st.rerun()


# ── Chat input ────────────────────────────────────────────────────
if prompt := st.chat_input("Paste a customer review, or attach audio/image via the sidebar…"):
    # Capture any uploaded files at the moment of sending
    audio_file = audio_upload
    image_file = image_upload

    # Add user message
    user_msg = {
        "role": "user",
        "content": prompt,
        "audio_name": audio_file.name if audio_file else None,
        "image_name": image_file.name if image_file else None,
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        if audio_file:
            st.caption(f"🎙️ {audio_file.name}")
        if image_file:
            st.caption(f"🖼️ {image_file.name}")

    # Run analysis
    with st.chat_message("assistant", avatar="🧠"):
        start = time.time()
        report = run_analysis(prompt, audio_file, image_file)
        elapsed = round((time.time() - start) * 1000)

        if report:
            s_parts = []
            s_parts.append(f"I detected **{report['customer_emotion']}** emotion with **{report['sentiment']}** sentiment ({report['confidence']:.0f}% confidence).")
            if report["main_issue"] != "N/A":
                s_parts.append(f"The main issue appears to be: **{report['main_issue']}**.")
            mods = ", ".join(m.title() for m in report["modalities_used"])
            s_parts.append(f"Modalities analysed: {mods}.")
            s_parts.append(f"⏱️ Processed in {elapsed}ms.")
            summary = " ".join(s_parts)

            st.markdown(summary)
            card_html = render_analysis_card(report)
            st.markdown(card_html, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": summary,
                "analysis_html": card_html,
            })
        else:
            err = "⚠️ Please provide at least one input (text, audio, or image)."
            st.warning(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

    st.rerun()
