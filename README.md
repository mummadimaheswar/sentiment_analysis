# 🧠 Multimodal Customer Sentiment Intelligence

A **ChatGPT / Gemini-style** web application that analyses customer feedback across **text, audio, and images** using state-of-the-art AI models.

![Sentiment Intelligence](https://img.shields.io/badge/AI-Multimodal-blueviolet)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)

---

## ✨ Features

| Modality | Model | What it does |
|----------|-------|--------------|
| **Text** | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | Sentiment classification + emotion & issue detection |
| **Audio** | Wav2Vec2 (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) | Speech emotion recognition from `.wav` files |
| **Image** | CLIP (`openai/clip-vit-base-patch32`) | Zero-shot image sentiment & issue classification |

- 🎨 **Modern chat UI** — dark theme, ChatGPT/Gemini look-and-feel
- 📊 **Rich analysis cards** — emotion, sentiment, confidence bars, per-modality details
- 💡 **Actionable suggestions** — automated response recommendations
- 🔀 **Multimodal fusion** — weighted combination of all modalities
- ⚡ **Lazy model loading** — only loads models you actually use
- 🐳 **Docker-ready** — one command to deploy

---

## 📁 Project Structure

```
├── streamlit_app.py            # ⭐ Streamlit chatbot UI (primary)
├── app/
│   ├── config.py               # Environment & model configuration
│   ├── main.py                 # FastAPI app (alternative backend)
│   ├── schemas.py              # Pydantic response models
│   ├── models/
│   │   ├── text_analyser.py    # RoBERTa text sentiment
│   │   ├── audio_analyser.py   # Wav2Vec2 audio emotion
│   │   ├── image_analyser.py   # CLIP image sentiment
│   │   └── fusion.py           # Multimodal fusion engine
│   ├── routes/
│   │   └── chat.py             # API endpoints (/analyse, /health)
│   ├── static/
│   │   ├── css/style.css       # ChatGPT-style dark theme
│   │   └── js/app.js           # Chat UI logic
│   └── templates/
│       └── index.html          # Single-page application
├── sentimentanaysis.py         # Original CLI version
├── run.py                      # FastAPI server launcher
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (Streamlit)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit chatbot

```bash
streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501** with a ChatGPT-style dark UI.

---

## 🖥️ Alternative: FastAPI Backend

```bash
python run.py
# Open http://127.0.0.1:8000
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

The Streamlit app will be available at `http://localhost:8501`.

---

## 🔌 API Reference

### `POST /api/analyse`

Accepts multipart form data with any combination of:

| Field   | Type   | Description |
|---------|--------|-------------|
| `text`  | string | Customer review text |
| `audio` | file   | Audio file (.wav, .mp3, .flac) |
| `image` | file   | Image file (.jpg, .png, .webp) |

**Response:**
```json
{
  "customer_emotion": "Frustration",
  "sentiment": "Negative",
  "main_issue": "Delivery delay",
  "confidence": 94.65,
  "modalities_used": ["text"],
  "modality_results": { ... },
  "suggestions": [ ... ],
  "processing_time_ms": 320
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

---

## 🧪 CLI Usage (legacy)

```bash
python sentimentanaysis.py --demo
python sentimentanaysis.py --text "Your product is terrible!"
python sentimentanaysis.py --text "..." --audio call.wav --image photo.jpg
```

---

## 📝 License

MIT
