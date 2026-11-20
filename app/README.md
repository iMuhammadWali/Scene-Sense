# SceneSense

Efficient **Video Question Answering** research MVP.

1. Process an MP4 once locally (5 FPS → CLIP → prune @ 0.95 → FAISS → VIDEO READY)
2. Classify each question with **BART-large-MNLI**
3. **SIMPLE → Fast path** — CLIP/FAISS retrieval only (**no VLM**)
4. **COMPLEX → Slow path** — CLIP/FAISS + temporal evidence → **OpenRouter VLM**

## Local vs API

| Component | Where | Model |
|-----------|-------|-------|
| CLIP | Local | `openai/clip-vit-base-patch32` |
| Classifier | Local | `facebook/bart-large-mnli` |
| FAISS / pruning | Local | N/A |
| Fast path (SIMPLE) | Local retrieval | **No VLM** |
| Slow path (COMPLEX) | OpenRouter API | `google/gemini-2.5-flash` (configurable) |

## Setup

```bash
cd backend
pip install -r requirements.txt
cd ..
python scripts/download_models.py   # CLIP + BART only
copy .env.example .env
# Edit .env → set OPENROUTER_API_KEY for COMPLEX queries
python scripts/verify_environment.py
.\start.ps1
```

Open http://localhost:5173

## Single start command

```powershell
.\start.ps1
```

## Tests

```bash
cd backend
pytest -q
```
