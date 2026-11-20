# SceneSense Backend

FastAPI service.

- **Local:** CLIP, BART-large-MNLI, FAISS, retrieval
- **API:** OpenRouter VLM for **COMPLEX** queries only
- **SIMPLE:** CLIP/FAISS evidence — **no VLM call**

```bash
pip install -r requirements.txt
cd ..
python scripts/download_models.py
copy .env.example .env   # set OPENROUTER_API_KEY
cd backend
python -m app.main
```

```bash
pytest -q
```
