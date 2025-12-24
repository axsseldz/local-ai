AI assistant with local document indexing, web search, and voice STT.

Run locally

1. Backend

```
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

2. Frontend

```
cd frontend
npm install
npm run dev
```

Notes: the backend expects Ollama at `http://127.0.0.1:11434` and a Vosk model at `backend/models/vosk-model-en-us-0.22-lgraph`.
