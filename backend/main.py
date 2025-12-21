from contextlib import asynccontextmanager
from db import init_db, add_memory, search_memory
from fastapi import FastAPI, HTTPException, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from embeddings import embed_texts
from faiss_store import search as faiss_search
from db import get_chunks_by_vector_ids
from pathlib import Path
from datetime import datetime, timezone
from index import run_index  
from typing import Any
from vosk import Model, KaldiRecognizer
import httpx
import faiss
import json
import asyncio
import os
import re
import tempfile
import subprocess

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
BASE_DIR = Path(__file__).resolve().parent.parent  # local-ai/
DOCS_DIR = BASE_DIR / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_INTERVAL_SECONDS = 180  # 3 minutes (tune later)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # tiny | base | small | medium
_WHISPER_MODEL = None  # lazy-loaded singleton
VOSK_MODEL_PATH = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
_vosk_model = None

INDEX_LOCK = asyncio.Lock()
INDEX_STATUS = {
    "state": "idle",               # idle | running | ok | error
    "is_indexing": False,
    "last_trigger": None,          # "startup" | "scheduled" | "manual"
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "stats": None,
}

def get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        if not VOSK_MODEL_PATH.exists():
            raise RuntimeError(f"Vosk model not found at: {VOSK_MODEL_PATH}")
        _vosk_model = Model(str(VOSK_MODEL_PATH))
    return _vosk_model

def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for voice: `faster-whisper`. "
                "Install with: pip install faster-whisper"
            ) from e

        # CPU-friendly defaults. (Works great on Mac; you can tune later.)
        _WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
    return _WHISPER_MODEL

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

async def _run_index_job(trigger: str):
    # Prevent overlapping runs
    if INDEX_LOCK.locked():
        return False

    async with INDEX_LOCK:
        INDEX_STATUS["state"] = "running"
        INDEX_STATUS["is_indexing"] = True
        INDEX_STATUS["last_trigger"] = trigger
        INDEX_STATUS["last_started_at"] = _now_iso()
        INDEX_STATUS["last_error"] = None

        try:
            stats = await run_index()
            INDEX_STATUS["stats"] = stats
            INDEX_STATUS["state"] = "ok"
        except Exception as e:
            INDEX_STATUS["state"] = "error"
            INDEX_STATUS["last_error"] = f"{type(e).__name__}: {str(e)}"
            # Optional: keep traceback for debugging
            # print(traceback.format_exc())
        finally:
            INDEX_STATUS["is_indexing"] = False
            INDEX_STATUS["last_finished_at"] = _now_iso()

    return True

async def _index_daemon(stop_event: asyncio.Event):
    # Run once on startup
    await _run_index_job("startup")

    while not stop_event.is_set():
        try:
            # wait N seconds or stop
            await asyncio.wait_for(stop_event.wait(), timeout=INDEX_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            # time to run scheduled indexing
            await _run_index_job("scheduled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    stop_event = asyncio.Event()
    task = asyncio.create_task(_index_daemon(stop_event))

    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except Exception:
            pass

app = FastAPI(title="Local AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    mode: str = "auto"          # auto | local | general
    model: str | None = None
    top_k: int = 6
    task: str | None = None     # qa | summary | None (auto-infer)

class RememberRequest(BaseModel):
    content: str
    source: str = "manual"

def _normalize_q(q: str) -> str:
    return (q or "").strip().lower()

def looks_like_doc_question(q: str) -> bool:
    """
    Strong hints user wants *their* local docs.
    """
    q = _normalize_q(q)
    doc_markers = [
        "my ", "mine", "resume", "cv", "kardex", "transcript", "document",
        "pdf", "file", "in this", "in the doc", "according to", "based on the document",
        "from my", "from the pdf", "from the file", "what does it say"
    ]
    return any(m in q for m in doc_markers)

def looks_like_general_question(q: str) -> bool:
    """
    Strong hints user wants general knowledge (even if local retrieval returns something vaguely similar).
    """
    q = _normalize_q(q)
    general_markers = [
        "what is", "who is", "explain", "how does", "define", "difference between",
        "history of", "when did", "why does", "examples of", "best way to",
    ]
    # If it also looks like a doc question, don't force general.
    return any(m in q for m in general_markers) and not looks_like_doc_question(q)

def clamp(s: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(s)))
    except Exception:
        return lo

async def llm_relevance_gate(question: str, sources: list[dict], model: str) -> tuple[bool, str]:
    """
    Optional small LLM 'judge' to prevent false positives:
    - returns (use_local, reason)
    Keep this tiny + deterministic.
    """
    # Take only a couple short snippets to keep it fast.
    def snip(t: str, n: int = 420) -> str:
        t = (t or "").strip()
        return t[:n] + ("…" if len(t) > n else "")

    blocks = []
    for s in sources[:2]:
        blocks.append(f"{s.get('label','S?')} ({s.get('doc_path','?')}):\n{snip(s.get('text_preview',''))}")

    prompt = (
        "You are a routing classifier for a local-docs assistant.\n"
        "Decide if the QUESTION can be answered using ONLY the provided SOURCES.\n"
        "If SOURCES are unrelated, say use_local=false.\n\n"
        f"QUESTION:\n{question}\n\n"
        "SOURCES:\n" + ("\n\n".join(blocks) if blocks else "(none)") + "\n\n"
        "Respond ONLY as strict JSON: {\"use_local\": true/false, \"reason\": \"...\"}\n"
    )

    try:
        raw = await ollama_generate(prompt, model)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return False, "gate:no-json"
        data = json.loads(m.group(0))
        use_local = bool(data.get("use_local", False))
        reason = str(data.get("reason", "")).strip()[:180]
        return use_local, f"gate:{reason or 'ok'}"
    except Exception:
        return False, "gate:error"

def infer_task(question: str) -> str:
    q = question.lower()
    summary_keywords = ["summarize", "summary", "overview", "bullet", "bullets", "tl;dr", "high level"]
    if any(k in q for k in summary_keywords):
        return "summary"
    return "qa"

def should_fallback_to_general(vector_ids: list[int], scores: list[float]) -> bool:
    """
    Heuristic: if we retrieved nothing OR the best similarity score is low,
    the question is probably not answerable from local docs.
    Tune threshold based on your embeddings/model.
    """
    if not vector_ids:
        return True
    best = scores[0] if scores else 0.0
    return best < 0.38  # adjust if needed (0.33-0.45 typical)

async def ollama_stream(prompt: str, model: str):
    """
    Yields text chunks from Ollama as they arrive (NDJSON stream).
    """
    payload = {"model": model, "prompt": prompt, "stream": True}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as r:
            r.raise_for_status()

            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Ollama streams partial text in `response`
                chunk = obj.get("response", "")
                if chunk:
                    yield chunk

                if obj.get("done"):
                    break

async def ollama_generate(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

def load_faiss_index():
    idx_path = Path(__file__).parent / "vector_index" / "index.faiss"
    if not idx_path.exists():
        return None
    idx = faiss.read_index(str(idx_path))
    if not isinstance(idx, faiss.IndexIDMap2):
        idx = faiss.IndexIDMap2(idx)
    return idx

async def build_local_prompt_and_sources(question: str, task: str, top_k: int) -> tuple[str | None, list, dict]:
    q_vec = (await embed_texts([question]))[0]

    index = load_faiss_index()
    if index is None:
        return None, [], {"top_k": top_k, "best_score": 0.0}

    k = top_k
    if task == "summary":
        k = max(k, 12)

    scores, vector_ids = faiss_search(index, q_vec, top_k=k)
    chunks = get_chunks_by_vector_ids(vector_ids)

    if not chunks:
        best = float(scores[0]) if scores else 0.0
        top3 = [float(x) for x in (scores[:3] if scores else [])]
        avg_top3 = sum(top3) / len(top3) if top3 else 0.0
        gap = (float(scores[0]) - float(scores[1])) if scores and len(scores) > 1 else 0.0

        return None, [], {"top_k": k, "best_score": best, "avg_top3": avg_top3, "score_gap": gap, "n_sources": 0}

    context_blocks = []
    sources = []
    for i, ch in enumerate(chunks):
        label = f"S{i+1}"
        context_blocks.append(
            f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
        )
        src_score = scores[i] if i < len(scores) else None
        sources.append({
            "label": label,
            "doc_path": ch["doc_path"],
            "chunk_index": ch["chunk_index"],
            "score": float(src_score) if src_score is not None else None,
            # small preview for routing gate (and optional UI later)
            "text_preview": (ch["text"] or "")[:500],
        })

    if task == "summary":
        instructions = (
            "TASK: Summarize using ONLY the provided SOURCES.\n"
            "You MUST synthesize a summary even if the sources are split across chunks.\n"
            "Do NOT say 'I don't know' just because a summary isn't explicitly written.\n"
            "If important sections are missing, make a partial summary and say what seems missing.\n"
            "Follow the user's format request (e.g. bullet points).\n"
            "Cite sources inline like [S1], [S2].\n"
        )
    else:
        instructions = (
            "TASK: Answer using ONLY the provided SOURCES.\n"
            "If the answer cannot be found in the sources, say 'I don't know'.\n"
            "Cite sources inline like [S1], [S2].\n"
        )

    prompt = (
        "You are a local, privacy-first assistant.\n\n"
        f"QUESTION:\n{question}\n\n"
        "SOURCES:\n" + "\n\n".join(context_blocks) + "\n\n"
        f"{instructions}"
    )

    best = float(scores[0]) if scores else 0.0
    top3 = [float(x) for x in (scores[:3] if scores else [])]
    avg_top3 = sum(top3) / len(top3) if top3 else 0.0
    gap = (float(scores[0]) - float(scores[1])) if scores and len(scores) > 1 else 0.0

    return prompt, sources, {
        "top_k": k,
        "best_score": best,
        "avg_top3": avg_top3,
        "score_gap": gap,
        "n_sources": len(sources),
    }

async def route_auto(question: str, task: str, top_k: int, model: str) -> dict[str, Any]:
    """
    Returns:
      {
        "final_mode": "auto->local"|"auto->general",
        "prompt": <string>,
        "sources": <list>,
        "retrieval": <dict>,
        "routing_reason": <string>
      }
    """
    prompt_local, sources_local, retrieval_local = await build_local_prompt_and_sources(question, task, top_k)

    best = clamp(retrieval_local.get("best_score", 0.0))
    avg3 = clamp(retrieval_local.get("avg_top3", 0.0))
    gap = clamp(retrieval_local.get("score_gap", 0.0), 0.0, 10.0)
    nsrc = int(retrieval_local.get("n_sources", 0) or 0)

    # Hard fallback if nothing retrieved
    if prompt_local is None or nsrc == 0:
        return {
            "final_mode": "auto->general",
            "prompt": question,
            "sources": [],
            "retrieval": {},
            "routing_reason": "no-local-context",
        }

    # Thresholds (tune over time)
    # - low confidence => general
    if best < 0.38 or avg3 < 0.34:
        return {
            "final_mode": "auto->general",
            "prompt": question,
            "sources": [],
            "retrieval": {},
            "routing_reason": f"low-retrieval(best={best:.2f},avg3={avg3:.2f})",
        }

    # - very high confidence => local
    if best >= 0.62:
        return {
            "final_mode": "auto->local",
            "prompt": prompt_local,
            "sources": sources_local,
            "retrieval": retrieval_local,
            "routing_reason": f"high-retrieval(best={best:.2f})",
        }

    # Middle zone: use heuristics + optional LLM gate to avoid false positives
    if looks_like_doc_question(question):
        return {
            "final_mode": "auto->local",
            "prompt": prompt_local,
            "sources": sources_local,
            "retrieval": retrieval_local,
            "routing_reason": f"doc-intent(best={best:.2f})",
        }

    if looks_like_general_question(question) and gap < 0.10:
        # If it looks general and retrieval isn't sharply confident, run a small gate
        use_local, reason = await llm_relevance_gate(question, sources_local, model)
        if not use_local:
            return {
                "final_mode": "auto->general",
                "prompt": question,
                "sources": [],
                "retrieval": {},
                "routing_reason": reason,
            }

    # Default: local
    return {
        "final_mode": "auto->local",
        "prompt": prompt_local,
        "sources": sources_local,
        "retrieval": retrieval_local,
        "routing_reason": f"mid-retrieval(best={best:.2f},gap={gap:.2f})",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    """
    Local speech-to-text.
    Frontend sends a recorded audio blob (webm/m4a/etc).
    We normalize with ffmpeg -> wav(16k, mono) then run Whisper locally.
    """
    if not file:
        raise HTTPException(status_code=400, detail="Missing audio file")

    # Save upload to temp
    suffix = Path(file.filename or "audio").suffix or ".webm"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src_path = td_path / f"input{suffix}"
        wav_path = td_path / "audio.wav"

        src_path.write_bytes(await file.read())

        # Convert to 16k mono wav for consistent Whisper input
        # Requires: brew install ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg failed. Install with: brew install ffmpeg",
            )

        try:
            model = _get_whisper_model()
            segments, info = model.transcribe(
                str(wav_path),
                beam_size=1,
                vad_filter=True,
            )
            text = "".join(seg.text for seg in segments).strip()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription error: {type(e).__name__}: {str(e)}")

    return {"text": text}

@app.get("/docs/list")
def docs_list():
    items = []
    for p in sorted(DOCS_DIR.glob("*")):
        if not p.is_file():
            continue
        # keep it simple; you can filter extensions later
        st = p.stat()
        items.append({
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"docs": items}

@app.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
    # Basic safety: no directories
    filename = os.path.basename(file.filename or "upload.bin")
    dest = DOCS_DIR / filename

    # Avoid overwriting: add suffix if exists
    if dest.exists():
        stem = dest.stem
        suf = dest.suffix
        i = 1
        while True:
            candidate = DOCS_DIR / f"{stem}_{i}{suf}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    content = await file.read()
    dest.write_bytes(content)

    # Kick indexing in background (do NOT block)
    try:
        asyncio.create_task(_run_index_job("upload"))
    except Exception:
        pass

    return {"ok": True, "saved_as": dest.name, "path": str(dest)}

@app.get("/index/status")
def index_status():
    return INDEX_STATUS

@app.post("/index/run")
async def index_run():
    # fire-and-forget (don’t block request)
    if INDEX_LOCK.locked():
        return {"ok": False, "started": False, "status": INDEX_STATUS}

    asyncio.create_task(_run_index_job("manual"))
    return {"ok": True, "started": True, "status": INDEX_STATUS}

@app.post("/remember")
def remember(req: RememberRequest):
    mem_id = add_memory(req.content, req.source)
    return {"ok": True, "id": mem_id}

@app.get("/memory/search")
def memory_search(q: str, limit: int = 10):
    return {"results": search_memory(q, limit)}

@app.post("/ask")
async def ask(req: AskRequest):
    model = req.model or DEFAULT_MODEL
    task = req.task or infer_task(req.question)

    # ---------- GENERAL MODE ----------
    async def run_general() -> dict:
        try:
            answer = await ollama_generate(req.question, model)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": answer,
            "mode": "general",
            "task": task,
            "sources": [],
            "used_tools": ["ollama"],
            "model": model,
        }

    # ---------- LOCAL MODE ----------
    async def run_local() -> dict:
        # 1) Embed the question
        q_vec = (await embed_texts([req.question]))[0]

        # 2) Load FAISS index
        index = load_faiss_index()
        if index is None:
            # No local KB yet
            return {
                "answer": "No local index found yet. Run indexing first, or use General mode.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
            }

        # 3) Retrieval settings
        k = req.top_k
        if task == "summary":
            k = max(k, 12)  # summaries usually need more context

        scores, vector_ids = faiss_search(index, q_vec, top_k=k)
        chunks = get_chunks_by_vector_ids(vector_ids)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in your local documents.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
            }

        # 4) Build context blocks + sources
        context_blocks = []
        sources = []
        for i, ch in enumerate(chunks):
            label = f"S{i+1}"
            context_blocks.append(
                f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
            )
            src_score = scores[i] if i < len(scores) else None
            sources.append({
                "label": label,
                "doc_path": ch["doc_path"],
                "chunk_index": ch["chunk_index"],
                "score": float(src_score) if src_score is not None else None,
            })

        # 5) Task-aware instructions (QA vs Summary)
        if task == "summary":
            instructions = (
                "TASK: Summarize using ONLY the provided SOURCES.\n"
                "You MUST synthesize a summary even if the sources are split across chunks.\n"
                "Do NOT say 'I don't know' just because a summary isn't explicitly written.\n"
                "If important sections are missing, make a partial summary and say what seems missing.\n"
                "Follow the user's format request (e.g. bullet points).\n"
                "Cite sources inline like [S1], [S2].\n"
            )
        else:
            instructions = (
                "TASK: Answer using ONLY the provided SOURCES.\n"
                "If the answer cannot be found in the sources, say 'I don't know'.\n"
                "Cite sources inline like [S1], [S2].\n"
            )

        prompt, sources, retrieval = await build_local_prompt_and_sources(req.question, task, req.top_k)

        if prompt is None:
            return {
                "answer": "I couldn't find anything relevant in your local documents.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
                "retrieval": retrieval,
            }

        try:
            answer = await ollama_generate(prompt, model)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": answer,
            "mode": "local",
            "task": task,
            "sources": sources,
            "used_tools": ["ollama", "faiss", "sqlite"],
            "model": model,
            "retrieval": retrieval,
        }

    # ---------- ROUTING ----------
    mode = (req.mode or "auto").lower().strip()

    if mode == "general":
        return await run_general()

    if mode == "local":
        return await run_local()

    if mode == "auto":
        routed = await route_auto(req.question, task, req.top_k, model)

        if routed["final_mode"] == "auto->general":
            result = await run_general()
            result["mode"] = "auto->general"
            result["routing_reason"] = routed["routing_reason"]
            return result

        # local
        try:
            answer = await ollama_generate(routed["prompt"], model)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": answer,
            "mode": "auto->local",
            "task": task,
            "sources": routed["sources"],
            "used_tools": ["ollama", "faiss", "sqlite"],
            "model": model,
            "retrieval": routed["retrieval"],
            "routing_reason": routed["routing_reason"],
        }

    raise HTTPException(status_code=400, detail="Unsupported mode. Use 'auto', 'local', or 'general'.")

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    model = req.model or DEFAULT_MODEL
    task = req.task or infer_task(req.question)
    mode = (req.mode or "auto").lower().strip()

    async def sse():
        # Decide route + build prompt (without generating yet)
        final_mode = mode
        sources = []
        retrieval = {}
        routing_reason = None

        if mode == "general":
            prompt = req.question

        elif mode == "local":
            prompt, sources, retrieval = await build_local_prompt_and_sources(req.question, task, req.top_k)
            if prompt is None:
                meta = {"mode": "local", "task": task, "sources": [], "model": model, "retrieval": retrieval}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield f"data: I couldn't find anything relevant in your local documents.\n\n"
                yield "event: done\ndata: {}\n\n"
                return

        elif mode == "auto":
            routed = await route_auto(req.question, task, req.top_k, model)
            final_mode = routed["final_mode"]
            prompt = routed["prompt"]
            sources = routed["sources"]
            retrieval = routed["retrieval"]
            routing_reason = routed["routing_reason"]

        else:
            meta = {"mode": "error", "task": task, "sources": [], "model": model}
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            yield f"data: Unsupported mode. Use 'auto', 'local', or 'general'.\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Send meta first (frontend uses this to set sources/mode/task)
        meta = {
            "mode": final_mode,
            "task": task,
            "sources": sources,
            "model": model,
            "retrieval": retrieval,
            "routing_reason": routing_reason if mode == "auto" else None,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Stream tokens
        try:
            async for chunk in ollama_stream(prompt, model):
                # SSE message
                yield f"data: {chunk}\n\n"
        except httpx.ConnectError:
            yield f"data: Cannot connect to Ollama. Is `ollama serve` running?\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")

@app.websocket("/voice/stt/ws")
async def voice_stt_ws(ws: WebSocket):
    await ws.accept()

    try:
        model = get_vosk_model()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    # El frontend enviará PCM16 mono 16kHz
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    partial_last = ""

    try:
        while True:
            data = await ws.receive_bytes()  # raw PCM16 bytes
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                text = (res.get("text") or "").strip()
                if text:
                    await ws.send_json({"type": "final", "text": text})
                    partial_last = ""
            else:
                pres = json.loads(recognizer.PartialResult())
                p = (pres.get("partial") or "").strip()
                # evita spamear si no cambió
                if p and p != partial_last:
                    partial_last = p
                    await ws.send_json({"type": "partial", "text": p})

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()



