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
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import httpx
import faiss
import json
import asyncio
import os
import tempfile
import subprocess
import time
import re

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

BASE_DIR = Path(__file__).resolve().parents[1]  # local-ai/
load_dotenv(BASE_DIR / ".env")

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

LLM_METRICS = {
    "tokens_per_second": None,
    "ttft_ms": None,
    "context_chars": None,
    "last_updated": None,
}

_METAL_CACHE = {"value": None, "ts": 0.0}

def _run_cmd(cmd: list[str], timeout: float = 1.5) -> str | None:
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except Exception:
        return None

def _parse_vm_stat(out: str) -> tuple[int | None, int | None]:
    if not out:
        return None, None
    page_size = 4096
    first_line = out.splitlines()[0] if out else ""
    m = re.search(r"page size of (\d+) bytes", first_line)
    if m:
        page_size = int(m.group(1))

    def grab(key: str) -> int:
        m2 = re.search(rf"{key}:\s+([\d]+)\.", out)
        return int(m2.group(1)) if m2 else 0

    free = grab("Pages free") + grab("Pages speculative")
    total = sum(
        grab(k)
        for k in [
            "Pages active",
            "Pages inactive",
            "Pages speculative",
            "Pages throttled",
            "Pages wired down",
            "Pages purgeable",
            "Pages occupied by compressor",
            "Pages free",
        ]
    )
    if total == 0:
        return None, None
    free_bytes = free * page_size
    used_bytes = (total * page_size) - free_bytes
    return used_bytes, total * page_size

def _parse_swap(out: str) -> tuple[int | None, int | None]:
    if not out:
        return None, None
    m_total = re.search(r"total = ([\d.]+)([KMG])", out)
    m_used = re.search(r"used = ([\d.]+)([KMG])", out)
    if not m_total or not m_used:
        return None, None

    def to_bytes(val: float, unit: str) -> int:
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(unit, 1)
        return int(val * mult)

    total = to_bytes(float(m_total.group(1)), m_total.group(2))
    used = to_bytes(float(m_used.group(1)), m_used.group(2))
    return used, total

def _get_cpu_percent() -> float | None:
    out = _run_cmd(["ps", "-A", "-o", "%cpu="], timeout=1.0)
    if not out:
        return None
    vals = []
    for line in out.splitlines():
        try:
            vals.append(float(line.strip()))
        except Exception:
            continue
    if not vals:
        return None
    ncpu_out = _run_cmd(["sysctl", "-n", "hw.ncpu"], timeout=1.0) or "1"
    try:
        ncpu = max(1, int(ncpu_out.strip()))
    except Exception:
        ncpu = 1
    total = sum(vals)
    return min(100.0, total / ncpu)

def _get_metal_status() -> bool | None:
    now = time.time()
    if _METAL_CACHE["value"] is not None and now - _METAL_CACHE["ts"] < 60:
        return _METAL_CACHE["value"]
    out = _run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=2.0)
    if not out:
        return None
    supported = "Metal: Supported" in out or "Metal Family" in out
    _METAL_CACHE["value"] = supported
    _METAL_CACHE["ts"] = now
    return supported

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

async def brave_web_search(query: str, count: int = 5):
    """
    Calls Brave Search API and returns normalized results:
    [{label, title, url, snippet}]
    """
    if not BRAVE_SEARCH_API_KEY:
        raise HTTPException(status_code=500, detail="BRAVE_SEARCH_API_KEY not set in .env")

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }

    params = {
        "q": query,
        "count": count,
        "search_lang": "en",
        "safesearch": "moderate",
        # You can add freshness like: "freshness": "week"
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

    results = []
    web = (data or {}).get("web", {})
    items = web.get("results", []) or []

    for i, it in enumerate(items[:count]):
        results.append({
            "label": f"W{i+1}",
            "title": it.get("title") or it.get("url") or f"Result {i+1}",
            "url": it.get("url") or "",
            "snippet": it.get("description") or "",
        })

    return results

def build_web_prompt(question: str, results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[{r['label']}] {r.get('title','')}\nURL: {r.get('url','')}\nSnippet: {r.get('snippet','')}"
        )

    return (
        "You are a privacy-first assistant.\n"
        "Use ONLY the WEB SOURCES below to answer.\n"
        "Keep the answer short by default; only go long if the user explicitly asks for detail.\n"
        "Do NOT cite or mention source IDs or websites inside the answer.\n"
        "If the sources don't contain enough info, say what is missing.\n\n"
        f"QUESTION:\n{question}\n\n"
        "WEB SOURCES:\n" + "\n\n".join(blocks) + "\n"
    )


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
    mode: str = "local"  # local | general | search
    model: str | None = None
    top_k: int = 6
    task: str | None = None     # qa | summary | None (auto-infer)

class RememberRequest(BaseModel):
    content: str
    source: str = "manual"

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
            "Keep it concise unless the user explicitly asks for a detailed/long answer.\n"
            "Do NOT cite or mention source IDs or filenames in the answer.\n"
        )
    else:
        instructions = (
            "TASK: Answer using ONLY the provided SOURCES.\n"
            "If the answer cannot be found in the sources, say 'I don't know'.\n"
            "Keep it concise unless the user explicitly asks for a detailed/long answer.\n"
            "Do NOT cite or mention source IDs or filenames in the answer.\n"
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/metrics")
def metrics():
    used_bytes, total_bytes = _parse_vm_stat(_run_cmd(["vm_stat"]))
    swap_used, swap_total = _parse_swap(_run_cmd(["sysctl", "vm.swapusage"]))
    cpu_percent = _get_cpu_percent()
    metal_supported = _get_metal_status()

    return {
        "system": {
            "memory_used_bytes": used_bytes,
            "memory_total_bytes": total_bytes,
            "swap_used_bytes": swap_used,
            "swap_total_bytes": swap_total,
            "cpu_percent": cpu_percent,
            "gpu_util_percent": None,
            "metal_supported": metal_supported,
        },
        "llm": {
            "tokens_per_second": LLM_METRICS["tokens_per_second"],
            "ttft_ms": LLM_METRICS["ttft_ms"],
            "context_chars": LLM_METRICS["context_chars"],
            "last_updated": LLM_METRICS["last_updated"],
        },
        "model": {
            "name": DEFAULT_MODEL,
            "quantization": None,
            "backend": "Ollama",
        },
    }

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
        prompt = (
            "You are a helpful assistant.\n"
            "Keep answers short by default; only go long if the user explicitly asks for a detailed/extended answer.\n"
            "Do NOT invent source mentions.\n\n"
            f"QUESTION:\n{req.question}\n"
        )
        try:
            answer = await ollama_generate(prompt, model)
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
                "Keep it concise unless the user explicitly asks for a detailed/long answer.\n"
                "Do NOT cite or mention source IDs or filenames in the answer.\n"
            )
        else:
            instructions = (
                "TASK: Answer using ONLY the provided SOURCES.\n"
                "If the answer cannot be found in the sources, say 'I don't know'.\n"
                "Keep it concise unless the user explicitly asks for a detailed/long answer.\n"
                "Do NOT cite or mention source IDs or filenames in the answer.\n"
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
    mode = (req.mode or "local").lower().strip()

    if mode == "general":
        return await run_general()

    if mode == "local":
        return await run_local()
    
    if mode == "search":
    # Brave search -> local LLM synthesis
        try:
            results = await brave_web_search(req.question, count=5)
            if not results:
                return {
                    "answer": "No web results found for that query.",
                    "mode": "search",
                    "task": task,
                    "sources": [],
                    "used_tools": ["brave_search", "ollama"],
                    "model": model,
                }

            prompt = build_web_prompt(req.question, results)
            answer = await ollama_generate(prompt, model)

            return {
                "answer": answer,
                "mode": "search",
                "task": task,
                "sources": results,
                "used_tools": ["brave_search", "ollama"],
                "model": model,
            }
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Brave Search error: {str(e)}")

    raise HTTPException(status_code=400, detail="Unsupported mode. Use 'local', 'general', or 'search'.")

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    model = req.model or DEFAULT_MODEL
    task = req.task or infer_task(req.question)
    mode = (req.mode or "local").lower().strip()

    async def sse():
        # Decide route + build prompt (without generating yet)
        final_mode = mode
        sources = []
        retrieval = {}
        if mode == "general":
            prompt = (
                "You are a helpful assistant.\n"
                "Keep answers short by default; only go long if the user explicitly asks for a detailed/extended answer.\n"
                "Do NOT invent source mentions.\n\n"
                f"QUESTION:\n{req.question}\n"
            )

        elif mode == "search":
            try:
                results = await brave_web_search(req.question, count=5)
            except httpx.HTTPError as e:
                meta = {"mode": "search", "task": task, "sources": [], "model": model}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield f"data: Brave Search error: {str(e)}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            if not results:
                meta = {"mode": "search", "task": task, "sources": [], "model": model}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield "data: No web results found for that query.\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            sources = results
            prompt = build_web_prompt(req.question, results)
            final_mode = "search"


        elif mode == "local":
            prompt, sources, retrieval = await build_local_prompt_and_sources(req.question, task, req.top_k)
            if prompt is None:
                meta = {"mode": "local", "task": task, "sources": [], "model": model, "retrieval": retrieval}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield f"data: I couldn't find anything relevant in your local documents.\n\n"
                yield "event: done\ndata: {}\n\n"
                return

        else:
            meta = {"mode": "error", "task": task, "sources": [], "model": model}
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            yield f"data: Unsupported mode. Use 'local', 'general', or 'search'.\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Send meta first (frontend uses this to set sources/mode/task)
        meta = {
            "mode": final_mode,
            "task": task,
            "sources": sources,
            "model": model,
            "retrieval": retrieval,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Stream tokens
        start_time = time.time()
        first_chunk_time = None
        token_count = 0
        LLM_METRICS["context_chars"] = len(prompt or "")
        LLM_METRICS["last_updated"] = time.time()
        try:
            async for chunk in ollama_stream(prompt, model):
                if first_chunk_time is None and chunk.strip():
                    first_chunk_time = time.time()
                token_count += len(chunk.split())
                # SSE message
                yield f"data: {chunk}\n\n"
        except httpx.ConnectError:
            yield f"data: Cannot connect to Ollama. Is `ollama serve` running?\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        finally:
            if first_chunk_time:
                elapsed = max(0.001, time.time() - first_chunk_time)
                LLM_METRICS["ttft_ms"] = round((first_chunk_time - start_time) * 1000)
                LLM_METRICS["tokens_per_second"] = round(token_count / elapsed, 2)
                LLM_METRICS["last_updated"] = time.time()

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
