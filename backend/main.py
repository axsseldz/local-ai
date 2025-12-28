from contextlib import asynccontextmanager
from db import init_db
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from embeddings import embed_texts
from faiss_store import search as faiss_search
from db import get_chunks_by_vector_ids, search_chunks_keyword
from pathlib import Path
from datetime import datetime, timezone
from index import run_index  
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import trafilatura
import httpx
import faiss
import json
import asyncio
import os
import subprocess
import time
import re

BASE_DIR = Path(__file__).resolve().parents[1]  
load_dotenv(BASE_DIR / ".env")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
AVAILABLE_MODELS = [
    "llama3.1:8b",
    "qwen2.5-coder:7b",
    "gpt-oss:20b",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
CONTEXT_WINDOW_TOKENS = 5000
DOCS_DIR = BASE_DIR / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_INTERVAL_SECONDS = 180 
VOSK_MODEL_PATH = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
INDEX_LOCK = asyncio.Lock()
MODEL_SWITCH_LOCK = asyncio.Lock()
_METAL_CACHE = {"value": None, "ts": 0.0}
_vosk_model = None
_CURRENT_LLM_MODEL: str | None = None
CONVERSATION_LOCK = asyncio.Lock()
CONVERSATIONS: dict[str, dict] = {}
MAX_CONTEXT_TOKENS = 2200
MAX_RECENT_TURNS = 3
WEB_MAX_BYTES = 1_500_000
WEB_FETCH_TIMEOUT = 15.0
WEB_FETCH_CONCURRENCY = 3
WEB_HEAD_PARAGRAPHS = 3
WEB_MAX_PARAGRAPHS = 8
WEB_MAX_CHARS = 2200

FORMAT_RULES = (
    "FORMAT:\n"
    "- Output MUST be valid GitHub-Flavored Markdown.\n"
    "- Use blank lines between paragraphs.\n"
    "- Lists must be real markdown lists (start lines with '-' or '1.').\n"
    "- Code must be fenced with triple backticks and a language when possible.\n"
    "- Tables MUST be real markdown tables:\n"
    "  * each row on its own line\n"
    "  * header separator row like: |---|---|\n"
    "  * NEVER output a table inline on one long line.\n"
    "- Do not output stray asterisks. Bold must be exactly **like this**.\n"
)

INDEX_STATUS = {
    "state": "idle",               
    "is_indexing": False,
    "last_trigger": None,          
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
    mode: str = "local" 
    model: str | None = None
    top_k: int = 6
    task: str | None = None
    conversation_id: str | None = None  

async def _run_index_job(trigger: str):
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
        finally:
            INDEX_STATUS["is_indexing"] = False
            INDEX_STATUS["last_finished_at"] = _now_iso()

    return True

async def _queue_index_job(trigger: str):
    if not INDEX_LOCK.locked():
        asyncio.create_task(_run_index_job(trigger))
        return

    async def wait_and_run():
        while INDEX_LOCK.locked():
            await asyncio.sleep(0.5)
        await _run_index_job(trigger)

    asyncio.create_task(wait_and_run())

async def _index_daemon(stop_event: asyncio.Event):
    await _run_index_job("startup")

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=INDEX_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            await _run_index_job("scheduled")

async def brave_web_search(query: str, count: int = 5):
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

async def _fetch_html(client: httpx.AsyncClient, url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalAI/1.0; +https://localhost)",
        "Accept": "text/html,application/xhtml+xml",
    }
    r = await client.get(url, headers=headers, follow_redirects=True)
    r.raise_for_status()
    content_type = (r.headers.get("content-type") or "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return ""
    content = r.content or b""
    if len(content) > WEB_MAX_BYTES:
        content = content[:WEB_MAX_BYTES]
    try:
        return content.decode(r.encoding or "utf-8", errors="ignore")
    except Exception:
        return ""

async def _build_web_context(query: str, results: list[dict]) -> tuple[list[dict], list[str]]:
    if not results:
        return [], []

    semaphore = asyncio.Semaphore(WEB_FETCH_CONCURRENCY)
    enriched: list[dict | None] = [None] * len(results)
    blocks: list[str | None] = [None] * len(results)

    async with httpx.AsyncClient(timeout=WEB_FETCH_TIMEOUT) as client:

        async def process_result(idx: int, r: dict) -> None:
            url = r.get("url") or ""
            if not url:
                return
            async with semaphore:
                try:
                    html = await _fetch_html(client, url)
                except Exception:
                    html = ""
            readable = _extract_readable_text(html, url)
            paragraphs = _extract_paragraphs(readable)
            selected = _select_relevant_paragraphs(paragraphs, query)
            excerpt = "\n\n".join(selected).strip()
            if not excerpt:
                excerpt = (r.get("snippet") or "").strip()
            if not excerpt:
                return

            title = r.get("title") or url
            label = r.get("label") or ""
            blocks[idx] = f"[{label}] {title}\nURL: {url}\nEXCERPT:\n{excerpt}"
            short_snippet = excerpt.split("\n", 1)[0][:280].strip()
            enriched[idx] = {
                "label": label,
                "title": title,
                "url": url,
                "snippet": short_snippet,
            }

        await asyncio.gather(*(process_result(i, r) for i, r in enumerate(results)))

    final_enriched = [e for e in enriched if e]
    final_blocks = [b for b in blocks if b]
    return final_enriched, final_blocks

async def ollama_stream(prompt: str, model: str):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True, 
        "options": {
            "num_ctx": CONTEXT_WINDOW_TOKENS,
            "temperature": 0.6,
        },
    }

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

                chunk = obj.get("response", "")
                if chunk:
                    yield chunk

                if obj.get("done"):
                    break

async def _ollama_unload_model(model: str) -> None:
    if not model:
        return
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/stop", json={"model": model})
            if r.status_code < 400:
                return
        except Exception:
            pass
        try:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            )
        except Exception:
            pass

async def _switch_llm_model(next_model: str) -> None:
    global _CURRENT_LLM_MODEL
    if not next_model:
        return
    async with MODEL_SWITCH_LOCK:
        prev = _CURRENT_LLM_MODEL
        if prev and prev != next_model:
            await _ollama_unload_model(prev)
        _CURRENT_LLM_MODEL = next_model
    
async def build_local_prompt_and_sources(
    question: str,
    task: str,
    top_k: int,
    retrieval_query: str,
) -> tuple[str | None, list, dict, list[str]]:
    query_text = retrieval_query or question
    q_vec = (await embed_texts([query_text]))[0]

    index = load_faiss_index()
    if index is None:
        return None, [], {"top_k": top_k, "best_score": 0.0}, []

    k = top_k
    if task == "summary":
        k = max(k, 12)

    scores, vector_ids = faiss_search(index, q_vec, top_k=k * 2)
    vec_scores = {vid: float(scores[i]) for i, vid in enumerate(vector_ids) if i < len(scores)}
    kw_hits = search_chunks_keyword(query_text, limit=k * 3)
    kw_scores = {int(h["vector_id"]): float(h.get("kw_score", 0)) for h in kw_hits}

    max_vec = max(vec_scores.values()) if vec_scores else 1.0
    max_kw = max(kw_scores.values()) if kw_scores else 1.0

    combined: dict[int, float] = {}
    for vid, score in vec_scores.items():
        combined[vid] = (score / max_vec)
    for vid, score in kw_scores.items():
        combined[vid] = combined.get(vid, 0.0) + 0.15 * (score / max_kw)

    ranked_ids = sorted(combined.keys(), key=lambda v: combined[v], reverse=True)[:k]
    chunks = get_chunks_by_vector_ids(ranked_ids)

    if not chunks:
        best = float(scores[0]) if scores else 0.0
        top3 = [float(x) for x in (scores[:3] if scores else [])]
        avg_top3 = sum(top3) / len(top3) if top3 else 0.0
        gap = (float(scores[0]) - float(scores[1])) if scores and len(scores) > 1 else 0.0

        return None, [], {"top_k": k, "best_score": best, "avg_top3": avg_top3, "score_gap": gap, "n_sources": 0}, []

    context_blocks = []
    sources = []
    for i, ch in enumerate(chunks):
        label = f"S{i+1}"
        context_blocks.append(
            f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
        )
        src_score = combined.get(int(ch.get("vector_id")), None)
        sources.append({
            "label": label,
            "doc_path": ch["doc_path"],
            "chunk_index": ch["chunk_index"],
            "score": float(src_score) if src_score is not None else None,
            "text_preview": (ch["text"] or "")[:500],
        })

    best = max(combined.values()) if combined else 0.0
    top3 = [float(x) for x in (scores[:3] if scores else [])]
    avg_top3 = sum(top3) / len(top3) if top3 else 0.0
    gap = (float(scores[0]) - float(scores[1])) if scores and len(scores) > 1 else 0.0

    return None, sources, {
        "top_k": k,
        "best_score": best,
        "avg_top3": avg_top3,
        "score_gap": gap,
        "n_sources": len(sources),
        "keyword_hits": len(kw_hits),
        "vector_hits": len(vector_ids),
        "retrieval_query": query_text,
    }, context_blocks

async def _rewrite_query(question: str, model: str) -> str:
    prompt = (
        "Rewrite the user question into a concise retrieval query. "
        "Keep key nouns, entities, and constraints. "
        "Return only the rewritten query text.\n\n"
        f"QUESTION:\n{question.strip()}"
    )
    try:
        rewritten = await _ollama_generate(prompt, model)
        return rewritten.strip() or question
    except Exception:
        return question

async def _ollama_generate(prompt: str, model: str) -> str:
    if not prompt:
        return ""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": CONTEXT_WINDOW_TOKENS,
            "temperature": 0.3,
        },
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

def _extract_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]

def _query_keywords(query: str) -> list[str]:
    if not query:
        return []
    words = re.findall(r"[a-zA-Z0-9']+", query.lower())
    stop = {
        "the", "and", "or", "but", "a", "an", "of", "to", "in", "for", "on", "with", "by",
        "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that",
        "these", "those", "it", "its", "into", "about", "over", "under", "what", "which",
        "who", "whom", "why", "how", "when", "where", "can", "could", "should", "would",
        "do", "does", "did", "done", "than", "then", "also", "only", "such",
    }
    deduped = []
    seen = set()
    for w in words:
        if len(w) < 3 or w in stop:
            continue
        if w not in seen:
            deduped.append(w)
            seen.add(w)
    return deduped

def _select_relevant_paragraphs(paragraphs: list[str], query: str) -> list[str]:
    if not paragraphs:
        return []
    keywords = _query_keywords(query)
    selected: list[str] = []
    seen = set()
    for i, para in enumerate(paragraphs):
        para_clean = para.strip()
        if not para_clean:
            continue
        para_lower = para_clean.lower()
        keep = i < WEB_HEAD_PARAGRAPHS
        if not keep and keywords:
            keep = any(k in para_lower for k in keywords)
        if keep and para_clean not in seen:
            selected.append(para_clean)
            seen.add(para_clean)
        if len(selected) >= WEB_MAX_PARAGRAPHS:
            break

    if not selected:
        selected = paragraphs[:WEB_HEAD_PARAGRAPHS]

    trimmed: list[str] = []
    total = 0
    for para in selected:
        next_total = total + len(para)
        if next_total > WEB_MAX_CHARS and trimmed:
            break
        trimmed.append(para)
        total = next_total
    return trimmed

def _extract_readable_text(html: str, url: str) -> str:
    if not html:
        return ""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_links=False,
        url=url,
    )
    return (text or "").strip()

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))

def _render_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = (m.get("role") or "user").title()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def _format_sources_block(sources: list[str], heading: str) -> str:
    if not sources:
        return ""
    return f"{heading}\n" + "\n\n".join(sources)

def _build_system_prompt(mode: str, task: str) -> str:
    lines = ["You are a local, privacy-first assistant."]
    if mode == "general":
        lines.extend([
            "Keep answers short by default; only go long if the user explicitly asks.",
            "Do NOT invent sources.",
        ])
    elif task == "summary":
        lines.extend([
            "TASK: Summarize using ONLY the provided SOURCES.",
            "You MUST synthesize a summary even if the sources are split across chunks.",
            "Do NOT say 'I don't know' just because a summary isn't explicitly written.",
            "If important sections are missing, make a partial summary and say what seems missing.",
            "Follow the user's format request (e.g. bullet points).",
            "Keep it concise unless the user explicitly asks for a detailed/long answer.",
            "Do NOT cite or mention source IDs or filenames in the answer.",
            "Treat SOURCES CONTEXT as authoritative.",
        ])
    elif mode == "search":
        lines.extend([
            "TASK: Answer using ONLY the WEB-RETRIEVED FACTUAL CONTEXT.",
            "If the answer cannot be found in the web context, say 'I don't know'.",
            "Keep it concise unless the user explicitly asks for a detailed/long answer.",
            "Do NOT cite or mention source IDs or filenames in the answer.",
            "Treat the web context as authoritative.",
        ])
    else:
        lines.extend([
            "TASK: Answer using ONLY the provided SOURCES.",
            "If the answer cannot be found in the sources, say 'I don't know'.",
            "Keep it concise unless the user explicitly asks for a detailed/long answer.",
            "Do NOT cite or mention source IDs or filenames in the answer.",
            "Treat SOURCES CONTEXT as authoritative.",
        ])
    lines.append(FORMAT_RULES)
    return "\n".join(lines).strip()

def _build_memory_prompt(
    system_prompt: str,
    summary: str,
    sources_block: str,
    messages: list[dict],
    user_payload: str,
) -> str:
    parts = [f"SYSTEM:\n{system_prompt}"]
    if summary:
        parts.append("SYSTEM:\nCONVERSATION CONTEXT (summary):\n" + summary.strip())
    if sources_block:
        parts.append("SYSTEM:\n" + sources_block.strip())
    if messages:
        rendered = _render_messages(messages)
        if rendered:
            parts.append("RECENT MESSAGES:\n" + rendered)
    parts.append("USER:\n" + user_payload.strip())
    return "\n\n".join(parts).strip()

def _select_top_k(question: str, base: int) -> int:
    q = (question or "").strip().lower()
    if not q:
        return base
    analytic = ["why", "how", "compare", "analysis", "explain", "summarize", "tradeoff", "pros", "cons"]
    if any(k in q for k in analytic) or len(q) > 120:
        return max(base, 10)
    if len(q) < 60:
        return max(4, min(base, 6))
    return base

def _build_summary_prompt(summary: str, messages: list[dict]) -> str:
    parts = [
        "Summarize the conversation below, focusing on user goals, constraints, decisions, and important facts, and keep it concise."
    ]
    if summary:
        parts.append("EXISTING SUMMARY:\n" + summary.strip())
    rendered = _render_messages(messages)
    if rendered:
        parts.append("MESSAGES:\n" + rendered)
    return "\n\n".join(parts).strip()

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

def _get_hw_memsize_bytes() -> int | None:
    out = _run_cmd(["sysctl", "-n", "hw.memsize"], timeout=1.0)
    if not out:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None

def _parse_vm_stat(out: str) -> dict:
    if not out:
        return {}

    page_size = 4096
    first_line = out.splitlines()[0] if out else ""
    m = re.search(r"page size of (\d+) bytes", first_line)
    if m:
        page_size = int(m.group(1))

    def grab(key: str) -> int:
        m2 = re.search(rf"{re.escape(key)}:\s+([\d]+)\.", out)
        return int(m2.group(1)) if m2 else 0

    active = grab("Pages active")
    wired = grab("Pages wired down")
    compressed = grab("Pages occupied by compressor")
    inactive = grab("Pages inactive")
    speculative = grab("Pages speculative")
    purgeable = grab("Pages purgeable")
    free = grab("Pages free")

    total_pages = active + inactive + speculative + wired + purgeable + compressed + free
    if total_pages == 0:
        return {}

    return {
        "used_bytes": (active + wired + compressed) * page_size,
        "free_bytes": (free + speculative) * page_size,
        "active_bytes": active * page_size,
        "wired_bytes": wired * page_size,
        "compressed_bytes": compressed * page_size,
        "inactive_bytes": inactive * page_size,
        "speculative_bytes": speculative * page_size,
        "purgeable_bytes": purgeable * page_size,
    }

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

def _get_process_rss_bytes(names: list[str]) -> int | None:
    out = _run_cmd(["ps", "-A", "-o", "rss=,comm="], timeout=1.5)
    if not out:
        return None
    targets = {n.lower() for n in names}
    total_kb = 0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rss_kb = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1].strip()
        base = os.path.basename(cmd).lower()
        if base in targets:
            total_kb += rss_kb
    return total_kb * 1024 if total_kb > 0 else None

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

def get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        if not VOSK_MODEL_PATH.exists():
            raise RuntimeError(f"Vosk model not found at: {VOSK_MODEL_PATH}")
        _vosk_model = Model(str(VOSK_MODEL_PATH))
    return _vosk_model

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def infer_task(question: str) -> str:
    q = question.lower()
    summary_keywords = ["summarize", "summary", "overview", "bullet", "bullets", "tl;dr", "high level"]
    if any(k in q for k in summary_keywords):
        return "summary"
    return "qa"

def load_faiss_index():
    idx_path = Path(__file__).parent / "vector_index" / "index.faiss"
    if not idx_path.exists():
        return None
    idx = faiss.read_index(str(idx_path))
    if not isinstance(idx, faiss.IndexIDMap2):
        idx = faiss.IndexIDMap2(idx)
    return idx

@app.get("/api/models")
def list_models():
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}

@app.get("/api/metrics")
def metrics():
    vm = _parse_vm_stat(_run_cmd(["vm_stat"]))
    hw_memsize = _get_hw_memsize_bytes()
    swap_used, swap_total = _parse_swap(_run_cmd(["sysctl", "vm.swapusage"]))
    cpu_percent = _get_cpu_percent()
    metal_supported = _get_metal_status()
    llm_rss_bytes = _get_process_rss_bytes(["ollama"])

    return {
        "system": {
            "memory_used_bytes": vm.get("used_bytes"),
            "memory_total_bytes": hw_memsize,
            "memory_free_bytes": vm.get("free_bytes"),
            "memory_active_bytes": vm.get("active_bytes"),
            "memory_wired_bytes": vm.get("wired_bytes"),
            "memory_compressed_bytes": vm.get("compressed_bytes"),
            "memory_inactive_bytes": vm.get("inactive_bytes"),
            "memory_speculative_bytes": vm.get("speculative_bytes"),
            "memory_purgeable_bytes": vm.get("purgeable_bytes"),
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
            "rss_bytes": llm_rss_bytes,
        },
        "model": {
        "name": _CURRENT_LLM_MODEL or DEFAULT_MODEL,
        "quantization": "Q4_K_M",
        "backend": "Ollama",
    },
    }

@app.get("/docs/list")
def docs_list():
    items = []
    for p in sorted(DOCS_DIR.glob("*")):
        if not p.is_file():
            continue

        st = p.stat()
        items.append({
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"docs": items}

@app.get("/index/status")
def index_status():
    return INDEX_STATUS

@app.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "upload.bin")
    dest = DOCS_DIR / filename

    content = await file.read()
    dest.write_bytes(content)

    try:
        await _queue_index_job("upload")
    except Exception:
        pass

    return {"ok": True, "saved_as": dest.name, "path": str(dest)}

@app.post("/index/run")
async def index_run():
    if INDEX_LOCK.locked():
        return {"ok": False, "started": False, "status": INDEX_STATUS}

    asyncio.create_task(_run_index_job("manual"))
    return {"ok": True, "started": True, "status": INDEX_STATUS}

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    model = req.model or DEFAULT_MODEL
    task = req.task or infer_task(req.question)
    mode = (req.mode or "local").lower().strip()
    conversation_id = req.conversation_id or "default"

    async def sse():
        await _switch_llm_model(model)
        final_mode = mode
        sources = []
        retrieval = {}
        context_blocks: list[str] = []
        
        if mode == "general":
            prompt = None

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

            sources, context_blocks = await _build_web_context(req.question, results)
            if not context_blocks:
                meta = {"mode": "search", "task": task, "sources": [], "model": model}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield "data: Web results could not be fetched or extracted.\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            final_mode = "search"


        elif mode == "local":
            adaptive_top_k = _select_top_k(req.question, req.top_k)
            retrieval_query = await _rewrite_query(req.question, model)
            prompt, sources, retrieval, context_blocks = await build_local_prompt_and_sources(
                req.question,
                task,
                adaptive_top_k,
                retrieval_query,
            )
            if not sources:
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

        async with CONVERSATION_LOCK:
            state = CONVERSATIONS.setdefault(conversation_id, {"summary": "", "messages": []})
            summary = state["summary"]
            recent_messages = list(state["messages"])

        system_prompt = _build_system_prompt(final_mode, task)
        sources_block = ""
        if final_mode in ("search", "local"):
            heading = "WEB-RETRIEVED FACTUAL CONTEXT:" if final_mode == "search" else "SOURCES:"
            sources_block = _format_sources_block(context_blocks, heading)
        user_payload = f"QUESTION:\n{req.question}"

        prompt = _build_memory_prompt(system_prompt, summary, sources_block, recent_messages, user_payload)
        est_tokens = _estimate_tokens(prompt)

        if est_tokens > MAX_CONTEXT_TOKENS and recent_messages:
            summary_prompt = _build_summary_prompt(summary, recent_messages)
            try:
                new_summary = await _ollama_generate(summary_prompt, model)
            except Exception:
                new_summary = summary

            pruned_messages = recent_messages[-(MAX_RECENT_TURNS * 2):]
            async with CONVERSATION_LOCK:
                state = CONVERSATIONS.setdefault(conversation_id, {"summary": "", "messages": []})
                state["summary"] = new_summary
                state["messages"] = pruned_messages
            summary = new_summary
            recent_messages = pruned_messages
            prompt = _build_memory_prompt(system_prompt, summary, sources_block, recent_messages, user_payload)

        meta = {
            "mode": final_mode,
            "task": task,
            "sources": sources,
            "model": model,
            "retrieval": retrieval,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        start_time = time.time()
        first_chunk_time = None
        token_count = 0
        assistant_text = ""
        LLM_METRICS["context_chars"] = len(prompt or "")
        LLM_METRICS["last_updated"] = time.time()
        try:
            async for chunk in ollama_stream(prompt, model):
                if first_chunk_time is None and chunk.strip():
                    first_chunk_time = time.time()
                token_count += len(chunk.split())
                assistant_text += chunk

                yield f"event: chunk\ndata: {json.dumps(chunk)}\n\n"
        except httpx.ConnectError:
            yield f"event: chunk\ndata: {json.dumps('Cannot connect to Ollama. Is `ollama serve` running?')}\n\n"
        except Exception as e:
            yield f"event: chunk\ndata: {json.dumps('Error: ' + str(e))}\n\n"
        finally:
            if first_chunk_time:
                elapsed = max(0.001, time.time() - first_chunk_time)
                LLM_METRICS["ttft_ms"] = round((first_chunk_time - start_time) * 1000)
                LLM_METRICS["tokens_per_second"] = round(token_count / elapsed, 2)
                LLM_METRICS["last_updated"] = time.time()

        if assistant_text.strip():
            async with CONVERSATION_LOCK:
                state = CONVERSATIONS.setdefault(conversation_id, {"summary": "", "messages": []})
                state["messages"].append({"role": "user", "content": req.question})
                state["messages"].append({"role": "assistant", "content": assistant_text.strip()})

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

    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    partial_last = ""

    try:
        while True:
            data = await ws.receive_bytes()  
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                text = (res.get("text") or "").strip()
                if text:
                    await ws.send_json({"type": "final", "text": text})
                    partial_last = ""
            else:
                pres = json.loads(recognizer.PartialResult())
                p = (pres.get("partial") or "").strip()
           
                if p and p != partial_last:
                    partial_last = p
                    await ws.send_json({"type": "partial", "text": p})

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
