import re

def _extract_markers(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    headings: list[tuple[int, str]] = []
    pages: list[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(True):
        raw = line.strip()
        if raw:
            m = re.match(r"^#{1,6}\s+(.*)$", raw)
            if m:
                headings.append((pos, m.group(1).strip()))
            m = re.match(r"^===\s*Page\s+(\d+)\s*===$", raw)
            if m:
                pages.append((pos, m.group(1)))
            m = re.match(r"^(Section|Chapter)\s+(.+)$", raw, re.IGNORECASE)
            if m:
                headings.append((pos, f"{m.group(1)} {m.group(2).strip()}"))
        pos += len(line)
    return headings, pages

def _last_before(markers: list[tuple[int, str]], pos: int) -> str | None:
    best = None
    for p, label in markers:
        if p <= pos:
            best = label
        else:
            break
    return best

def _detect_symbol(chunk: str) -> str | None:
    for line in chunk.splitlines()[:6]:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        m = re.match(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", raw)
        if m:
            return m.group(2)
        m = re.match(r"^(function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", raw)
        if m:
            return m.group(2)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", raw)
        if m:
            return m.group(1)
    return None

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200, doc_path: str | None = None) -> list[str]:

    text = (text or "").strip()
    if not text:
        return []

    headings, pages = _extract_markers(text)

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            section = _last_before(headings, i)
            page = _last_before(pages, i)
            symbol = _detect_symbol(chunk)
            meta_parts = []
            if doc_path:
                meta_parts.append(f"doc={doc_path}")
            if section:
                meta_parts.append(f"section={section}")
            if page:
                meta_parts.append(f"page={page}")
            if symbol:
                meta_parts.append(f"symbol={symbol}")
            if meta_parts:
                chunk = "META: " + " | ".join(meta_parts) + "\n" + chunk
            chunks.append(chunk)
        i = max(j - overlap, j)  
        if i == j:
            i = j
    return chunks
