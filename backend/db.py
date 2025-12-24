import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  source TEXT DEFAULT 'manual'
);

CREATE INDEX IF NOT EXISTS idx_episodic_created_at ON episodic_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_episodic_content ON episodic_memory(content);

-- Documents that have been indexed
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE,
  file_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active', -- active | deleted
  indexed_at TEXT NOT NULL
);

-- Chunks belonging to a document (each chunk corresponds to one FAISS vector)
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  vector_id INTEGER NOT NULL UNIQUE,
  FOREIGN KEY(doc_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

def add_memory(content: str, source: str = "manual") -> int:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO episodic_memory(content, created_at, source) VALUES (?, ?, ?)",
            (content, now, source),
        )
        return int(cur.lastrowid)

def search_memory(query: str, limit: int = 10):
    like = f"%{query}%"
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at, source FROM episodic_memory WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (like, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def upsert_document(path_str: str, file_hash: str) -> int:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        row = conn.execute("SELECT id, file_hash, status FROM documents WHERE path = ?", (path_str,)).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO documents(path, file_hash, status, indexed_at) VALUES (?, ?, 'active', ?)",
                (path_str, file_hash, now),
            )
            return int(cur.lastrowid)

        conn.execute(
            "UPDATE documents SET file_hash=?, status='active', indexed_at=? WHERE id=?",
            (file_hash, now, int(row["id"])),
        )
        return int(row["id"])

def get_document_by_path(path_str: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (path_str,)).fetchone()
        return dict(row) if row else None

def list_active_document_paths() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT path FROM documents WHERE status='active'").fetchall()
        return [r["path"] for r in rows]

def mark_document_deleted(path_str: str) -> list[int]:
    with get_conn() as conn:
        doc = conn.execute("SELECT id FROM documents WHERE path = ?", (path_str,)).fetchone()
        if not doc:
            return []
        doc_id = int(doc["id"])
        vec_rows = conn.execute("SELECT vector_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        vector_ids = [int(r["vector_id"]) for r in vec_rows]

        conn.execute("UPDATE documents SET status='deleted' WHERE id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))  # keep DB clean in strict sync
        return vector_ids

def chunk_exists_for_doc(doc_id: int, file_hash: str) -> bool:
    return False

def get_doc_hash_and_status(path_str: str):
    with get_conn() as conn:
        row = conn.execute("SELECT file_hash, status FROM documents WHERE path = ?", (path_str,)).fetchone()
        if not row:
            return None
        return {"file_hash": row["file_hash"], "status": row["status"]}

def insert_chunks(doc_id: int, chunks: list[str], vector_ids: list[int]) -> None:
    with get_conn() as conn:
        for i, (txt, vid) in enumerate(zip(chunks, vector_ids)):
            conn.execute(
                "INSERT INTO chunks(doc_id, chunk_index, text, vector_id) VALUES (?, ?, ?, ?)",
                (doc_id, i, txt, int(vid)),
            )

def get_chunks_by_vector_ids(vector_ids: list[int]) -> list[dict]:
    if not vector_ids:
        return []

    order = {vid: i for i, vid in enumerate(vector_ids)}

    placeholders = ",".join(["?"] * len(vector_ids))
    sql = f"""
    SELECT
      c.vector_id,
      c.chunk_index,
      c.text,
      d.path AS doc_path
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    WHERE c.vector_id IN ({placeholders})
    """

    with get_conn() as conn:
        rows = conn.execute(sql, vector_ids).fetchall()
        items = [dict(r) for r in rows]
        items.sort(key=lambda x: order.get(int(x["vector_id"]), 10**9))
        return items
