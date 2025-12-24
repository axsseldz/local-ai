from pathlib import Path
import numpy as np
import faiss

FAISS_DIR = Path(__file__).parent / "vector_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = FAISS_DIR / "index.faiss"

def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def load_or_create(dim: int) -> faiss.IndexIDMap2:
    if INDEX_PATH.exists():
        idx = faiss.read_index(str(INDEX_PATH))

        if not isinstance(idx, faiss.IndexIDMap2):
            idx = faiss.IndexIDMap2(idx)
        return idx

    base = faiss.IndexFlatIP(dim)  
    idx = faiss.IndexIDMap2(base)
    faiss.write_index(idx, str(INDEX_PATH))
    return idx

def save(index: faiss.IndexIDMap2) -> None:
    faiss.write_index(index, str(INDEX_PATH))

def add_vectors(index: faiss.IndexIDMap2, vectors: list[list[float]], ids: list[int]) -> None:
    x = np.array(vectors, dtype="float32")
    x = _normalize(x)
    id_arr = np.array(ids, dtype="int64")
    index.add_with_ids(x, id_arr)

def remove_ids(index: faiss.IndexIDMap2, ids: list[int]) -> None:
    if not ids:
        return
    id_arr = np.array(ids, dtype="int64")
    sel = faiss.IDSelectorBatch(id_arr.size, faiss.swig_ptr(id_arr))
    index.remove_ids(sel)

def search(index: faiss.IndexIDMap2, query_vec: list[float], top_k: int = 5):
    x = np.array([query_vec], dtype="float32")
    x = _normalize(x)
    scores, ids = index.search(x, top_k)
    return scores[0].tolist(), [int(i) for i in ids[0].tolist() if i != -1]
