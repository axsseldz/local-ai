import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

async def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for t in texts:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
            )
            r.raise_for_status()
            data = r.json()
            vectors.append(data["embedding"])
    return vectors
