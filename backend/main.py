from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Local AI Backend")

class AskRequest(BaseModel):
    question: str
    mode: str = "general"  

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    
    return {
        "answer": f"(stub) You asked: {req.question}",
        "mode": req.mode,
        "sources": [],
        "used_tools": ["stub"]
    }