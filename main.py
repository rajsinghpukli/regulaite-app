# rag/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import ask

app = FastAPI(title="RegulAIte API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok while you test; later you can restrict to your site origin
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    q: str
    mode: Optional[str] = None
    include_web: bool = True
    depth: str = "standard"

@app.post("/api/ask")
def api_ask(inp: AskIn):
    return ask(inp.q, mode_hint=inp.mode, include_web=inp.include_web, depth=inp.depth)

@app.get("/healthz")
def health():
    return {"ok": True}
