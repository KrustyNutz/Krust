"""Local REST API for J.A.R.V.I.S — integrate with other apps, webhooks, scripts."""
from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="J.A.R.V.I.S API", version="1.0")

_brain = None
_memory = None


def init(brain, memory):
    global _brain, _memory
    _brain = brain
    _memory = memory


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


class MemoryRequest(BaseModel):
    content: str
    category: str = "general"
    importance: int = 5
    tags: Optional[list[str]] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _brain:
        raise HTTPException(503, "JARVIS not initialized")

    if req.stream:
        async def generate():
            for event_type, payload in _brain.chat(req.message):
                if event_type == "text":
                    yield payload
        return StreamingResponse(generate(), media_type="text/plain")

    parts = []
    for event_type, payload in _brain.chat(req.message):
        if event_type == "text":
            parts.append(payload)
    return {"response": "".join(parts)}


@app.get("/memory/recall")
async def recall(q: str, limit: int = 5):
    if not _memory:
        raise HTTPException(503, "JARVIS not initialized")
    return {"results": _memory.recall(q, limit)}


@app.post("/memory/remember")
async def remember(req: MemoryRequest):
    if not _memory:
        raise HTTPException(503, "JARVIS not initialized")
    mid = _memory.remember(req.content, req.category, req.importance, req.tags)
    return {"id": mid, "stored": req.content}


@app.get("/memory/list")
async def list_memories(category: str = ""):
    if not _memory:
        raise HTTPException(503, "JARVIS not initialized")
    return {"memories": _memory.list_memories(category)}


@app.get("/stats")
async def stats():
    if not _memory:
        raise HTTPException(503, "JARVIS not initialized")
    return _memory.stats()


@app.post("/reset")
async def reset():
    if not _brain:
        raise HTTPException(503, "JARVIS not initialized")
    _brain.reset()
    return {"status": "history cleared"}


def start_api(brain, memory, host: str = "127.0.0.1", port: int = 7777):
    import uvicorn
    init(brain, memory)
    uvicorn.run(app, host=host, port=port, log_level="warning")
