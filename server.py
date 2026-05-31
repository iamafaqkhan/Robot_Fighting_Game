"""
Local development server.

Production on Vercel uses api/predict.py. Locally:

    python server.py
    open http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.predict import app as predict_app
from rl.inference import get_agent

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Robot Boxing (local)")
app.mount("/api/predict", predict_app)


@app.on_event("startup")
def warmup_model() -> None:
    get_agent()


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/app.js")
async def serve_app_js() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app.js")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
