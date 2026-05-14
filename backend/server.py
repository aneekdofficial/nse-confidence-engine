"""
NSE Confidence Engine — API Server
Exposes /signals, /scan, /stream (SSE) and serves the dashboard.
"""

import json, asyncio, time
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import engine (adjust path if running from project root)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from engine import run_scan, save_results, WATCHLIST

DATA_FILE = Path("data/latest.json")
SCAN_INTERVAL = 900   # seconds between auto-scans (15 min)

# ─── Shared state ─────────────────────────────────────────────────────────────
_state = {"signals": [], "updated": None, "scanning": False}


def _load_cached():
    if DATA_FILE.exists():
        try:
            raw = json.loads(DATA_FILE.read_text())
            _state["signals"] = raw.get("signals", [])
            _state["updated"] = raw.get("updated")
        except Exception:
            pass


def _do_scan():
    if _state["scanning"]:
        return
    _state["scanning"] = True
    try:
        results = run_scan()
        save_results(results)
        _state["signals"] = results
        _state["updated"] = datetime.now().isoformat()
    finally:
        _state["scanning"] = False


async def _background_scanner():
    while True:
        await asyncio.get_event_loop().run_in_executor(None, _do_scan)
        await asyncio.sleep(SCAN_INTERVAL)


# ─── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_cached()
    if not _state["signals"]:
        asyncio.get_event_loop().run_in_executor(None, _do_scan)
    task = asyncio.create_task(_background_scanner())
    yield
    task.cancel()

app = FastAPI(title="NSE Confidence Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static frontend if it exists
STATIC_DIR = Path(__file__).parent.parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return JSONResponse({"status": "NSE Confidence Engine running", "endpoints": ["/signals", "/scan", "/stream"]})


@app.get("/signals")
async def get_signals(action: str | None = None, min_conf: float = 0):
    sigs = _state["signals"]
    if action:
        sigs = [s for s in sigs if s["signal"]["action"].upper() == action.upper()]
    if min_conf:
        sigs = [s for s in sigs if s["scores"]["composite"] >= min_conf]
    return {"updated": _state["updated"], "count": len(sigs), "signals": sigs}


@app.post("/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    if _state["scanning"]:
        return {"status": "scan already in progress"}
    background_tasks.add_task(_do_scan)
    return {"status": "scan triggered"}


@app.get("/stream")
async def sse_stream():
    """Server-Sent Events — the dashboard subscribes here for live updates."""
    async def event_gen():
        last_ts = None
        while True:
            ts = _state.get("updated")
            if ts != last_ts and _state["signals"]:
                payload = json.dumps({
                    "updated": ts,
                    "signals": _state["signals"][:50]   # top 50
                })
                yield f"data: {payload}\n\n"
                last_ts = ts
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(10)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health():
    return {
        "ok": True,
        "scanning": _state["scanning"],
        "signal_count": len(_state["signals"]),
        "updated": _state["updated"],
    }
