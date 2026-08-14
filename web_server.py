# web_server.py
# Веб-интерфейс для управления пайплайном кластеризации.
# Starlette + Uvicorn, SSE для live-логов.

import asyncio
import json
import multiprocessing
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import logging
from logger_utils import logger
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from config import cfg
from pipeline_state import PipelineState, STAGES, STAGE_LABELS
from pipeline_progress import progress
try:
    import summarize_only
except ImportError:
    class _DummySummarizeOnly:
        stop_event = None
        def run_summarize_sync(self): return {"ok": False, "error": "summarize_only not found"}
    summarize_only = _DummySummarizeOnly()

pipeline = PipelineState()

DATA_DIR = cfg.ROOT / "PipelineData"
RESET_STATUS_FILE = DATA_DIR / "reset_status.json"
PIPELINE_STATUS_FILE = DATA_DIR / "pipeline_status.json"
STOP_FLAG_FILE = DATA_DIR / "stop.flag"
CONFIRM_FLAG_FILE = DATA_DIR / "confirm.flag"

_pipeline_thread: Optional[threading.Thread] = None
_pipeline_running = False
_pipeline_error: Optional[str] = None
_log_subscribers: list = []
_log_buffer: list = []

_summarize_thread: Optional[threading.Thread] = None
_summarize_running = False
_summarize_error: Optional[str] = None

_reset_process: Optional[threading.Thread] = None
_reset_running = False
_reset_error: Optional[str] = None
_reset_last_log: str = ""

_pipeline_process: Optional[multiprocessing.Process] = None


# ===== Helpers: file-based IPC for subprocess tasks =====

def _read_status(path: Path) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"running": False, "error": None, "last_log": ""}

def _write_status(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to write status file {path}: {e}")

def _read_pipeline_running() -> bool:
    return _read_status(PIPELINE_STATUS_FILE).get("running", False)

def _write_pipeline_running(val: bool):
    data = _read_status(PIPELINE_STATUS_FILE)
    data["running"] = val
    if not val:
        data["error"] = None
        _remove_stop_flag()
    _write_status(PIPELINE_STATUS_FILE, data)

def _set_pipeline_error(err: str):
    data = _read_status(PIPELINE_STATUS_FILE)
    data["running"] = False
    data["error"] = err
    _remove_stop_flag()
    _write_status(PIPELINE_STATUS_FILE, data)

def _set_stop_flag():
    STOP_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG_FILE.write_text("1")

def _remove_stop_flag():
    if STOP_FLAG_FILE.exists():
        STOP_FLAG_FILE.unlink()

def _set_confirm_flag():
    CONFIRM_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_FLAG_FILE.write_text("1")

def _remove_confirm_flag():
    if CONFIRM_FLAG_FILE.exists():
        CONFIRM_FLAG_FILE.unlink()

def _wait_confirm_flag(timeout: float = 1.0) -> bool:
    deadline = time.time() + 100  # 100 sec max
    while time.time() < deadline:
        if CONFIRM_FLAG_FILE.exists():
            return True
        time.sleep(timeout)
    return False

def _is_stopped() -> bool:
    return STOP_FLAG_FILE.exists()


# ===== Subprocess workers =====

def _reset_worker(test_mode: bool, status_path: str):
    """Full reset worker — runs in separate process, no GIL blocking."""
    import json as _json
    from pathlib import Path as _Path
    sp = _Path(status_path)
    def _w(d):
        try:
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, 'w', encoding='utf-8') as f:
                _json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass
    try:
        _w({"running": True, "error": None, "last_log": "Starting reset..."})
        from pipeline_state import PipelineState as PS
        ps = PS()
        _w({"running": True, "error": None, "last_log": "Clearing database..."})
        ps.full_reset(test_mode=test_mode)
        _w({"running": False, "error": None, "last_log": "Reset finished successfully"})
    except Exception as e:
        _w({"running": False, "error": str(e), "last_log": f"Reset failed: {e}"})


def _pipeline_worker(status_path: str, stop_path: str, confirm_path: str):
    """Pipeline worker — runs in separate process."""
    try:
        import asyncio as _aio
        from main import run_pipeline

        # Patch stop_event and confirmation
        import main as _main_mod
        _main_mod.stop_event = type('FakeEvent', (), {'is_set': lambda: os.path.exists(stop_path), 'set': lambda: None, 'clear': lambda: None, 'wait': lambda s, timeout=1.0: os.path.exists(stop_path)})()

        # Patch confirmation event
        _main_mod._stage_confirmation_event = type('FakeEvent', (), {
            'set': lambda: open(confirm_path, 'w').write('1') if not None else None,
            'clear': lambda: os.remove(confirm_path) if os.path.exists(confirm_path) else None,
            'wait': lambda s, timeout=1.0: _wait_confirm_file(confirm_path, timeout),
            'is_set': lambda: os.path.exists(confirm_path),
        })()

        _aio.run(run_pipeline())
        _write_status(Path(status_path), {"running": False, "error": None, "last_log": "Pipeline completed"})
    except Exception as e:
        _write_status(Path(status_path), {"running": False, "error": str(e), "last_log": f"Pipeline failed: {e}"})
    finally:
        for p in [Path(stop_path), Path(confirm_path)]:
            if p.exists():
                p.unlink()


def _wait_confirm_file(path: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    return False

def _get_stop_event():
    from main import stop_event
    return stop_event


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.original_handlers = []
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        ))

    def install(self):
        logger_obj = logging.getLogger("FileOrganizer")
        logger_obj.addHandler(self)
        self.original_handlers = logger_obj.handlers[:]

    def emit(self, record):
        msg = self.format(record)
        _log_buffer.append(msg)
        if len(_log_buffer) > 2000:
            _log_buffer[:] = _log_buffer[-1000:]
        for q in _log_subscribers:
            if hasattr(q, 'put_nowait'):
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass


log_capture = LogCapture()


# ===== Веб-страница =====

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline — Cluster Manager</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #232636;
  --border: #2e3148; --text: #e1e4f0; --text2: #8b8fa8;
  --accent: #6c8cff; --accent2: #4f6fd7;
  --green: #4ade80; --yellow: #fbbf24; --red: #f87171;
  --orange: #fb923c;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
a { color: var(--accent); text-decoration: none; }

/* Layout */
.app { display: flex; min-height: 100vh; }
.sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; position: fixed; top: 0; left: 0; bottom: 0; z-index: 10; }
.sidebar-logo { padding: 20px 16px; border-bottom: 1px solid var(--border); }
.sidebar-logo h1 { font-size: 16px; font-weight: 700; color: var(--accent); }
.sidebar-logo span { font-size: 11px; color: var(--text2); }
.nav { flex: 1; padding: 8px 0; }
.nav a { display: flex; align-items: center; gap: 10px; padding: 10px 16px; color: var(--text2); font-size: 13px; transition: all .15s; }
.nav a:hover, .nav a.active { color: var(--text); background: var(--surface2); }
.nav a.active { border-right: 2px solid var(--accent); color: var(--accent); }
.main { margin-left: 220px; flex: 1; padding: 24px; max-width: 1200px; }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }
.card-header { padding: 12px 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; display: flex; align-items: center; justify-content: space-between; }
.card-body { padding: 16px; }

/* Stages */
.stage-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border); gap: 12px; }
.stage-row:last-child { border-bottom: none; }
.stage-num { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.stage-num.done { background: var(--green); color: #000; }
.stage-num.current { background: var(--accent); color: #fff; animation: pulse 2s infinite; }
.stage-num.pending { background: var(--surface2); color: var(--text2); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .6; } }
.stage-info { flex: 1; }
.stage-info .name { font-size: 13px; font-weight: 600; }
.stage-info .ts { font-size: 11px; color: var(--text2); }
.stage-actions { display: flex; gap: 6px; }

/* Buttons */
.btn { padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; border: none; transition: all .15s; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-danger { background: var(--red); color: #fff; }
.btn-danger:hover { background: #dc2626; }
.btn-warning { background: #d97706; color: #fff; }
.btn-warning:hover { background: #b45309; }
.btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text2); }
.btn-outline:hover { border-color: var(--accent); color: var(--accent); }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-sm { padding: 4px 10px; font-size: 11px; }

/* Stats */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.stat-card { background: var(--surface2); border-radius: 8px; padding: 14px; }
.stat-card .label { font-size: 11px; color: var(--text2); margin-bottom: 4px; }
.stat-card .value { font-size: 22px; font-weight: 700; }

/* Config */
.cfg-row { display: flex; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid var(--border); gap: 12px; }
.cfg-row:last-child { border-bottom: none; }
.cfg-key { width: 220px; font-size: 12px; font-family: 'Consolas', monospace; color: var(--accent); flex-shrink: 0; padding-top: 6px; }
.cfg-val { flex: 1; }
.cfg-val input { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 4px; width: 100%; font-size: 12px; font-family: 'Consolas', monospace; }
.cfg-val input:focus { outline: none; border-color: var(--accent); }
.cfg-hint { margin-top: 6px; padding: 8px 10px; background: rgba(108,140,255,0.06); border-left: 3px solid var(--accent); border-radius: 0 4px 4px 0; }
.hint-desc { display: block; font-size: 11px; color: var(--text); line-height: 1.5; margin-bottom: 4px; }
.hint-tag { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 3px; margin-right: 6px; margin-top: 2px; }
.hint-range { background: rgba(108,140,255,0.15); color: var(--accent); }
.hint-up { background: rgba(74,222,128,0.12); color: var(--green); }
.hint-down { background: rgba(248,113,113,0.12); color: var(--red); }

/* Log */
.log-container { background: #000; border-radius: 6px; padding: 12px; max-height: 500px; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 11px; line-height: 1.6; }
.log-container .log-line { color: #a0a8c0; }
.log-line.error { color: var(--red); }
.log-line.warning { color: var(--yellow); }
.log-line.info { color: var(--green); }

/* Tabs */
.tabs { display: flex; gap: 0; margin-bottom: 16px; }
.tab { padding: 8px 16px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; color: var(--text2); }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Cluster result */
.cluster-card { background: var(--surface2); border-radius: 6px; padding: 12px; margin-bottom: 8px; }
.cluster-card h4 { font-size: 13px; color: var(--accent); margin-bottom: 6px; }
.cluster-card .meta { font-size: 11px; color: var(--text2); }
.cluster-card .files { font-size: 11px; max-height: 60px; overflow-y: auto; margin-top: 6px; color: var(--text2); }

.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700; }
.badge-green { background: rgba(74,222,128,.15); color: var(--green); }
.badge-yellow { background: rgba(251,191,36,.15); color: var(--yellow); }
.badge-red { background: rgba(248,113,113,.15); color: var(--red); }

.progress-bar { height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; margin-top: 12px; }
.progress-fill { height: 100%; background: var(--accent); transition: width .3s; }

/* Sub-stages (extraction groups) */
.substage-container { margin-top: 12px; margin-left: 40px; border-left: 2px solid var(--border); padding-left: 16px; }
.substage-row { display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); gap: 10px; }
.substage-row:last-child { border-bottom: none; }
.substage-icon { width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; }
.substage-icon.done { background: var(--green); color: #000; }
.substage-icon.running { background: var(--accent); color: #fff; animation: pulse 2s infinite; }
.substage-icon.pending { background: var(--surface2); color: var(--text2); }
.substage-icon.current { background: var(--yellow); color: #000; animation: pulse 2s infinite; }
.substage-icon.rolled_back { background: var(--red); color: #fff; }
.substage-icon.stopped { background: var(--orange); color: #fff; }
.substage-icon.skipped { background: var(--text2); color: #000; }
.substage-icon.locked { background: #2a2d3d; color: #565a73; }
.substage-info { flex: 1; }
.substage-info .name { font-size: 12px; font-weight: 600; text-transform: uppercase; }
.substage-info .stats { font-size: 11px; color: var(--text2); }
.substage-actions { display: flex; gap: 4px; align-items: center; }

/* Format blocks (stage 2: two substages per format) */
.fmt-block { background: var(--surface2); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; }
.fmt-title { font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text); letter-spacing: .5px; margin-bottom: 8px; }
.fmt-title .cnt { font-size: 11px; color: var(--text2); font-weight: 400; text-transform: none; letter-spacing: 0; }
.fmt-block .substage-row { border-top: 1px dashed var(--border); border-bottom: none; }
.fmt-block .substage-row:first-of-type { border-top: none; }

.group-confirm-panel { background: var(--surface2); border: 1px solid var(--accent); border-radius: 8px; padding: 14px; margin-top: 12px; margin-left: 40px; }
.group-confirm-panel h4 { font-size: 13px; color: var(--accent); margin-bottom: 10px; }
.group-confirm-panel .stats-row { display: flex; gap: 16px; margin-bottom: 12px; font-size: 12px; }
.group-confirm-panel .stat-item { display: flex; flex-direction: column; }
.group-confirm-panel .stat-label { font-size: 10px; color: var(--text2); }
.group-confirm-panel .stat-value { font-size: 16px; font-weight: 700; }
.group-confirm-panel .actions { display: flex; gap: 8px; }
</style>
</head>
<body>
<div class="app">
  <nav class="sidebar">
    <div class="sidebar-logo">
      <h1>Cluster Pipeline</h1>
      <span>File Organizer</span>
    </div>
    <div class="nav">
      <a href="#" class="active" onclick="showTab('dashboard');return false">Dashboard</a>
      <a href="#" onclick="showTab('stages');return false">Stages</a>
      <a href="#" onclick="showTab('config');return false">Config</a>
      <a href="#" onclick="showTab('logs');return false">Logs</a>
      <a href="#" onclick="showTab('results');return false">Results</a>
    </div>
    <div style="padding:12px 16px;border-top:1px solid var(--border)">
      <button class="btn btn-primary" style="width:100%" id="btn-start" onclick="startPipeline()">Start Pipeline</button>
      <button class="btn btn-danger" style="width:100%;margin-top:6px;display:none" id="btn-stop" onclick="stopPipeline()">Stop</button>
      <button class="btn btn-primary" style="width:100%;margin-top:6px;display:none;background:var(--green)" id="btn-continue" onclick="confirmNextStage()">Continue to Next Stage</button>
      <button class="btn btn-outline" style="width:100%;margin-top:6px;display:none" id="btn-unlock" onclick="forceUnlock()">Force Unlock</button>
      <hr style="border-color:var(--border);margin:12px 0">
      <button class="btn btn-primary" style="width:100%;background:#8b5cf6" id="btn-summarize" onclick="startSummarize()">Summarize Only (LLM)</button>
      <button class="btn btn-danger" style="width:100%;margin-top:6px;display:none" id="btn-summarize-stop" onclick="stopSummarize()">Stop Summarize</button>
    </div>
  </nav>

  <div class="main">
    <!-- Dashboard -->
    <div id="tab-dashboard" class="tab-content">
      <div class="stats-grid" id="stats-grid"></div>
      <div class="card" style="margin-top:16px">
        <div class="card-header">Progress</div>
        <div class="card-body">
          <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
          <div style="margin-top:8px;font-size:12px;color:var(--text2)" id="progress-text">Idle</div>
          <div id="file-progress" style="display:none;margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span id="fp-stage" style="font-weight:600;color:var(--accent)"></span>
              <span id="fp-count" style="color:var(--text2)"></span>
            </div>
            <div class="progress-bar" style="height:8px">
              <div class="progress-fill" id="fp-bar" style="width:0%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text2);margin-top:6px">
              <span id="fp-elapsed"></span>
              <span id="fp-eta"></span>
              <span id="fp-pct"></span>
            </div>
            <div style="font-size:11px;color:var(--text2);margin-top:4px" id="fp-status"></div>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-header">Recent Logs</div>
        <div class="card-body">
          <div class="log-container" id="mini-log" style="max-height:200px"></div>
        </div>
      </div>
    </div>

    <!-- Stages -->
    <div id="tab-stages" class="tab-content" style="display:none">
      <div class="card">
        <div class="card-header">
          Pipeline Stages
          <div style="display:flex;gap:6px">
            <button class="btn btn-outline btn-sm" onclick="refreshStages()">Refresh</button>
            <button class="btn btn-danger btn-sm" onclick="fullReset(false)">Full Reset</button>
            <button class="btn btn-warning btn-sm" onclick="fullReset(true)">Test Reset</button>
          </div>
        </div>
        <div class="card-body" id="stages-list"></div>
          <div id="rollback-status" style="display:none"></div>
          <div id="extraction-groups-container"></div>
          <div id="group-confirm-container"></div>
      </div>
    </div>

    <!-- Config -->
    <div id="tab-config" class="tab-content" style="display:none">
      <div class="card">
        <div class="card-header">
          Configuration
          <div style="display:flex;gap:6px">
            <button class="btn btn-primary btn-sm" onclick="saveConfig()">Save</button>
            <button class="btn btn-outline btn-sm" onclick="loadConfig()">Reset</button>
          </div>
        </div>
        <div class="card-body" id="config-list"></div>
      </div>
    </div>

    <!-- Logs -->
    <div id="tab-logs" class="tab-content" style="display:none">
      <div class="card">
        <div class="card-header">
          Live Logs
          <button class="btn btn-outline btn-sm" onclick="clearLogDisplay()">Clear</button>
        </div>
        <div class="card-body">
          <div class="log-container" id="log-display"></div>
        </div>
      </div>
    </div>

    <!-- Results -->
    <div id="tab-results" class="tab-content" style="display:none">
      <div class="card">
        <div class="card-header">Clustering Report</div>
        <div class="card-body" id="report-content">
          <p style="color:var(--text2)">No report yet. Run the pipeline first.</p>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-header">File Statistics</div>
        <div class="card-body" id="file-stats">
          <p style="color:var(--text2)">No data yet.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let es = null;
let currentTab = 'dashboard';
let pipelineRunning = false;

function updateButtons(running, awaitingConfirmation) {
  pipelineRunning = running;
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnContinue = document.getElementById('btn-continue');
  const btnUnlock = document.getElementById('btn-unlock');
  if (running) {
    btnStart.style.display = 'none';
    btnStop.style.display = 'block';
    btnContinue.style.display = awaitingConfirmation ? 'block' : 'none';
    btnUnlock.style.display = 'none';
  } else {
    btnStart.style.display = 'block';
    btnStop.style.display = 'none';
    btnContinue.style.display = awaitingConfirmation ? 'block' : 'none';
    btnUnlock.style.display = 'block';
  }
}

function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.getElementById('tab-' + name).style.display = 'block';
  document.querySelectorAll('.nav a').forEach(a => a.classList.remove('active'));
  event.target.classList.add('active');
  currentTab = name;
  if (name === 'results') loadReport();
  if (name === 'stages') refreshStages();
  if (name === 'config') loadConfig();
}

function connectSSE() {
  if (es) es.close();
  es = new EventSource('/api/logs/stream');
  es.onmessage = function(e) {
    appendLog(e.data);
  };
  es.onerror = function() { setTimeout(connectSSE, 3000); };
}

function appendLog(msg) {
  const line = document.createElement('div');
  line.className = 'log-line';
  if (msg.includes('ERROR')) line.classList.add('error');
  else if (msg.includes('WARNING')) line.classList.add('warning');
  else if (msg.includes('INFO')) line.classList.add('info');
  line.textContent = msg;
  const miniLog = document.getElementById('mini-log');
  const logDisplay = document.getElementById('log-display');
  if (miniLog) { miniLog.appendChild(line); miniLog.scrollTop = miniLog.scrollHeight; }
  if (logDisplay) { logDisplay.appendChild(line.cloneNode(true)); logDisplay.scrollTop = logDisplay.scrollHeight; }
  if (miniLog && miniLog.children.length > 100) miniLog.removeChild(miniLog.firstChild);
  if (logDisplay && logDisplay.children.length > 3000) logDisplay.removeChild(logDisplay.firstChild);
}

function clearLogDisplay() {
  document.getElementById('log-display').innerHTML = '';
}

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

async function refreshStages() {
  const data = await fetchJSON('/api/status');
  const el = document.getElementById('stages-list');
  el.innerHTML = '';
  let completedCount = 0;
  updateButtons(data.running, data.awaiting_confirmation);

  const stage1Completed = data.stages[0]?.completed;
  const stage2Index = 1;
  const extractionGroups = data.extraction_groups || [];
  const showSubstages = stage1Completed && extractionGroups.length > 0;
  const busy = data.stage2_busy;

  data.stages.forEach((s, i) => {
    if (s.completed) completedCount++;

    if (i === stage2Index && showSubstages) {
      // Заголовок этапа 2
      const numClass = s.completed ? 'done' : (i === data.resume_from ? 'current' : 'pending');
      const headerBadge = s.completed
        ? '<span class="badge badge-green">Done</span>'
        : '<span class="badge badge-yellow">In Progress</span>';
      const headerRow = document.createElement('div');
      headerRow.className = 'stage-row';
      headerRow.innerHTML = `
        <div class="stage-num ${numClass}">${i + 1}</div>
        <div class="stage-info">
          <div class="name">${s.label}</div>
          <div class="ts">произвольный порядок · по одному подэтапу за раз</div>
        </div>
        <div class="stage-actions">${headerBadge}</div>`;
      el.appendChild(headerRow);

      // Блоки форматов с двумя подэтапами
      const cont = document.createElement('div');
      cont.className = 'substage-container';
      extractionGroups.forEach(g => {
        cont.insertAdjacentHTML('beforeend', formatBlockHtml(g, busy));
      });
      el.appendChild(cont);
    } else {
      // Обычный этап
      const row = document.createElement('div');
      row.className = 'stage-row';
      const numClass = s.completed ? 'done' : (i === data.resume_from ? 'current' : 'pending');
      const isResumePoint = (i === data.resume_from);
      const isRunning = data.running;
      let actionsHtml = '';
      if (s.completed) {
        actionsHtml = '<span class="badge badge-green">Done</span> <button class="btn btn-danger btn-sm" onclick="rollbackStage(' + (i + 1) + ')">Rollback from here</button>';
      } else if (isResumePoint) {
        actionsHtml = '<span class="badge badge-yellow">Next</span>';
        if (isRunning) {
          actionsHtml += ' <span style="color:var(--orange);font-size:11px">&#9654; Running</span>';
        } else {
          actionsHtml += ' <button class="btn btn-danger btn-sm" onclick="rollbackStage(' + (i + 1) + ')">Rollback from here</button>';
        }
      } else {
        actionsHtml = '<span class="badge" style="background:var(--surface2);color:var(--text2)">Pending</span>';
      }
      row.innerHTML = `
        <div class="stage-num ${numClass}">${i + 1}</div>
        <div class="stage-info">
          <div class="name">${s.label}</div>
          <div class="ts">${s.completed ? s.timestamp : '—'}</div>
        </div>
        <div class="stage-actions">${actionsHtml}</div>`;
      el.appendChild(row);
    }
  });

  updateProgress(data.stages, completedCount, data.resume_from, data.awaiting_confirmation, data.current_stage_completed);
}

const FORMAT_LABELS = {
  'pdf_text': 'PDF (Text)', 'pdf_scan': 'PDF (Scan)', 'pdf_tables': 'PDF (Tables)',
  'word_docx': 'Word (.docx)', 'word_doc': 'Word (.doc)', 'word_rtf': 'Word (.rtf)',
  'word_txt': 'Word (.txt)', 'word_odt': 'Word (.odt)',
  'excel_xlsx': 'Excel (.xlsx)', 'excel_csv': 'Excel (.csv)', 'excel_ods': 'Excel (.ods)',
  'image_jpg': 'Image (JPG)', 'image_png': 'Image (PNG)', 'image_gif': 'Image (GIF)',
  'image_bmp': 'Image (BMP)', 'image_tiff': 'Image (TIFF)', 'image_webp': 'Image (WEBP)',
  'xml_xml': 'XML (.xml)', 'xml_xsd': 'XML (.xsd)', 'xml_xsl': 'XML (.xsl/.xslt)', 'xml_wsdl': 'XML (.wsdl)',
};

function fmtLabel(t) { return FORMAT_LABELS[t] || (t || '').toUpperCase(); }

function substageRowHtml(g, kind, busy) {
  const status = kind === 'extract' ? g.extract_status : g.embed_status;
  const isImage = (g.type || '').startsWith('image_');
  const nm = kind === 'extract'
    ? (isImage ? 'Извлечение (OCR Tesseract)' : 'Извлечение текста / изображений')
    : 'Построение эмбеддингов';

  let stats = '';
  if (kind === 'extract') {
    if (status === 'completed' || status === 'skipped') stats = `OK: ${g.extract_ok} · ошибок: ${g.extract_errors} · ${g.extract_elapsed}s`;
    else if (status === 'running') stats = `Обработка ${g.total} файлов...`;
    else if (status === 'rolled_back') stats = 'Откат выполнен · файлы возвращены в Sorted';
    else if (status === 'stopped') stats = 'Остановлено';
    else stats = `${g.total} файлов · ${g.workers} воркеров`;
  } else {
    if (status === 'completed' || status === 'skipped') stats = `OK: ${g.embed_ok} · ошибок: ${g.embed_errors} · ${g.embed_elapsed}s`;
    else if (status === 'running') stats = 'Построение эмбеддингов...';
    else if (status === 'locked') stats = 'Доступно после завершения извлечения';
    else if (status === 'rolled_back') stats = 'Откат выполнен';
    else if (status === 'stopped') stats = 'Остановлено';
    else stats = 'Готово к запуску';
  }

  let iconClass = 'pending', icon = '•', badge = '', actions = '';
  const d = busy ? 'disabled' : '';
  if (status === 'completed' || status === 'skipped') {
    iconClass = 'done'; icon = '✓'; badge = '<span class="badge badge-green">Done</span>';
    actions = `<button class="btn btn-danger btn-sm" ${d} onclick="rollbackSubstage('${g.type}','${kind}')">Rollback</button>`;
  } else if (status === 'running') {
    iconClass = 'running'; icon = '▶'; badge = '<span class="badge badge-yellow">Running</span>';
    actions = `<button class="btn btn-danger btn-sm" onclick="stopSubstage()">Stop</button>`;
  } else if (status === 'locked') {
    iconClass = 'locked'; icon = '🔒'; badge = '<span class="badge" style="background:var(--surface2);color:var(--text2)">Locked</span>';
    actions = `<button class="btn btn-primary btn-sm" disabled>Start</button>`;
  } else if (status === 'rolled_back') {
    iconClass = 'rolled_back'; icon = '↺'; badge = '<span class="badge badge-red">Rolled Back</span>';
    actions = `<button class="btn btn-primary btn-sm" ${d} onclick="startSubstage('${g.type}','${kind}')">Start</button>`;
  } else if (status === 'stopped') {
    iconClass = 'stopped'; icon = '■'; badge = '<span class="badge badge-red">Stopped</span>';
    actions = `<button class="btn btn-primary btn-sm" ${d} onclick="startSubstage('${g.type}','${kind}')">Start</button>`;
  } else {
    iconClass = 'pending'; icon = '•'; badge = '<span class="badge" style="background:var(--surface2);color:var(--text2)">Pending</span>';
    actions = `<button class="btn btn-primary btn-sm" ${d} onclick="startSubstage('${g.type}','${kind}')">Start</button>`;
  }

  return `<div class="substage-row">
      <div class="substage-icon ${iconClass}">${icon}</div>
      <div class="substage-info"><div class="name" style="text-transform:none">${nm}</div><div class="stats">${stats}</div></div>
      <div class="substage-actions">${badge} ${actions}</div>
    </div>`;
}

function formatBlockHtml(g, busy) {
  let rows = '';
  if (g.has_extraction) rows += substageRowHtml(g, 'extract', busy);
  rows += substageRowHtml(g, 'embed', busy);
  const onlyNote = g.has_extraction ? '' : ' <span style="font-size:10px;color:var(--text2)">(только эмбеддинги)</span>';
  return `<div class="fmt-block">
      <div class="fmt-title">${fmtLabel(g.type)}${onlyNote} <span class="cnt">${g.total} файлов · ${g.workers} воркеров</span></div>
      ${rows}
    </div>`;
}

async function startSubstage(fmt, kind) {
  try {
    const r = await fetch('/api/substage/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({format: fmt, substage: kind})
    });
    const resp = await r.json();
    if (!resp.ok) alert('Error: ' + (resp.error || 'unknown'));
    refreshStages();
  } catch (e) { alert('Failed: ' + e.message); }
}

async function rollbackSubstage(fmt, kind) {
  const label = kind === 'extract' ? 'извлечения (вместе с эмбеддингами)' : 'эмбеддингов';
  if (!confirm(`Откатить результаты ${label} для «${fmtLabel(fmt)}»?`)) return;
  try {
    const r = await fetch('/api/substage/rollback', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({format: fmt, substage: kind})
    });
    const resp = await r.json();
    if (!resp.ok) alert('Error: ' + (resp.error || 'unknown'));
    refreshStages();
    loadDashboard();
  } catch (e) { alert('Failed: ' + e.message); }
}

async function stopSubstage() {
  if (!confirm('Остановить текущий подэтап?')) return;
  try {
    await fetch('/api/substage/stop', {method: 'POST'});
    refreshStages();
  } catch (e) { alert('Failed: ' + e.message); }
}

function updateProgress(stages, completed, resumeFrom, awaitingConfirmation, currentStageCompleted) {
  const total = stages.length;
  const pct = total > 0 ? Math.round(completed / total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  let txt;
  if (completed === total) {
    txt = 'All stages completed';
  } else if (awaitingConfirmation) {
    txt = 'Awaiting confirmation: ' + currentStageCompleted + ' completed. Click Continue to proceed.';
  } else if (resumeFrom !== null) {
    txt = 'Resume from: ' + stages[resumeFrom].name;
  } else {
    txt = 'Idle';
  }
  document.getElementById('progress-text').textContent = txt + ' (' + completed + '/' + total + ')';
}

async function loadDashboard() {
  const data = await fetchJSON('/api/stats');
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '';
  const items = [
    {label: 'Source Files', value: data.source_files},
    {label: 'Processed', value: data.processed_files},
    {label: 'Errors', value: data.error_files},
    {label: 'Clusters', value: data.clusters},
    {label: 'Sorted Files', value: data.sorted_files},
    {label: 'Text Embeddings', value: data.embeddings_text},
    {label: 'Image Embeddings', value: data.embeddings_image},
  ];
  
  // Phase 2: Статистика по форматам (все 23 формата)
  if (data.format_stats) {
    const formatLabels = {
      'pdf_text': 'PDF Text',
      'pdf_scan': 'PDF Scan',
      'pdf_tables': 'PDF Tables',
      'word_docx': 'Word Docx',
      'word_doc': 'Word Doc',
      'word_rtf': 'Word Rtf',
      'word_txt': 'Word Txt',
      'word_odt': 'Word Odt',
      'excel_xlsx': 'Excel Xlsx',
      'excel_csv': 'Excel Csv',
      'excel_ods': 'Excel Ods',
      'image_jpg': 'Image Jpg',
      'image_png': 'Image Png',
      'image_gif': 'Image Gif',
      'image_bmp': 'Image Bmp',
      'image_tiff': 'Image Tiff',
      'image_webp': 'Image Webp',
      'xml_xml': 'XML Xml',
      'xml_xsd': 'XML Xsd',
      'xml_xsl': 'XML Xsl',
      'xml_wsdl': 'XML Wsdl',
    };
    
    for (const [fmt, count] of Object.entries(data.format_stats)) {
      items.push({
        label: formatLabels[fmt] || fmt,
        value: count
      });
    }
  }
  items.forEach(it => {
    const card = document.createElement('div');
    card.className = 'stat-card';
    card.innerHTML = `<div class="label">${it.label}</div><div class="value">${it.value}</div>`;
    grid.appendChild(card);
  });
  refreshStages();
}

async function loadReport() {
  const data = await fetchJSON('/api/report');
  const el = document.getElementById('report-content');
  if (data.error) { el.innerHTML = '<p style="color:var(--text2)">No report available.</p>'; return; }
  let html = '';
  if (data.clusters) {
    data.clusters.forEach((c, i) => {
      html += `<div class="cluster-card"><h4>Cluster ${c.id} — ${c.count} docs</h4>
        <div class="meta">Similarity: min=${c.min_sim}, max=${c.max_sim}, avg=${c.avg_sim}</div>
        <div class="meta">Formats: ${c.formats}</div>
        ${c.central_docs.length ? '<div class="files">Central: ' + c.central_docs.join(', ') + '</div>' : ''}
      </div>`;
    });
  }
  if (data.summary) html += '<pre style="color:var(--text2);font-size:11px;white-space:pre-wrap;margin-top:12px">' + data.summary + '</pre>';
  el.innerHTML = html || '<p style="color:var(--text2)">No clusters found.</p>';

  const fs = document.getElementById('file-stats');
  if (data.file_stats) {
    let shtml = '<div class="stats-grid">';
    for (const [k, v] of Object.entries(data.file_stats)) {
      shtml += `<div class="stat-card"><div class="label">${k}</div><div class="value">${v}</div></div>`;
    }
    shtml += '</div>';
    fs.innerHTML = shtml;
  }
}

async function loadConfig() {
  const data = await fetchJSON('/api/config');
  const el = document.getElementById('config-list');
  el.innerHTML = '';
  data.forEach(item => {
    const row = document.createElement('div');
    row.className = 'cfg-row';
    const inputType = typeof item.value === 'boolean' ? 'checkbox' : 'text';
    const checkedAttr = typeof item.value === 'boolean' ? (item.value ? 'checked' : '') : '';
    let hintHtml = '';
    if (item.desc) {
      hintHtml = `<div class="cfg-hint"><span class="hint-desc">${item.desc}</span>`;
      if (item.range) hintHtml += `<span class="hint-tag hint-range">Диапазон: ${item.range}</span>`;
      if (item.increase && item.increase !== '—') hintHtml += `<span class="hint-tag hint-up">&uarr; ${item.increase}</span>`;
      if (item.decrease && item.decrease !== '—') hintHtml += `<span class="hint-tag hint-down">&darr; ${item.decrease}</span>`;
      hintHtml += '</div>';
    }
    row.innerHTML = `
      <div class="cfg-key">${item.key}</div>
      <div class="cfg-val">
        ${inputType === 'checkbox'
          ? `<input type="checkbox" data-key="${item.key}" ${checkedAttr} onchange="toggleBool(this)">`
          : `<input type="text" data-key="${item.key}" value="${item.value}">`
        }
        ${hintHtml}
      </div>`;
    el.appendChild(row);
  });
}

function toggleBool(cb) {
    const key = cb.dataset.key;
    if (key) {
        const input = document.querySelector(`input[data-key="${key}"]`);
        if (input && input.type === 'hidden') {
            input.value = cb.checked ? 'true' : 'false';
        }
    }
}

async function saveConfig() {
  const inputs = document.querySelectorAll('#config-list input[data-key]');
  const payload = {};
  inputs.forEach(inp => {
    const key = inp.dataset.key;
    if (inp.type === 'checkbox') {
      payload[key] = inp.checked;
    } else {
      const val = inp.value;
      if (val === 'true') payload[key] = true;
      else if (val === 'false') payload[key] = false;
      else if (/^\d+$/.test(val)) payload[key] = parseInt(val);
      else if (/^\d+\.\d+$/.test(val)) payload[key] = parseFloat(val);
      else payload[key] = val;
  }
  });
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const resp = await r.json();
  if (resp.ok) alert('Config saved.');
  else alert('Error: ' + resp.error);
}

async function startPipeline() {
  const r = await fetch('/api/pipeline/start', {method: 'POST'});
  const resp = await r.json();
  if (resp.ok) {
    updateButtons(true);
  } else {
    alert(resp.error || 'Pipeline already running');
  }
}

async function stopPipeline() {
  const btn = document.getElementById('btn-stop');
  btn.disabled = true;
  btn.textContent = 'Stopping...';
  await fetch('/api/pipeline/stop', {method: 'POST'});
  btn.textContent = 'Stopped';
  btn.classList.remove('btn-danger');
  btn.classList.add('btn-outline');
  setTimeout(() => {
    updateButtons(false);
    btn.disabled = false;
    btn.textContent = 'Stop';
    btn.classList.remove('btn-outline');
    btn.classList.add('btn-danger');
  }, 2000);
}

async function forceUnlock() {
  if (!confirm('Force unlock? Use this only if the pipeline appears stuck after a crash.')) return;
  const r = await fetch('/api/pipeline/unlock', {method: 'POST'});
  const resp = await r.json();
  if (resp.ok) {
    alert('Pipeline unlocked. You can now start or rollback.');
    updateButtons(false);
    refreshStages();
  } else {
    alert('Error: ' + (resp.error || 'unknown'));
  }
}

let summarizeRunning = false;

async function startSummarize() {
  const r = await fetch('/api/summarize/start', {method: 'POST'});
  const resp = await r.json();
  if (resp.ok) {
    summarizeRunning = true;
    document.getElementById('btn-summarize').style.display = 'none';
    document.getElementById('btn-summarize-stop').style.display = 'block';
  } else {
    alert(resp.error || 'Failed to start summarize');
  }
}

async function stopSummarize() {
  document.getElementById('btn-summarize-stop').disabled = true;
  document.getElementById('btn-summarize-stop').textContent = 'Stopping...';
  await fetch('/api/summarize/stop', {method: 'POST'});
  document.getElementById('btn-summarize-stop').textContent = 'Stopped';
  setTimeout(() => {
    summarizeRunning = false;
    document.getElementById('btn-summarize').style.display = 'block';
    document.getElementById('btn-summarize-stop').style.display = 'none';
    document.getElementById('btn-summarize-stop').disabled = false;
    document.getElementById('btn-summarize-stop').textContent = 'Stop Summarize';
  }, 2000);
}

// Poll summarize status every 3s
setInterval(async () => {
  if (!summarizeRunning) return;
  try {
    const data = await fetchJSON('/api/summarize/status');
    if (!data.running) {
      summarizeRunning = false;
      document.getElementById('btn-summarize').style.display = 'block';
      document.getElementById('btn-summarize-stop').style.display = 'none';
      if (data.error) {
        appendLog('ERROR - Summarize failed: ' + data.error);
      } else {
        appendLog('INFO - Summarize completed');
      }
    }
  } catch (e) {}
}, 3000);

async function confirmNextStage() {
  const btn = document.getElementById('btn-continue');
  btn.disabled = true;
  btn.textContent = 'Continuing...';
  try {
    const r = await fetch('/api/pipeline/confirm', {method: 'POST'});
    const resp = await r.json();
    if (resp.ok) {
      btn.textContent = 'Continued!';
      setTimeout(() => {
        btn.disabled = false;
        btn.textContent = 'Continue to Next Stage';
        btn.style.display = 'none';
        refreshStages();
      }, 1000);
    } else {
      alert('Error: ' + (resp.error || 'unknown'));
      btn.disabled = false;
      btn.textContent = 'Continue to Next Stage';
    }
  } catch (e) {
    alert('Failed: ' + e.message);
    btn.disabled = false;
    btn.textContent = 'Continue to Next Stage';
  }
}

async function rollbackStage(n) {
  const stageLabels = ['Sort', 'Process Formats', 'Cluster + Refine'];
  if (!confirm('Rollback from stage ' + n + '? This will undo stages ' + n + '-' + stageLabels.length + ' and their artifacts.')) return;

  const statusEl = document.getElementById('rollback-status');
  statusEl.style.display = 'block';
  statusEl.style.background = 'var(--surface2)';
  statusEl.style.color = 'var(--accent)';
  statusEl.style.padding = '10px 14px';
  statusEl.style.borderRadius = '6px';
  statusEl.style.marginBottom = '12px';
  statusEl.style.fontSize = '13px';
  statusEl.textContent = 'Rolling back stages ' + n + '-' + stageLabels.length + '...';

  try {
    const r = await fetch('/api/pipeline/rollback?stage=' + n + '&mode=full', {method: 'POST'});
    const resp = await r.json();
    if (resp.ok) {
      const undone = [];
      for (let i = n - 1; i < stageLabels.length; i++) undone.push((i + 1) + '. ' + stageLabels[i]);
      statusEl.style.background = 'rgba(74,222,128,0.12)';
      statusEl.style.color = 'var(--green)';
      statusEl.textContent = 'Rollback complete! Undone: ' + undone.join(', ');
      refreshStages();
      loadDashboard();
    } else {
      statusEl.style.background = 'rgba(248,113,113,0.12)';
      statusEl.style.color = 'var(--red)';
      statusEl.textContent = 'Rollback error: ' + (resp.error || 'unknown');
    }
  } catch (e) {
    statusEl.style.background = 'rgba(248,113,113,0.12)';
    statusEl.style.color = 'var(--red)';
    statusEl.textContent = 'Rollback failed: ' + e.message;
  }
}

async function fullReset(testMode) {
  console.log('fullReset called, testMode=' + testMode);
  const label = testMode ? 'TEST reset (21 files from SourceFiles_test.rar)?' : 'Full reset (all data deleted, source files restored from SourceFiles.rar)?';
  if (!confirm(label)) { console.log('Reset cancelled by user'); return; }
  console.log('Reset confirmed');
  const statusEl = document.getElementById('rollback-status');
  if (!statusEl) { alert('Status element not found'); return; }
  statusEl.style.display = 'block';
  statusEl.style.background = 'var(--surface2)';
  statusEl.style.color = 'var(--accent)';
  statusEl.style.padding = '10px 14px';
  statusEl.style.borderRadius = '6px';
  statusEl.style.marginBottom = '12px';
  statusEl.style.fontSize = '13px';
  statusEl.textContent = testMode ? 'Resetting with test data...' : 'Resetting all pipeline data...';
  console.log('Status element updated, sending request...');
  try {
    const url = testMode ? '/api/pipeline/reset?test=1' : '/api/pipeline/reset';
    console.log('POST ' + url);
    const r = await fetch(url, {method: 'POST'});
    console.log('Response status: ' + r.status);
    const resp = await r.json();
    if (!resp.ok) {
      statusEl.style.background = 'rgba(248,113,113,0.12)';
      statusEl.style.color = 'var(--red)';
      statusEl.textContent = 'Reset error: ' + (resp.error || 'unknown');
      return;
    }
    // poll reset status
    const poll = setInterval(async () => {
      try {
        const s = await fetch('/api/pipeline/reset/status').then(x => x.json());
        statusEl.textContent = s.last_log || (s.running ? 'Reset in progress...' : 'Reset finished');
        if (!s.running) {
          clearInterval(poll);
          if (s.error) {
            statusEl.style.background = 'rgba(248,113,113,0.12)';
            statusEl.style.color = 'var(--red)';
            statusEl.textContent = 'Reset error: ' + s.error;
          } else {
            statusEl.style.background = 'rgba(74,222,128,0.12)';
            statusEl.style.color = 'var(--green)';
            statusEl.textContent = testMode
              ? 'Test reset complete! 21 test files restored from SourceFiles_test.rar.'
              : 'Full reset complete! All data deleted, source files restored from archive.';
          }
          refreshStages();
          loadDashboard();
        }
      } catch (e) {
        clearInterval(poll);
        statusEl.style.background = 'rgba(248,113,113,0.12)';
        statusEl.style.color = 'var(--red)';
        statusEl.textContent = 'Reset status poll failed: ' + e.message;
      }
    }, 1000);
  } catch (e) {
    statusEl.style.background = 'rgba(248,113,113,0.12)';
    statusEl.style.color = 'var(--red)';
    statusEl.textContent = 'Reset failed: ' + e.message;
  }
}

setInterval(() => { if (currentTab === 'dashboard') loadDashboard(); }, 5000);
setInterval(() => { if (currentTab === 'stages') refreshStages(); }, 5000);
setInterval(() => { if (currentTab === 'results') loadReport(); }, 10000);

// File-level progress polling (every 2s always)
setInterval(updateFileProgress, 2000);

function formatDuration(secs) {
  if (!secs || secs <= 0) return '—';
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}

async function updateFileProgress() {
  const fp = document.getElementById('file-progress');
  try {
    const data = await fetchJSON('/api/progress');
    if (!data || data.stage === 0 || data.total === 0) {
      fp.style.display = 'none';
      return;
    }
    fp.style.display = 'block';
    document.getElementById('fp-stage').textContent = data.stage_label;
    document.getElementById('fp-count').textContent = data.completed + ' / ' + data.total;
    document.getElementById('fp-bar').style.width = data.percentage + '%';
    document.getElementById('fp-pct').textContent = data.percentage + '%';
    document.getElementById('fp-elapsed').textContent = 'Elapsed: ' + formatDuration(data.elapsed);
    document.getElementById('fp-eta').textContent = data.eta != null ? 'ETA: ' + formatDuration(data.eta) : 'ETA: —';
    document.getElementById('fp-status').textContent = data.status || '';
  } catch (e) {
    fp.style.display = 'none';
  }
}

connectSSE();
loadDashboard();
refreshStages();
</script>
</body>
</html>"""


# ===== API обработчики =====

async def api_index(request):
    return HTMLResponse(HTML_PAGE)


async def api_status(request):
    from main import _awaiting_confirmation, _current_stage_completed
    from stages import stage2_processing as s2
    state = pipeline._load_state()
    stages_info = []
    for s in STAGES:
        info = state.get("stages", {}).get(s, {})
        stages_info.append({
            "name": s,
            "label": STAGE_LABELS[s],
            "completed": info.get("completed", False),
            "timestamp": info.get("timestamp", None),
        })
    resume = pipeline.get_resume_point()
    resume_from = STAGES.index(resume) if resume else None

    return JSONResponse({
        "stages": stages_info,
        "running": _pipeline_running or _read_pipeline_running(),
        "error": _pipeline_error,
        "resume_from": resume_from,
        "awaiting_confirmation": _awaiting_confirmation,
        "current_stage_completed": _current_stage_completed,
        "awaiting_substage_select": s2._awaiting_substage_select,
        "stage2_active": list(s2._stage2_active) if s2._stage2_active else None,
        "stage2_busy": s2._stage2_busy,
        "extraction_groups": s2._extraction_groups,
        "current_group_index": s2._current_group_index,
    })


async def api_stats(request):
    source_count = 0
    if cfg.SOURCE_DIR.exists():
        source_count = sum(1 for _ in cfg.SOURCE_DIR.rglob("*.*") if _.is_file())

    sorted_count = 0
    for format_target in cfg.FORMAT_TARGETS.values():
        target_dir = cfg.ROOT / format_target
        if target_dir.exists():
            sorted_count += sum(1 for _ in target_dir.rglob("*") if _.is_file())

    error_count = 0
    if cfg.ERRORS_DIR.exists():
        error_count = sum(1 for _ in cfg.ERRORS_DIR.rglob("*") if _.is_file())

    extracted_count = 0
    if cfg.EXTRACT_ROOT.exists():
        extracted_count = sum(1 for _ in cfg.EXTRACT_ROOT.rglob("*") if _.is_file())

    emb_text = 0
    emb_image = 0
    try:
        from database import get_db
        db = get_db()
        stats = db.get_stats()
        emb_text = stats.get("text_emb", 0)
        emb_image = stats.get("img_emb", 0)
    except Exception:
        pass

    if not emb_text and not emb_image:
        npy_count = 0
        if cfg.EMBEDDINGS_DIR.exists():
            npy_count = sum(1 for _ in cfg.EMBEDDINGS_DIR.rglob("*.npy") if _.is_file())
        emb_text = npy_count

    cluster_count = 0
    clusters_dir = cfg.ROOT / "Clusters"
    if clusters_dir.exists():
        subdirs = [d for d in clusters_dir.iterdir() if d.is_dir()]
        cluster_count = len(subdirs)

    processed_count = 0
    stage2_data = pipeline.load_data("stage_2_process_formats")
    if stage2_data:
        processed_count = len(stage2_data)

    # Phase 2: Статистика по форматам (все 23 формата)
    format_stats = {}
    for format_type in cfg.FORMAT_TARGETS.keys():
        target_dir = cfg.ROOT / cfg.FORMAT_TARGETS[format_type]
        if target_dir.exists():
            file_count = sum(1 for _ in target_dir.rglob("*") if _.is_file())
            format_stats[format_type] = file_count
    
    return JSONResponse({
        "source_files": source_count,
        "sorted_files": sorted_count,
        "processed_files": processed_count,
        "error_files": error_count,
        "extracted_files": extracted_count,
        "embeddings_count": emb_text + emb_image,
        "embeddings_text": emb_text,
        "embeddings_image": emb_image,
        "clusters": cluster_count,
        "format_stats": format_stats,  # Phase 1: статистика по форматам (PDF PoC)
    })


async def api_progress(request):
    return JSONResponse(progress.get_snapshot())


async def api_report(request):
    report_path = cfg.ROOT / "report.txt"
    if not report_path.exists():
        return JSONResponse({"error": "no_report"})

    text = report_path.read_text(encoding="utf-8")
    clusters = []
    file_stats = {}
    summary_lines = []

    current_cluster = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("--- Cluster"):
            if current_cluster:
                clusters.append(current_cluster)
            parts = line.replace("---", "").replace("Cluster", "").strip().split()
            cid = parts[0] if parts else "?"
            current_cluster = {"id": cid, "count": 0, "min_sim": 0, "max_sim": 0, "avg_sim": 0, "formats": "", "central_docs": []}
        elif current_cluster is not None:
            if line.startswith("Total Documents:"):
                current_cluster["count"] = int(line.split(":")[1].strip())
            elif line.startswith("Formats:"):
                current_cluster["formats"] = line.split(":", 1)[1].strip()
                try:
                    file_stats.update(eval(line.split(":", 1)[1].strip()))
                except Exception:
                    pass
            elif line.startswith("Similarity Stats"):
                parts = line.split("->", 1)[1] if "->" in line else ""
                for token in parts.replace(",", " ").split():
                    if token.startswith("Min:"):
                        current_cluster["min_sim"] = token.replace("Min:", "")
                    elif token.startswith("Max:"):
                        current_cluster["max_sim"] = token.replace("Max:", "")
                    elif token.startswith("Avg:"):
                        current_cluster["avg_sim"] = token.replace("Avg:", "")
            elif line.startswith("Central Documents:"):
                docs = line.split(":", 1)[1].strip()
                if docs:
                    current_cluster["central_docs"] = [d.strip() for d in docs.split(",")]
        else:
            summary_lines.append(line)

    if current_cluster:
        clusters.append(current_cluster)

    return JSONResponse({
        "clusters": clusters,
        "summary": "\n".join(summary_lines),
        "file_stats": file_stats,
    })


CONFIG_HINTS = {
    "SIMILARITY_THRESHOLD": {
        "desc": "Порог сходства для кластеризации. Определяет, насколько похожими должны быть документы, чтобы попасть в один кластер.",
        "range": "0.0 — 1.0",
        "increase": "Больше порог → больше мелких плотных кластеров (строгая группировка)",
        "decrease": "Меньше порог → меньше крупных рыхлых кластеров (мягкая группировка)",
    },
    "TOP_N_CLUSTERS": {
        "desc": "Количество лучших кластеров, которые будут вынесены в отдельные папки. Остальные документы не перемещаются.",
        "range": "1 — 100",
        "increase": "Больше кластеров → больше папок, каждый кластер содержит меньше документов",
        "decrease": "Меньше кластеров → меньше папок, каждый кластер содержит больше документов",
    },
    "MAX_CONCURRENT_FILES": {
        "desc": "Количество параллельных процессов извлечения данных (PDF, Office). Каждый процесс загружает CPU и оперативную память.",
        "range": "1 — 16",
        "increase": "Быстрее, но выше нагрузка на CPU/RAM, возможны конфликты LibreOffice",
        "decrease": "Медленнее, но стабильнее для больших файлов",
    },
    "LO_MAX_WORKERS": {
        "desc": "Максимум параллельных процессов LibreOffice для конвертации Office→PDF. Ограничивает одновременные вызовы soffice.exe.",
        "range": "1 — 6",
        "increase": "Быстрее конвертация, но выше потребление RAM (каждый процесс ~200 МБ)",
        "decrease": "Медленнее, но надёжнее при нестабильной конвертации",
    },
    "COM_ENABLED": {
        "desc": "Использовать Microsoft Office (Word/Excel) через COM для конвертации документов. Быстрее LibreOffice в 2-5 раз.",
        "range": "true / false",
        "increase": "—",
        "decrease": "Выключить — вернуться к LibreOffice",
    },
    "COM_WORD_WORKERS": {
        "desc": "Максимум параллельных процессов Word через COM (fallback после Aspose). Ограничивает одновременные WinWord.exe.",
        "range": "1 — 6",
        "increase": "Быстрее fallback, но больше одновременных процессов WinWord.exe (~100-200 МБ RAM каждый)",
        "decrease": "Меньше нагрузки на систему, медленнее",
    },
    "ASPOSE_WORKERS": {
        "desc": "Максимум параллельных воркеров для Word/RTF файлов (Aspose.Words for Java). Каждый запускает отдельный JVM-процесс.",
        "range": "1 — 16",
        "increase": "Быстрее обработка, но выше потребление RAM (~300 МБ на JVM)",
        "decrease": "Меньше нагрузки, медленнее",
    },
    "COM_EXCEL_WORKERS": {
        "desc": "Максимум параллельных конверсий Excel (xlsx, xls, csv, ods) через COM.",
        "range": "1 — 6",
        "increase": "Быстрее, но больше процессов Excel.exe",
        "decrease": "Меньше нагрузки, медленнее",
    },
    "EMB_BATCH_SIZE": {
        "desc": "Размер батча при генерации эмбеддингов. Сколько текстов/изображений обрабатывается за один проход модели.",
        "range": "1 — 128",
        "increase": "Выше throughput на GPU, но больше потребление видеопамяти",
        "decrease": "Меньше видеопамяти, но медленнее обработка",
    },
    "EMB_GPU_DEVICE": {
        "desc": "CUDA-устройство для генерации эмбеддингов. cuda:0 — основная видеокарта, cuda:1 — вторая.",
        "range": "cuda:0, cuda:1, cpu",
        "increase": "—",
        "decrease": "—",
    },
    "LLM_ENABLED": {
        "desc": "Включить/выключить LLM саммаризацию через LM Studio. Если выключено — используется исходный текст.",
        "range": "true / false",
        "increase": "—",
        "decrease": "—",
    },
    "LLM_MIN_TEXT_LENGTH": {
        "desc": "Минимальная длина текста (в символах), при которой документ отправляется на саммаризацию. Короткие тексты пропускаются.",
        "range": "0 — 1000",
        "increase": "Меньше документов саммаризируется (только длинные)",
        "decrease": "Больше документов саммаризируется (включая короткие)",
    },
    "LLM_MAX_TEXT_LENGTH": {
        "desc": "Максимум символов текста, отправляемых в LLM. Тексты длиннее этого значения обрезаются (начало + конец).",
        "range": "1000 — 50000",
        "increase": "Больше контекста для LLM, но медленнее и дороже",
        "decrease": "Меньше контекста, быстрее, но теряется информация",
    },
    "HF_MODEL_PATH": {
        "desc": "Путь к модели Qwen3.5-4B (HF transformers).",
        "range": "Путь к папке",
        "increase": "—",
        "decrease": "—",
    },
    "HF_BATCH_SIZE": {
        "desc": "Размер батча при саммаризации через transformers. Больше = быстрее, но выше расход VRAM.",
        "range": "1 — 32",
        "increase": "Быстрее, больше VRAM",
        "decrease": "Медленнее, меньше VRAM",
    },
    "HF_MAX_TOKENS": {
        "desc": "Максимум новых токенов при генерации Qwen3.5-4B.",
        "range": "64 — 2048",
        "increase": "Длиннее ответ, больше деталей",
        "decrease": "Короче ответ, меньше шанс мусора",
    },
    "DIST_MATRIX_CHUNK": {
        "desc": "Размер чанка при вычислении матрицы расстояний на GPU. При большом количестве документов (>5000) матрица вычисляется блоками.",
        "range": "1000 — 10000",
        "increase": "Меньше блоков, но выше пиковое потребление видеопамяти",
        "decrease": "Меньше видеопамяти, но больше блоков и накладных расходов",
    },
    "EMB_DIMENSION": {
        "desc": "Размерность вектора эмбеддингов. Должна совпадать с выходом модели (768 для nomic-embed).",
        "range": "128, 384, 768",
        "increase": "— (определяется моделью)",
        "decrease": "— (определяется моделью)",
    },
    "PG_HOST": {
        "desc": "Хост PostgreSQL для хранения эмбеддингов (pgvector).",
        "range": "IP или hostname",
        "increase": "—",
        "decrease": "—",
    },
    "PG_PORT": {
        "desc": "Порт PostgreSQL.",
        "range": "1 — 65535",
        "increase": "—",
        "decrease": "—",
    },
    "PG_DB": {
        "desc": "Имя базы данных PostgreSQL.",
        "range": "Строка",
        "increase": "—",
        "decrease": "—",
    },
    "PG_USER": {
        "desc": "Пользователь PostgreSQL для подключения.",
        "range": "Строка",
        "increase": "—",
        "decrease": "—",
    },
}


async def api_config_get(request):
    import dataclasses
    fields = dataclasses.fields(cfg)
    result = []
    editable_keys = {
        "SIMILARITY_THRESHOLD", "TOP_N_CLUSTERS", "MAX_CONCURRENT_FILES",
        "LO_MAX_WORKERS", "COM_ENABLED", "COM_WORD_WORKERS", "COM_EXCEL_WORKERS",
        "ASPOSE_WORKERS",
        "EMB_BATCH_SIZE", "EMB_GPU_DEVICE",
        "LLM_ENABLED", "LLM_MIN_TEXT_LENGTH", "LLM_MAX_TEXT_LENGTH",
        "HF_MODEL_PATH", "HF_BATCH_SIZE", "HF_MAX_TOKENS",
        "DIST_MATRIX_CHUNK", "EMB_DIMENSION",
        "PG_HOST", "PG_PORT", "PG_DB", "PG_USER",
    }
    for f in fields:
        if f.name in editable_keys:
            val = getattr(cfg, f.name)
            if isinstance(val, Path):
                val = str(val)
            hint = CONFIG_HINTS.get(f.name, {})
            result.append({
                "key": f.name,
                "value": val,
                "type": type(val).__name__,
                "desc": hint.get("desc", ""),
                "range": hint.get("range", ""),
                "increase": hint.get("increase", ""),
                "decrease": hint.get("decrease", ""),
            })
    return JSONResponse(result)


async def api_config_set(request):
    import dataclasses
    body = await request.json()
    fields_map = {f.name: f for f in dataclasses.fields(cfg)}
    for key, value in body.items():
        if key in fields_map and hasattr(cfg, key):
            try:
                field_type = fields_map[key].type
                if field_type == Path:
                    value = Path(value)
                elif field_type == bool:
                    value = bool(value)
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                object.__setattr__(cfg, key, value)
            except Exception:
                pass
    _save_config_to_file(body)
    return JSONResponse({"ok": True})


def _save_config_to_file(updates: dict):
    config_path = Path(__file__).parent / "config_override.json"
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(updates)
    config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


async def api_pipeline_start(request):
    global _pipeline_running, _pipeline_error
    if _pipeline_running or _read_pipeline_running():
        return JSONResponse({"ok": False, "error": "Pipeline is already running"})

    _pipeline_running = True
    _pipeline_error = None
    _remove_stop_flag()
    _remove_confirm_flag()
    _write_pipeline_running(True)

    def run():
        global _pipeline_running, _pipeline_error
        import asyncio as _aio
        from main import run_pipeline
        try:
            _aio.run(run_pipeline())
        except Exception as e:
            _pipeline_error = str(e)
        finally:
            _pipeline_running = False
            _write_pipeline_running(False)
            _remove_stop_flag()
            _remove_confirm_flag()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    logger.info("Pipeline thread started")

    return JSONResponse({"ok": True})


async def api_pipeline_stop(request):
    _set_stop_flag()
    return JSONResponse({"ok": True})


async def api_pipeline_unlock(request):
    global _pipeline_running, _pipeline_error
    _pipeline_running = False
    _pipeline_error = None
    _remove_stop_flag()
    _remove_confirm_flag()
    _write_pipeline_running(False)
    return JSONResponse({"ok": True})


async def api_pipeline_confirm(request):
    _set_confirm_flag()
    return JSONResponse({"ok": True})


async def api_substage_start(request):
    """Запуск подэтапа (извлечение/эмбеддинги) для конкретного формата.

    body: {"format": "word_docx", "substage": "extract"|"embed"}
    Если пайплайн не запущен — запускает его и дожидается ожидания выбора подэтапа.
    """
    from stages import stage2_processing as s2

    body = await request.json()
    fmt = body.get("format")
    sub = body.get("substage")

    if not fmt or sub not in ("extract", "embed"):
        return JSONResponse({"ok": False, "error": "format and substage ('extract'|'embed') required"})

    if s2._stage2_busy:
        active = list(s2._stage2_active) if s2._stage2_active else None
        return JSONResponse({"ok": False, "error": f"Подэтап уже выполняется: {active}"})

    # Запускаем пайплайн, если он не работает.
    if not _pipeline_running and not _read_pipeline_running():
        resp = await api_pipeline_start(request)
        if isinstance(resp, JSONResponse):
            try:
                data = json.loads(resp.body)
                if not data.get("ok", True):
                    return resp
            except Exception:
                pass

    # Ждём, пока пайплайн дойдёт до ожидания выбора подэтапа (макс 120 сек).
    # Если отправить подэтап ДО этого — он уйдёт в _pending_substage и повиснет,
    # если пайплайн уже успел пройти проверку _pending_substage (race condition).
    for _ in range(240):
        if s2._awaiting_substage_select:
            break
        await asyncio.sleep(0.5)

    if not s2._awaiting_substage_select:
        return JSONResponse({"ok": False, "error":
            "Пайплайн не дошёл до этапа 2 за 120 сек. "
            "Проверьте логи — возможно, этап 1 не выполнен или произошла ошибка."})

    # Пайплайн ждёт выбора — отправляем немедленно
    ok = s2.select_substage_to_start(fmt, sub)
    if not ok:
        return JSONResponse({"ok": False, "error": "Недопустимый выбор (формат неизвестен или подэтап заблокирован)"})

    return JSONResponse({"ok": True, "format": fmt, "substage": sub})


async def api_substage_rollback(request):
    """Откат результатов подэтапа формата.

    body: {"format": "word_docx", "substage": "extract"|"embed"}
    extract → откат извлечения (и зависимых эмбеддингов), возврат файлов в Sorted.
    embed   → откат только эмбеддингов (извлечение сохраняется).
    """
    from stages import stage2_processing as s2

    body = await request.json()
    fmt = body.get("format")
    sub = body.get("substage")

    if not fmt or sub not in ("extract", "embed"):
        return JSONResponse({"ok": False, "error": "format and substage ('extract'|'embed') required"})

    if s2._stage2_busy:
        return JSONResponse({"ok": False, "error": "Нельзя откатывать, пока выполняется подэтап"})

    grp = s2._find_group(fmt)
    if grp is None:
        return JSONResponse({"ok": False, "error": f"Неизвестный формат: {fmt}"})

    sorted_data = pipeline.load_data("stage_1_sort") or []
    result = {"ok": False, "error": ""}

    def do_rollback():
        try:
            if sub == "extract":
                s2._rollback_extraction(fmt, sorted_data)
                grp["extract_status"] = "rolled_back"
                grp["embed_status"] = "locked"
                grp["extract_ok"] = grp["extract_errors"] = 0
                grp["extract_elapsed"] = 0.0
                grp["embed_ok"] = grp["embed_errors"] = 0
                grp["embed_elapsed"] = 0.0
            else:
                s2._rollback_embeddings(fmt, sorted_data)
                grp["embed_status"] = "rolled_back"
                grp["embed_ok"] = grp["embed_errors"] = 0
                grp["embed_elapsed"] = 0.0
            s2._persist_stage2_state()
            pipeline.mark_incomplete("stage_2_process_formats")
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, do_rollback)
    return JSONResponse(result)


async def api_substage_stop(request):
    """Остановка текущего выполняющегося подэтапа."""
    from stages import stage2_processing as s2
    _get_stop_event().set()
    s2._stage2_busy = False
    return JSONResponse({"ok": True})


async def api_pipeline_rollback(request):
    global _pipeline_running
    if _pipeline_running or _read_pipeline_running():
        return JSONResponse({"ok": False, "error": "Pipeline is running, cannot rollback"})

    stage_num = int(request.query_params.get("stage", "1"))
    mode = request.query_params.get("mode", "full")
    soft = (mode == "soft")
    stage_name = STAGES[stage_num - 1]
    result = {"ok": False, "error": ""}

    def do_rollback():
        try:
            pipeline.rollback_from(stage_name, soft=soft)
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

    import concurrent.futures
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, do_rollback)

    return JSONResponse(result)


async def api_pipeline_reset(request):
    global _reset_process, _reset_running
    # Only block if there is an actively running pipeline THREAD in this process.
    # Stale pipeline_status.json is ignored — reset is a destructive override that
    # clears all state unconditionally.
    if _pipeline_running:
        return JSONResponse({"ok": False, "error": "Pipeline is running, cannot reset"})
    # Clear any stale pipeline/stop/confirm flags before starting the reset.
    _write_pipeline_running(False)
    _remove_stop_flag()
    _remove_confirm_flag()

    # Derive running state from the actual process so a stale flag can't block forever.
    _reset_running = bool(_reset_process and _reset_process.is_alive())
    if _reset_running:
        return JSONResponse({"ok": False, "error": "Reset is already running"})

    test_mode = request.query_params.get("test", "").lower() in ("1", "true", "yes")
    logger.info(f"Full reset requested (test_mode={test_mode})")

    _reset_running = True
    _write_status(RESET_STATUS_FILE, {"running": True, "error": None, "last_log": "Reset started..."})

    # Run in a daemon thread (same process) instead of a separate process, so that
    # full_reset's log messages propagate to the web server's logger -> console and
    # the web UI Logs panel (LogCapture). full_reset is I/O-bound (taskkill, unrar,
    # file deletion, DB via its own pooled connection), so it does not block the
    # event loop in a meaningful way. Mirrors the pipeline, which also runs in a thread.
    _reset_process = threading.Thread(
        target=_reset_worker,
        args=(test_mode, str(RESET_STATUS_FILE)),
        daemon=True,
    )
    _reset_process.start()
    logger.info("Reset thread started")

    return JSONResponse({"ok": True, "message": "Reset started", "test_mode": test_mode})


async def api_pipeline_reset_status(request):
    global _reset_running
    data = _read_status(RESET_STATUS_FILE)

    # full_reset() deletes the whole PipelineData directory (including RESET_STATUS_FILE)
    # in the middle of the reset, so the status file is temporarily absent while files are
    # still being restored. Therefore the reset THREAD's liveness — not the (possibly
    # deleted) status file — is the source of truth for whether the reset is still running.
    # This guarantees "Reset complete" is reported only after the thread (and thus the full
    # restore from the archive) has actually finished.
    thread_alive = bool(_reset_process and _reset_process.is_alive())

    if thread_alive:
        _reset_running = True
        last_log = data.get("last_log") or "Reset in progress..."
        return JSONResponse({
            "ok": True,
            "running": True,
            "error": None,
            "last_log": last_log,
        })

    # Thread has finished (or none started). The worker rewrites the status file with the
    # final result at the very end, so trust it now.
    _reset_running = False
    return JSONResponse({
        "ok": True,
        "running": False,
        "error": data.get("error"),
        "last_log": data.get("last_log", ""),
    })


async def api_summarize_start(request):
    global _summarize_running, _summarize_error, _summarize_thread
    if _pipeline_running or _read_pipeline_running():
        return JSONResponse({"ok": False, "error": "Pipeline is running, cannot start standalone summarize"})
    if _summarize_running:
        return JSONResponse({"ok": False, "error": "Summarize already running"})

    _summarize_running = True
    _summarize_error = None
    summarize_only.stop_event.clear()

    def run():
        global _summarize_running, _summarize_error
        try:
            result = summarize_only.run_summarize_sync()
            if not result.get("ok"):
                _summarize_error = result.get("error", "Unknown error")
        except Exception as e:
            _summarize_error = str(e)
        finally:
            _summarize_running = False

    _summarize_thread = threading.Thread(target=run, daemon=True)
    _summarize_thread.start()
    return JSONResponse({"ok": True})


async def api_summarize_stop(request):
    summarize_only.stop_event.set()
    return JSONResponse({"ok": True})


async def api_summarize_status(request):
    snap = progress.get_snapshot()
    return JSONResponse({
        "running": _summarize_running,
        "error": _summarize_error,
        "progress": snap,
    })


async def api_logs_stream(request):
    q = asyncio.Queue()
    _log_subscribers.append(q)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = q.get_nowait()
                    yield f"data: {msg}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        finally:
            if q in _log_subscribers:
                _log_subscribers.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def api_logs(request):
    return JSONResponse({"logs": _log_buffer[-200:]})


# ===== App =====

from contextlib import asynccontextmanager


def _init_extraction_groups():
    """Инициализация _extraction_groups при запуске web-сервера с восстановлением статусов."""
    from stages import stage2_processing as s2
    
    if not pipeline.is_completed("stage_1_sort"):
        return
    
    saved = pipeline.load_stage2_groups() or {}
    sorted_data = pipeline.load_data("stage_1_sort")
    if not sorted_data:
        # Нет сохранённых данных — загружаем из Sorted директорий напрямую
        sorted_data = []
        for fmt, target_rel in cfg.FORMAT_TARGETS.items():
            d = cfg.ROOT / target_rel
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        sorted_data.append({"source": str(f), "type": fmt})
        # Также сканируем FailedExtraction и ErrorFiles — иначе формат, у которого все
        # файлы ушли в ошибки, не получит группы, и откатить его будет невозможно.
        # Определяем format_type через БД (документы там хранят свой тип).
        seen_formats: set = set()
        try:
            from database import DatabaseManager
            db = DatabaseManager()
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT format_type FROM documents
                    WHERE file_path ILIKE '%\\FailedExtraction\\%'
                       OR file_path ILIKE '%\\ErrorFiles\\%'
                """)
                for row in cur.fetchall():
                    fmt = row[0]
                    if fmt and fmt not in seen_formats:
                        seen_formats.add(fmt)
                        # Добавляем заглушку — файлы уже есть в БД, в группе достаточно
                        # записи чтобы кнопка Rollback появилась.
                        sorted_data.append({"source": None, "type": fmt})
            db.close()
        except Exception as e:
            logger.warning(f"Не удалось получить форматы из FailedExtraction/ErrorFiles: {e}")
        # Для форматов, которые есть в сохранённом состоянии, но не представлены
        # ни одним файлом в Sorted/Failed/Error — создаём пустые группы (чтобы
        # кнопка Rollback была доступна).
        for fmt, sg in saved.items():
            if fmt not in seen_formats and sg.get("extract") in ("completed", "rolled_back", "running"):
                # Создаём запись-заглушку: 0 файлов, но с сохранённым статусом
                sorted_data.append({"source": None, "type": fmt})
                seen_formats.add(fmt)
        if not sorted_data:
            return
    
    s2._prepare_extraction_groups(sorted_data)
    groups = s2._extraction_groups
    # Сбрасываем total для групп, созданных из заглушек (source=None)
    for g in groups:
        if g["total"] > 0:
            real_files = sum(1 for item in sorted_data
                            if item["type"] == g["type"] and item["source"] is not None)
            if real_files < g["total"]:
                g["total"] = real_files
    logger.info(f"Инициализировано {len(groups)} групп для этапа 2")
    
    # Восстанавливаем сохранённые под-статусы substage
    if saved:
        for g in groups:
            fmt = g["type"]
            if fmt in saved:
                sg = saved[fmt]
                # Если статус был "running" при сохранении — после перезагрузки
                # сервера/интерфейса процесс извлечения уже не работает.
                # Меняем на "stopped", чтобы пользователь мог перезапустить.
                raw_extract = sg.get("extract", "pending")
                if raw_extract == "running":
                    raw_extract = "stopped"
                g["extract_status"] = raw_extract
                raw_embed = sg.get("embed", "locked")
                if raw_embed == "running":
                    raw_embed = "stopped"
                g["embed_status"] = raw_embed
                g["extract_ok"] = sg.get("extract_stats", {}).get("ok", 0)
                g["extract_errors"] = sg.get("extract_stats", {}).get("errors", 0)
                g["extract_elapsed"] = sg.get("extract_stats", {}).get("elapsed", 0.0)
                g["embed_ok"] = sg.get("embed_stats", {}).get("ok", 0)
                g["embed_errors"] = sg.get("embed_stats", {}).get("errors", 0)
                g["embed_elapsed"] = sg.get("embed_stats", {}).get("elapsed", 0.0)
        # Форматы из сохранённого состояния, которых нет в группах (пустые заглушки),
        # уже должны быть — но на всякий случай добавляем недостающие.
        for fmt, sg in saved.items():
            if not any(g["type"] == fmt for g in groups):
                from stages.stage2_processing import _format_has_extraction
                he = _format_has_extraction(fmt)
                groups.append({
                    "type": fmt, "total": 0, "workers": 4,
                    "has_extraction": he,
                    "extract_status": sg.get("extract", "pending"),
                    "embed_status": sg.get("embed", "locked"),
                    "extract_ok": sg.get("extract_stats", {}).get("ok", 0),
                    "extract_errors": sg.get("extract_stats", {}).get("errors", 0),
                    "extract_elapsed": sg.get("extract_stats", {}).get("elapsed", 0.0),
                    "embed_ok": sg.get("embed_stats", {}).get("ok", 0),
                    "embed_errors": sg.get("embed_stats", {}).get("errors", 0),
                    "embed_elapsed": sg.get("embed_stats", {}).get("elapsed", 0.0),
                })
    
    all_done = all(
        g["embed_status"] in ("completed", "skipped")
        for g in groups
    )
    if pipeline.is_completed("stage_2_process_formats"):
        if not all_done:
            logger.warning(
                f"Этап 2 помечен как завершённый, но не у всех групп построены эмбеддинги. "
                f"Статусы: {[(g['type'], g['extract_status'], g['embed_status']) for g in groups]}"
            )
            pipeline.mark_incomplete("stage_2_process_formats")
    elif all_done and groups:
        pipeline.mark_completed("stage_2_process_formats")
        logger.info("Все эмбеддинги этапа 2 уже построены. Этап 2 отмечен как завершённый.")


@asynccontextmanager
async def lifespan(app):
    from logger_utils import suppress_proactor_connection_errors
    suppress_proactor_connection_errors()
    _init_extraction_groups()
    yield

app = Starlette(routes=[
    Route("/", api_index),
    Route("/api/status", api_status),
    Route("/api/stats", api_stats),
    Route("/api/progress", api_progress),
    Route("/api/report", api_report),
    Route("/api/config", api_config_get),
    Route("/api/config", api_config_set, methods=["POST"]),
    Route("/api/pipeline/start", api_pipeline_start, methods=["POST"]),
    Route("/api/pipeline/stop", api_pipeline_stop, methods=["POST"]),
    Route("/api/pipeline/unlock", api_pipeline_unlock, methods=["POST"]),
    Route("/api/pipeline/confirm", api_pipeline_confirm, methods=["POST"]),
    Route("/api/pipeline/rollback", api_pipeline_rollback, methods=["POST"]),
    Route("/api/pipeline/reset", api_pipeline_reset, methods=["POST"]),
    Route("/api/pipeline/reset/status", api_pipeline_reset_status, methods=["GET"]),
    Route("/api/substage/start", api_substage_start, methods=["POST"]),
    Route("/api/substage/rollback", api_substage_rollback, methods=["POST"]),
    Route("/api/substage/stop", api_substage_stop, methods=["POST"]),
    Route("/api/summarize/start", api_summarize_start, methods=["POST"]),
    Route("/api/summarize/stop", api_summarize_stop, methods=["POST"]),
    Route("/api/summarize/status", api_summarize_status),
    Route("/api/logs/stream", api_logs_stream),
    Route("/api/logs", api_logs),
], lifespan=lifespan)

# Логирование всех HTTP-запросов
class _RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.info(f"HTTP {request.method} {request.url.path}")
        return await call_next(request)

app.add_middleware(_RequestLogMiddleware)


def run_server(host: str = "0.0.0.0", port: int = 8080):
    log_capture.install()
    print(f"\n  Web UI: http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
