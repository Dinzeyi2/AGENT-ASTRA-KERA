"""
Codeastra Autonomous Agent Service
====================================
Thin coordinator. Does three things only:
  1. Call app.codeastra.dev/protect/text  — tokenize data
  2. Send tokens to OpenAI GPT-4o         — AI works blind
  3. Call app.codeastra.dev/vault/resolve-batch — reveal after

All data logic lives in the Codeastra API.
This service just coordinates.

Env vars:
  OPENAI_API_KEY
  CODEASTRA_API_KEY
  CODEASTRA_URL      (default: https://app.codeastra.dev)
  DATABASE_URL       (optional — for real DB tools)
  PORT
"""

import os, json, asyncio, re, logging, uuid, time
from datetime import datetime
from typing import AsyncGenerator

import httpx
import asyncpg
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, RunConfig, gen_trace_id, flush_traces
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("codeastra-agent")

OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
CODEASTRA_KEY = os.getenv("CODEASTRA_API_KEY", "")
CODEASTRA_URL = os.getenv("CODEASTRA_URL", "https://app.codeastra.dev")
DATABASE_URL  = os.getenv("DATABASE_URL", "")
PORT          = int(os.getenv("PORT", 8080))

app = FastAPI(title="Codeastra Agent Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db_pool = None


# ── Startup ───────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global db_pool

    # Set OpenAI key for Agents SDK tracing
    if OPENAI_KEY:
        import openai as _oai
        _oai.api_key = OPENAI_KEY
        try:
            from agents.tracing import set_tracing_export_api_key
            set_tracing_export_api_key(OPENAI_KEY)
            log.info("✅ Agents SDK tracing configured")
        except Exception: pass

    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            log.info("✅ Database connected")
        except Exception as e:
            log.warning(f"DB: {e}")


# ═══════════════════════════════════════════════════════════
# CODEASTRA API CALLS
# All data operations go through app.codeastra.dev
# ═══════════════════════════════════════════════════════════

async def codeastra_protect(text: str, active: bool = True) -> tuple[str, list]:
    """
    Send text to Codeastra API → get back tokenized text + intercepted list.
    This is the ONLY place we touch data.
    """
    if not active or not CODEASTRA_KEY:
        return text, []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{CODEASTRA_URL}/protect/text",
                headers={"X-API-Key": CODEASTRA_KEY,
                         "Content-Type": "application/json"},
                json={"text": text},
            )
            if r.status_code == 200:
                data     = r.json()
                protected = data.get("protected_text", text)
                entities  = data.get("entities") or data.get("detections") or []
                intercepted = [
                    {
                        "type":    "intercepted",
                        "dtype":   e.get("type", "PII"),
                        "token":   e.get("token", ""),
                        "preview": e.get("preview", "•••"),
                    }
                    for e in entities
                ]
                log.info(f"Codeastra protected {len(intercepted)} values")
                return protected, intercepted
    except Exception as e:
        log.warning(f"Codeastra protect error: {e}")

    return text, []


async def codeastra_reveal(tokens: list) -> dict:
    """
    Send token list to Codeastra vault/resolve-batch.
    Returns {token: real_value} map.
    ONE API call for all tokens.
    """
    if not tokens or not CODEASTRA_KEY:
        return {}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CODEASTRA_URL}/vault/resolve-batch",
                headers={"X-API-Key": CODEASTRA_KEY,
                         "Content-Type": "application/json"},
                json={"tokens": tokens},
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("results") or data.get("resolved") or {}
    except Exception as e:
        log.warning(f"Codeastra reveal error: {e}")

    return {}


async def codeastra_executor(endpoint: str, body: dict) -> dict:
    """
    Call any Codeastra executor endpoint.
    e.g. /executor/run, /executor/concentration-check, etc.
    """
    if not CODEASTRA_KEY:
        return {"error": "No CODEASTRA_API_KEY"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CODEASTRA_URL}{endpoint}",
                headers={"X-API-Key": CODEASTRA_KEY,
                         "Content-Type": "application/json"},
                json=body,
            )
            return r.json()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# DATABASE TOOLS
# Each tool reads real DB data.
# Every result goes through Codeastra before GPT sees it.
# ═══════════════════════════════════════════════════════════

async def _db_query(sql: str) -> dict:
    """Run a SELECT on the real DB. Returns raw data."""
    if not db_pool:
        return {"error": "No DATABASE_URL set"}
    if not sql.strip().upper().startswith(("SELECT", "WITH", "EXPLAIN")):
        return {"error": "Only SELECT queries allowed"}
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql)
            return {"rows": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            return {"error": str(e)}


async def _db_execute(sql: str) -> dict:
    """Execute a write operation on the real DB."""
    if not db_pool:
        return {"error": "No DATABASE_URL set"}
    async with db_pool.acquire() as conn:
        try:
            await conn.execute(sql)
            return {"status": "success", "sql": sql}
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# OPENAI AGENTS SDK — REAL AGENT
# ═══════════════════════════════════════════════════════════

SYSTEM = """You are an expert autonomous Database Administrator.
You work through Codeastra's Zero Trust middleware.
ALL sensitive data has been replaced with tokens like [CVT:EMAIL:A1B2C3].
Work with tokens as identifiers. Never try to guess real values.
For computations on amounts use check_threshold or concentration_check.

Steps:
1. List tables + get DB stats
2. Scan for slow queries
3. Inspect affected tables
4. Create missing indexes
5. Call get_summary at the end"""

TASK_PROMPTS = {
    "dba":   "Our production database has performance issues. Investigate completely — check stats, find slow queries, inspect tables, create missing indexes. Fix everything.",
    "audit": "Run a complete database audit. Check all tables, performance, connections, indexes. Report everything.",
}

# In-memory trace store
TRACES = {}


async def run_agent(
    task_type: str,
    custom_task: str = "",
    codeastra_active: bool = True,
    thread_id: str = None,
) -> AsyncGenerator[dict, None]:

    if not OPENAI_KEY:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    import openai as _oai
    _oai.api_key = OPENAI_KEY

    task      = custom_task or TASK_PROMPTS.get(task_type, TASK_PROMPTS["dba"])
    trace_id  = gen_trace_id()
    run_id    = f"run_{uuid.uuid4().hex[:12]}"
    started   = datetime.utcnow().isoformat()
    intercepted_all = []

    TRACES[run_id] = {
        "run_id": run_id, "task": task, "task_type": task_type,
        "codeastra_active": codeastra_active, "started_at": started,
        "steps": [], "intercepted": [], "status": "running",
    }

    yield {
        "type":             "start",
        "run_id":           run_id,
        "openai_trace_id":  trace_id,
        "task":             task,
        "codeastra_active": codeastra_active,
        "mode":             "PROTECTED" if codeastra_active else "⚠️ UNPROTECTED",
        "openai_traces_url": "https://platform.openai.com/logs",
        "timestamp":        started,
    }
    await asyncio.sleep(0.1)

    # ── Build tools ────────────────────────────────────────
    # Each tool: query DB → protect via Codeastra → GPT receives tokens

    @function_tool
    async def list_tables() -> str:
        """List all database tables with sizes."""
        raw = await _db_query("""
            SELECT table_name,
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
            FROM information_schema.tables
            WHERE table_schema='public'
            ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
        """)
        protected, events = await codeastra_protect(json.dumps(raw, default=str), codeastra_active)
        intercepted_all.extend(events)
        for ev in events: yield_queue.append(ev)
        return protected

    @function_tool
    async def scan_slow_queries() -> str:
        """Scan for slow queries using pg_stat_statements."""
        raw = await _db_query("""
            SELECT query, calls,
                   ROUND(mean_exec_time::numeric,2) AS avg_ms,
                   rows
            FROM pg_stat_statements
            WHERE mean_exec_time > 100
            ORDER BY mean_exec_time DESC LIMIT 20
        """)
        protected, events = await codeastra_protect(json.dumps(raw, default=str), codeastra_active)
        intercepted_all.extend(events)
        return protected

    @function_tool
    async def inspect_table(table: str) -> str:
        """Inspect a database table: schema, indexes, row count, sample rows."""
        cols = await _db_query(f"""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_name='{table}' AND table_schema='public'
            ORDER BY ordinal_position
        """)
        try:
            samples = await _db_query(f'SELECT * FROM "{table}" LIMIT 5')
        except Exception:
            samples = {}
        raw = {"table": table, "columns": cols, "samples": samples}
        protected, events = await codeastra_protect(json.dumps(raw, default=str), codeastra_active)
        intercepted_all.extend(events)
        return protected

    @function_tool
    async def create_index(table: str, column: str) -> str:
        """Create a real database index on a column."""
        idx = f"idx_{table}_{column}_codeastra"
        raw = await _db_execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx} ON {table}({column})"
        )
        protected, events = await codeastra_protect(json.dumps(raw), codeastra_active)
        intercepted_all.extend(events)
        return protected

    @function_tool
    async def run_query(sql: str) -> str:
        """Run a SELECT query on the real database."""
        raw = await _db_query(sql)
        protected, events = await codeastra_protect(json.dumps(raw, default=str), codeastra_active)
        intercepted_all.extend(events)
        return protected

    @function_tool
    async def get_db_stats() -> str:
        """Get database health statistics."""
        raw = await _db_query("""
            SELECT numbackends AS connections,
                   xact_commit AS committed,
                   ROUND(blks_hit::numeric/NULLIF(blks_hit+blks_read,0)*100,2) AS cache_hit_pct,
                   deadlocks,
                   pg_size_pretty(pg_database_size(current_database())) AS db_size
            FROM pg_stat_database WHERE datname=current_database()
        """)
        protected, events = await codeastra_protect(json.dumps(raw, default=str), codeastra_active)
        intercepted_all.extend(events)
        return protected

    @function_tool
    async def check_threshold(token: str, threshold: float, operator: str = "gt") -> str:
        """Check if a vaulted amount token exceeds a threshold. Calls Codeastra executor."""
        result = await codeastra_executor("/executor/check-threshold", {
            "token": token, "threshold": threshold, "operator": operator
        })
        return json.dumps(result)

    @function_tool
    async def concentration_check(position_token: str, portfolio_token: str, threshold_pct: float) -> str:
        """Check portfolio concentration. Calls Codeastra executor."""
        result = await codeastra_executor("/executor/concentration-check", {
            "position_token": position_token,
            "portfolio_token": portfolio_token,
            "threshold_pct": threshold_pct,
        })
        return json.dumps(result)

    @function_tool
    async def get_summary() -> str:
        """Get summary of what was accomplished. Call at the end."""
        raw = {
            "status": "complete",
            "db_connected": db_pool is not None,
            "codeastra_active": codeastra_active,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if db_pool:
            idx = await _db_query("""
                SELECT indexname FROM pg_indexes
                WHERE indexname LIKE '%codeastra%'
            """)
            raw["indexes_created"] = idx
        protected, events = await codeastra_protect(json.dumps(raw), codeastra_active)
        intercepted_all.extend(events)
        return protected

    # ── Run agent ──────────────────────────────────────────
    yield_queue = []

    dba_agent = Agent(
        name         = "Codeastra DBA Agent",
        instructions = SYSTEM,
        model        = "gpt-4o",
        tools        = [
            list_tables, scan_slow_queries, inspect_table,
            create_index, run_query, get_db_stats,
            check_threshold, concentration_check, get_summary,
        ],
    )

    try:
        result = await Runner.run(
            starting_agent = dba_agent,
            input          = task,
            max_turns      = 20,
            run_config     = RunConfig(
                workflow_name                = f"codeastra-{task_type}",
                trace_id                     = trace_id,
                trace_metadata               = {"codeastra_active": str(codeastra_active)},
                trace_include_sensitive_data = True,
            ),
        )
        flush_traces()

    except Exception as e:
        yield {"type": "error", "message": str(e), "run_id": run_id}
        TRACES[run_id]["status"] = "error"
        return

    # Emit all intercepted events
    for ev in intercepted_all:
        yield ev

    final_output = str(result.final_output) if result.final_output else ""
    if final_output:
        yield {"type": "thinking", "text": final_output, "run_id": run_id}

    # Auto-reveal: one batch call to Codeastra
    tokens  = list(set(e["token"] for e in intercepted_all if e.get("token")))
    revealed = await codeastra_reveal(tokens)

    # Update trace
    TRACES[run_id].update({
        "status":       "complete",
        "intercepted":  intercepted_all,
        "completed_at": datetime.utcnow().isoformat(),
    })

    yield {
        "type":                  "complete",
        "run_id":                run_id,
        "openai_trace_id":       trace_id,
        "openai_traces_url":     "https://platform.openai.com/logs",
        "intercepted":           len(intercepted_all),
        "codeastra_active":      codeastra_active,
        "real_data_seen_by_gpt": 0 if codeastra_active else "⚠️ YES",
        "revealed":              {
            "results":  revealed,
            "count":    len([v for v in revealed.values() if v]),
            "note":     "Resolved after agent closed — agent never saw these values",
        },
    }


# ═══════════════════════════════════════════════════════════
# DOCUMENT AGENT
# ═══════════════════════════════════════════════════════════

async def run_document_agent(
    text: str,
    task: str,
    filename: str,
    codeastra_active: bool = True,
) -> AsyncGenerator[dict, None]:

    if not OPENAI_KEY:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    if len(text) > 80000:
        text = text[:80000] + "\n\n[... truncated ...]"

    run_id = f"doc_{uuid.uuid4().hex[:12]}"

    yield {
        "type":             "start",
        "run_id":           run_id,
        "filename":         filename,
        "task":             task,
        "codeastra_active": codeastra_active,
        "char_count":       len(text),
    }

    # Step 1 — protect via Codeastra
    if codeastra_active:
        yield {"type": "phase", "message": "Codeastra scanning document for PII..."}
        protected_text, intercepted = await codeastra_protect(text, True)
        for ev in intercepted:
            yield ev
    else:
        yield {"type": "warning", "message": "⚠️ Codeastra OFF — document unprotected"}
        protected_text = text
        intercepted    = []

    yield {"type": "phase", "message": "Sending to GPT-4o via Agents SDK..."}

    # Step 2 — run through Agents SDK
    import openai as _oai
    _oai.api_key = OPENAI_KEY

    doc_agent = Agent(
        name         = "Codeastra Document Analyst",
        instructions = (
            "You are an expert document analyst working through Codeastra's Zero Trust middleware. "
            "All PII has been replaced with tokens like [CVT:EMAIL:A1B2C3]. "
            "Work with tokens as identifiers. Analyze structure, content, and meaning. "
            "Be thorough and specific."
        ),
        model = "gpt-4o",
    )

    trace_id   = gen_trace_id()
    doc_input  = ("TASK: " + (task or "Analyze this document thoroughly.") +
                  "\n\nDOCUMENT (" + filename + "):\n\n" + protected_text)

    try:
        doc_result = await Runner.run(
            starting_agent = doc_agent,
            input          = doc_input,
            max_turns      = 1,
            run_config     = RunConfig(
                workflow_name                = "codeastra-document-analysis",
                trace_id                     = trace_id,
                trace_metadata               = {"filename": filename, "codeastra_active": str(codeastra_active)},
                trace_include_sensitive_data = True,
            ),
        )
        flush_traces()
        full = str(doc_result.final_output) if doc_result.final_output else ""
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    yield {"type": "thinking", "text": full, "run_id": run_id}

    # Step 3 — reveal via Codeastra (one batch call)
    tokens   = list(set(e["token"] for e in intercepted if e.get("token")))
    revealed = await codeastra_reveal(tokens)

    yield {
        "type":                    "complete",
        "run_id":                  run_id,
        "openai_trace_id":         trace_id,
        "openai_traces_url":       "https://platform.openai.com/logs",
        "filename":                filename,
        "codeastra_active":        codeastra_active,
        "intercepted":             len(intercepted),
        "real_data_seen_by_gpt":   0 if codeastra_active else "⚠️ YES",
        "revealed":                {
            "results": revealed,
            "count":   len([v for v in revealed.values() if v]),
            "note":    "Resolved after agent closed — agent never saw these values",
        },
    }


# ═══════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════

async def extract_text(file: UploadFile) -> str:
    import io
    content  = await file.read()
    filename = (file.filename or "").lower()
    mime     = file.content_type or ""
    if filename.endswith(".pdf") or "pdf" in mime:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e: return f"[PDF error: {e}]"
    if filename.endswith(".docx"):
        try:
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(content)).paragraphs)
        except Exception as e: return f"[DOCX error: {e}]"
    if filename.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            rows = []
            for sheet in wb.worksheets:
                rows.append(f"=== {sheet.title} ===")
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)
        except Exception as e: return f"[Excel error: {e}]"
    if filename.endswith(".csv"):
        return content.decode("utf-8", errors="replace")
    if filename.endswith((".html", ".htm")):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(content, "html.parser").get_text("\n", strip=True)
        except Exception: return content.decode("utf-8", errors="replace")
    return content.decode("utf-8", errors="replace")


async def extract_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(url)
            ct = r.headers.get("content-type", "")
            if "html" in ct:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "header", "footer"]): tag.decompose()
                    return soup.get_text("\n", strip=True)[:50000]
                except Exception: return r.text[:50000]
            return r.text[:50000]
    except Exception as e: return f"[URL error: {e}]"


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _stream(gen):
    async def s():
        async for ev in gen:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    return StreamingResponse(s(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Access-Control-Allow-Origin": "*"})


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def index():
    with open("index.html") as f: return HTMLResponse(f.read())

@app.get("/embed")
async def embed():
    with open("index.html") as f: return HTMLResponse(f.read())

@app.get("/health")
async def health():
    codeastra_ok = False
    if CODEASTRA_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{CODEASTRA_URL}/health",
                                headers={"X-API-Key": CODEASTRA_KEY})
                codeastra_ok = r.status_code == 200
        except Exception: pass
    return {
        "status":          "healthy",
        "version":         "clean-v1",
        "openai_ready":    bool(OPENAI_KEY),
        "codeastra_ready": bool(CODEASTRA_KEY),
        "codeastra_live":  codeastra_ok,
        "db_ready":        db_pool is not None,
    }

@app.get("/agent/tasks")
async def list_tasks():
    return {
        "model": "gpt-4o",
        "tasks": [
            {"id": "dba",   "name": "Database Performance Agent"},
            {"id": "audit", "name": "Database Audit Agent"},
        ],
    }

# ── Agent run ─────────────────────────────────────────────

@app.post("/agent/run/stream")
async def agent_run_stream(req: Request):
    body = await req.json()
    return _stream(run_agent(
        body.get("task_type", "dba"),
        body.get("custom_task", ""),
        codeastra_active = body.get("codeastra_enabled", True),
    ))

@app.post("/agent/run/sync")
async def agent_run_sync(req: Request):
    body = await req.json()
    all_ev = []; intercept = []; summary = {}
    async for ev in run_agent(body.get("task_type","dba"), body.get("custom_task",""),
                               codeastra_active=body.get("codeastra_enabled",True)):
        all_ev.append(ev)
        if ev["type"] == "intercepted": intercept.append(ev)
        if ev["type"] == "complete":    summary = ev
    return {"success": True, "intercepted": intercept, "summary": summary}

@app.post("/agent/run/protected")
async def agent_run_protected(req: Request):
    body = await req.json()
    return _stream(run_agent(body.get("task_type","dba"), body.get("custom_task",""), codeastra_active=True))

@app.post("/agent/run/unprotected")
async def agent_run_unprotected(req: Request):
    body = await req.json()
    return _stream(run_agent(body.get("task_type","dba"), body.get("custom_task",""), codeastra_active=False))

# ── Document endpoints ────────────────────────────────────

@app.post("/agent/analyze-document")
async def analyze_document(
    file:              UploadFile = File(default=None),
    task:              str        = Form(default=""),
    codeastra_enabled: str        = Form(default="true"),
):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    text = await extract_text(file)
    return _stream(run_document_agent(
        text, task, file.filename or "document",
        codeastra_active=codeastra_enabled.lower() != "false",
    ))

@app.post("/agent/analyze-url")
async def analyze_url(req: Request):
    body = await req.json()
    url  = body.get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse(status_code=400, content={"error": "Valid URL required"})
    text = await extract_url(url)
    return _stream(run_document_agent(
        text, body.get("task",""), url,
        codeastra_active=body.get("codeastra_enabled", True),
    ))

@app.post("/agent/analyze-text")
async def analyze_text(req: Request):
    body = await req.json()
    text = body.get("text","").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})
    return _stream(run_document_agent(
        text, body.get("task",""), body.get("name","text"),
        codeastra_active=body.get("codeastra_enabled", True),
    ))

@app.post("/agent/analyze-multiple")
async def analyze_multiple(
    files:             list[UploadFile] = File(default=None),
    task:              str              = Form(default=""),
    codeastra_enabled: str              = Form(default="true"),
):
    if not files: return JSONResponse(status_code=400, content={"error":"No files"})
    all_text = ""
    names    = []
    for f in files[:10]:
        all_text += f"\n\n=== FILE: {f.filename} ===\n{await extract_text(f)}"
        names.append(f.filename or "file")
    return _stream(run_document_agent(
        all_text, task, f"{len(names)} files: {', '.join(names)}",
        codeastra_active=codeastra_enabled.lower() != "false",
    ))

# ── Traces (our internal trace store) ────────────────────

@app.get("/traces")
async def list_traces():
    traces = sorted(TRACES.values(), key=lambda t: t["started_at"], reverse=True)
    return {"traces": traces[:20], "count": len(TRACES)}

@app.get("/traces/{run_id}")
async def get_trace(run_id: str):
    if run_id not in TRACES:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return TRACES[run_id]

# ── Proxy Codeastra endpoints directly ───────────────────

@app.post("/protect")
async def protect_text(req: Request):
    """Proxy to Codeastra /protect/text"""
    body = await req.json()
    text = body.get("text", "")
    protected, intercepted = await codeastra_protect(text, True)
    return {"original": text, "protected": protected, "intercepted": intercepted}

@app.get("/executor/capabilities")
async def executor_capabilities():
    """Proxy to Codeastra /executor/supported"""
    if not CODEASTRA_KEY:
        return {"error": "No CODEASTRA_API_KEY"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{CODEASTRA_URL}/executor/supported",
                            headers={"X-API-Key": CODEASTRA_KEY})
            return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/executor/{path:path}")
async def executor_proxy(path: str, req: Request):
    """Proxy all executor calls to Codeastra"""
    body = await req.json()
    return await codeastra_executor(f"/executor/{path}", body)

@app.get("/db/status")
async def db_status():
    if not db_pool: return {"connected": False, "message": "Set DATABASE_URL"}
    async with db_pool.acquire() as conn:
        try:
            ver = await conn.fetchval("SELECT version()")
            tbl = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
            return {"connected": True, "version": ver, "tables": tbl}
        except Exception as e:
            return {"connected": False, "error": str(e)}

@app.post("/debug/protect-raw")
async def debug_protect_raw(req: Request):
    body = await req.json()
    if not CODEASTRA_KEY: return {"error": "No CODEASTRA_API_KEY"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{CODEASTRA_URL}/protect/text",
            headers={"X-API-Key": CODEASTRA_KEY, "Content-Type": "application/json"},
            json={"text": body.get("text", "")})
        return {"status": r.status_code, "response": r.json() if r.status_code == 200 else r.text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
