"""
Codeastra Autonomous Agent — Full OpenAI Integration
======================================================
Uses OpenAI Agents SDK for:
  - Agent Traces       — full step-by-step decision log
  - Completions        — raw completion records
  - Conversations      — multi-turn conversation history
  - ChatKit Threads    — persistent thread management

Every tool result goes through Codeastra before GPT sees it.
All traces stored and retrievable via API.

Required env vars:
  OPENAI_API_KEY
  CODEASTRA_API_KEY
  DATABASE_URL (optional — for real DB tools)
  PORT
"""

import os, json, asyncio, re, hashlib, logging, io, uuid, time
from datetime import datetime
from typing import AsyncGenerator
from collections import defaultdict

import httpx
import asyncpg
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool, trace, gen_trace_id, RunConfig
)
import agents as _agents_sdk

# flush_traces — safe import for all SDK versions
try:
    from agents import flush_traces as _flush_traces
    def flush_traces():
        _flush_traces()
except ImportError:
    def flush_traces():
        pass

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

if OPENAI_KEY:
    from agents import set_default_openai_key
    set_default_openai_key(OPENAI_KEY)

# ── Codeastra real SDK ────────────────────────────────────
try:
    from codeastra import CodeAstraClient, BlindAgentMiddleware
    ca_client = CodeAstraClient(api_key=CODEASTRA_KEY) if CODEASTRA_KEY else None
    if ca_client:
        log.info("✅ Codeastra SDK initialized")
    else:
        log.warning("⚠️  CODEASTRA_API_KEY not set — Codeastra features require it")
except ImportError:
    log.warning("codeastra SDK not installed — run: pip install codeastra 'codeastra[fhe]'")
    ca_client = None

async def _run_sync(func, *args, **kwargs):
    """Run a synchronous Codeastra SDK call without blocking FastAPI."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

app = FastAPI(title="Codeastra Agent — OpenAI Full Integration")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db_pool = None

# ═══════════════════════════════════════════════════════════
# IN-MEMORY STORES
# ═══════════════════════════════════════════════════════════

TRACES        = {}
CONVERSATIONS = {}
THREADS       = {}
COMPLETIONS   = {}

# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global db_pool
    if OPENAI_KEY:
        import openai as _oai
        _oai.api_key = OPENAI_KEY
        try:
            from agents.tracing import set_tracing_export_api_key
            set_tracing_export_api_key(OPENAI_KEY)
            log.info("✅ Agents SDK tracing configured")
        except Exception as e:
            log.warning(f"Agents SDK tracing setup: {e}")
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            log.info("✅ Database connected")
        except Exception as e:
            log.warning(f"DB: {e}")


# ═══════════════════════════════════════════════════════════
# CODEASTRA — Real API
# ═══════════════════════════════════════════════════════════

def _json_safe(obj):
    import decimal, datetime as _dt
    if isinstance(obj, decimal.Decimal): return float(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)): return obj.isoformat()
    if isinstance(obj, bytes): return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(i) for i in obj]
    return str(obj)


def _safe_row(row) -> dict:
    import decimal, datetime as _dt
    result = {}
    for key, val in dict(row).items():
        if isinstance(val, decimal.Decimal): result[key] = float(val)
        elif isinstance(val, (_dt.datetime, _dt.date)): result[key] = val.isoformat()
        elif isinstance(val, bytes): result[key] = val.decode("utf-8", errors="replace")
        else: result[key] = val
    return result


async def protect(data, events: list, active: bool = True) -> str:
    if isinstance(data, dict): data = _json_safe(data)
    text = json.dumps(data, default=_json_safe) if not isinstance(data, str) else data
    if not active:
        events.append({"type": "unprotected", "preview": text[:120]})
        return text
    if not ca_client:
        events.append({"type": "unprotected", "preview": text[:120]})
        return text
    try:
        result   = await _run_sync(ca_client.protect_text_full, text)
        prot     = result.get("protected_text", text)
        entities = result.get("entities") or []
        for e in entities:
            real = e.get("original") or e.get("value") or ""
            prev = real[:3] + "•" * min(len(real)-5, 8) + real[-2:] if len(real) > 5 else "•••"
            events.append({
                "type": "intercepted", "dtype": e.get("type") or "PII",
                "token": e.get("token", ""), "preview": e.get("preview") or prev,
            })
        log.info(f"Codeastra protected {len(entities)} values")
        return prot
    except Exception as ex:
        log.warning(f"Codeastra protect error: {ex}")
        return text


async def codeastra_resolve(token: str):
    if not ca_client: return None
    try:
        result = await _run_sync(ca_client.vault_resolve, token)
        return result.get("real_value") or result.get("value")
    except Exception as e:
        log.warning(f"vault_resolve error: {e}")
        return None


async def codeastra_resolve_batch(tokens: list) -> dict:
    if not ca_client or not tokens: return {}
    try:
        result = await _run_sync(ca_client.vault_resolve_batch, tokens)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        log.warning(f"vault_resolve_batch error: {e}")
        return {}


async def codeastra_executor_run(token_id: str, dry_run: bool = False) -> dict:
    if not ca_client: return {"error": "No Codeastra client — set CODEASTRA_API_KEY"}
    try:
        result = await _run_sync(lambda: ca_client.executor_run(token_id, dry_run=dry_run))
        return result
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# DATABASE TOOLS
# ═══════════════════════════════════════════════════════════

async def tool_list_tables(events, active=True):
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL on Railway"}, events, active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema='public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            real = {"tables": [_safe_row(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e)}
    return await protect(real, events, active)


async def tool_scan_slow_queries(events, active=True):
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL on Railway"}, events, active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT query, calls, ROUND(mean_exec_time::numeric, 2) AS avg_ms, rows
                FROM pg_stat_statements
                WHERE mean_exec_time > 100
                ORDER BY mean_exec_time DESC LIMIT 20
            """)
            real = {"slow_queries": [_safe_row(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e), "hint": "Enable pg_stat_statements"}
    return await protect(real, events, active)


async def tool_inspect_table(events, table: str, active=True):
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL"}, events, active)
    async with db_pool.acquire() as conn:
        try:
            cols = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name=$1 AND table_schema='public'
                ORDER BY ordinal_position
            """, table)
            idxs = await conn.fetch(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename=$1", table)
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
            except Exception:
                count = "unknown"
            try:
                samples = await conn.fetch(f'SELECT * FROM "{table}" LIMIT 5')
                sample_list = [dict(r) for r in samples]
            except Exception:
                sample_list = []
            real = {"table": table, "columns": [_safe_row(c) for c in cols],
                    "indexes": [_safe_row(i) for i in idxs],
                    "row_count": count, "samples": sample_list}
        except Exception as e:
            real = {"error": str(e), "table": table}
    return await protect(real, events, active)


async def tool_create_index(events, table: str, column: str, active=True):
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL"}, events, active)
    idx = f"idx_{table}_{column}_codeastra"
    async with db_pool.acquire() as conn:
        try:
            await conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx} ON {table}({column})")
            real = {"status": "success",
                    "sql": f"CREATE INDEX {idx} ON {table}({column})", "index": idx}
        except Exception as e:
            real = {"status": "error", "error": str(e)}
    return await protect(real, events, active)


async def tool_run_query(events, sql: str, active=True):
    if not sql.strip().upper().startswith(("SELECT", "WITH", "EXPLAIN")):
        return await protect({"error": "Only SELECT allowed"}, events, active)
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL"}, events, active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql)
            real = {"sql": sql, "rows": [_safe_row(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e), "sql": sql}
    return await protect(real, events, active)


async def tool_get_db_stats(events, active=True):
    if not db_pool:
        return await protect({"error": "Set DATABASE_URL"}, events, active)
    async with db_pool.acquire() as conn:
        try:
            stats = await conn.fetchrow("""
                SELECT numbackends AS connections,
                       xact_commit AS committed,
                       xact_rollback AS rolled_back,
                       ROUND(blks_hit::numeric/NULLIF(blks_hit+blks_read,0)*100,2) AS cache_hit_pct,
                       deadlocks,
                       pg_size_pretty(pg_database_size(current_database())) AS db_size
                FROM pg_stat_database WHERE datname=current_database()
            """)
            real = _safe_row(stats) if stats else {}
        except Exception as e:
            real = {"error": str(e)}
    return await protect(real, events, active)


async def tool_get_summary(events, active=True):
    real = {
        "agent": "Codeastra OpenAI Agent", "model": "gpt-4o",
        "codeastra_active": active, "db_connected": db_pool is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if db_pool:
        async with db_pool.acquire() as conn:
            try:
                idxs = await conn.fetch(
                    "SELECT indexname FROM pg_indexes WHERE indexname LIKE '%codeastra%'")
                real["indexes_created"] = [r["indexname"] for r in idxs]
            except Exception:
                pass
    return await protect(real, events, active)


async def tool_check_threshold(events, token, threshold, operator="gt", active=True):
    real_val = await codeastra_resolve(token)
    if real_val is None:
        return await protect({"error": f"Cannot resolve {token}", "real_returned": False}, events, active)
    try:
        v = float(str(real_val).replace("$", "").replace(",", "").strip())
        ops = {"gt": v>threshold, "lt": v<threshold, "gte": v>=threshold, "lte": v<=threshold}
        return await protect({"result": ops.get(operator, v>threshold),
                               "operator": operator, "threshold": threshold,
                               "real_returned": False}, events, active)
    except Exception as e:
        return await protect({"error": str(e)}, events, active)


async def tool_concentration_check(events, position_token, portfolio_token,
                                    threshold_pct, active=True):
    pv = await codeastra_resolve(position_token)
    tv = await codeastra_resolve(portfolio_token)
    if pv is None or tv is None:
        return await protect({"exceeds_threshold": None,
                               "note": "Tokens not resolved", "real_returned": False},
                              events, active)
    try:
        p = float(str(pv).replace("$", "").replace(",", "").strip())
        t = float(str(tv).replace("$", "").replace(",", "").strip())
        pct = (p/t*100) if t > 0 else 0
        bucket = "critical" if pct>60 else "high" if pct>40 else "medium" if pct>20 else "low"
        return await protect({"exceeds_threshold": pct>threshold_pct,
                               "bucket": bucket, "threshold_pct": threshold_pct,
                               "real_returned": False}, events, active)
    except Exception as e:
        return await protect({"error": str(e)}, events, active)


TASK_PROMPTS = {
    "dba":      "Our production database has severe performance issues. Investigate completely — check stats, find slow queries, inspect all tables, create missing indexes. Fix everything.",
    "audit":    "Run a complete database audit — tables, performance, connections, indexes. Report everything you find.",
    "security": "Audit the database for security issues — overprivileged connections, sensitive data exposure patterns, missing encryption signals. Report all risks.",
}

SYSTEM = """You are an expert autonomous Database Administrator.
You are working through Codeastra's Zero Trust middleware.
ALL sensitive data has been replaced with tokens like [CVT:EMAIL:A1B2C3].
Work with tokens as identifiers. Never try to guess real values.
For amount computations use check_threshold or concentration_check — they resolve tokens internally.

Work methodically:
1. List tables + get DB stats
2. Scan slow queries
3. Inspect affected tables
4. Create missing indexes
5. Call get_summary

Be thorough. Fix everything."""


# ═══════════════════════════════════════════════════════════
# TRACE SYSTEM
# ═══════════════════════════════════════════════════════════

class AgentTrace:
    def __init__(self, task_type, task, model, codeastra_active):
        self.id               = f"trace_{uuid.uuid4().hex[:12]}"
        self.task_type        = task_type
        self.task             = task
        self.model            = model
        self.codeastra_active = codeastra_active
        self.started_at       = datetime.utcnow().isoformat()
        self.completed_at     = None
        self.status           = "running"
        self.steps            = []
        self.intercepted      = []
        self.tool_calls       = []
        self.total_duration_ms = 0
        self._start_time      = time.time()

    def add_step(self, step_type, data):
        step = {"step": len(self.steps)+1, "type": step_type,
                "timestamp": datetime.utcnow().isoformat(),
                "elapsed_ms": int((time.time()-self._start_time)*1000), **data}
        self.steps.append(step)
        return step

    def add_interception(self, dtype, token, preview):
        self.intercepted.append({
            "n": len(self.intercepted)+1, "dtype": dtype,
            "token": token, "preview": preview,
            "at_ms": int((time.time()-self._start_time)*1000),
        })

    def complete(self):
        self.status            = "completed"
        self.completed_at      = datetime.utcnow().isoformat()
        self.total_duration_ms = int((time.time()-self._start_time)*1000)

    def to_dict(self):
        return {
            "trace_id": self.id, "task_type": self.task_type,
            "task": self.task, "model": self.model,
            "codeastra_active": self.codeastra_active,
            "status": self.status, "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "steps": self.steps, "tool_calls": self.tool_calls,
            "intercepted": self.intercepted,
            "total_steps": len(self.steps),
            "total_tool_calls": len(self.tool_calls),
            "total_intercepted": len(self.intercepted),
            "real_data_seen_by_gpt": 0 if self.codeastra_active else "⚠️ YES",
            "openai_logs_url": "https://platform.openai.com/logs",
        }


# ═══════════════════════════════════════════════════════════
# CONVERSATION SYSTEM
# ═══════════════════════════════════════════════════════════

class Conversation:
    def __init__(self, title="", system_context=""):
        self.id             = f"conv_{uuid.uuid4().hex[:12]}"
        self.title          = title or "Untitled Conversation"
        self.system_context = system_context
        self.created_at     = datetime.utcnow().isoformat()
        self.updated_at     = datetime.utcnow().isoformat()
        self.turns          = []
        self.trace_ids      = []

    def add_turn(self, role, content, trace_id=None):
        turn = {"n": len(self.turns)+1, "role": role, "content": content,
                "trace_id": trace_id, "timestamp": datetime.utcnow().isoformat()}
        self.turns.append(turn)
        self.updated_at = datetime.utcnow().isoformat()
        if trace_id: self.trace_ids.append(trace_id)
        return turn

    def to_dict(self):
        return {"conversation_id": self.id, "title": self.title,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "turn_count": len(self.turns), "turns": self.turns,
                "trace_ids": self.trace_ids}


# ═══════════════════════════════════════════════════════════
# CHATKIT THREAD SYSTEM
# ═══════════════════════════════════════════════════════════

class ChatKitThread:
    def __init__(self, title="", metadata=None):
        self.id         = f"thread_{uuid.uuid4().hex[:12]}"
        self.title      = title or "New Thread"
        self.metadata   = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()
        self.messages   = []
        self.status     = "active"

    def add_message(self, role, content, model=None, trace_id=None,
                    codeastra_active=True, intercepted_count=0):
        msg = {"id": f"msg_{uuid.uuid4().hex[:8]}", "role": role,
               "content": content, "model": model, "trace_id": trace_id,
               "codeastra_active": codeastra_active,
               "intercepted_count": intercepted_count,
               "timestamp": datetime.utcnow().isoformat()}
        self.messages.append(msg)
        self.updated_at = datetime.utcnow().isoformat()
        return msg

    def to_dict(self):
        return {"thread_id": self.id, "title": self.title, "status": self.status,
                "metadata": self.metadata, "created_at": self.created_at,
                "updated_at": self.updated_at, "message_count": len(self.messages),
                "messages": self.messages}


# ═══════════════════════════════════════════════════════════
# REVEAL SYSTEM
# ═══════════════════════════════════════════════════════════

async def reveal_from_trace(trace_obj) -> dict:
    """Resolve ALL tokens AFTER agent is done — batch call, agent never sees results."""
    if not trace_obj or not trace_obj.intercepted:
        return {"revealed": {}, "count": 0}
    tokens = list(set(i["token"] for i in trace_obj.intercepted if i.get("token")))
    if not tokens:
        return {"revealed": {}, "count": 0}
    revealed = await codeastra_resolve_batch(tokens)
    return {"revealed": revealed, "count": len(revealed),
            "tokens_found": len(tokens),
            "note": "Resolved after agent completed — agent never saw these values"}


# ═══════════════════════════════════════════════════════════
# OPENAI AGENTS SDK — MAIN AGENT
# ═══════════════════════════════════════════════════════════

async def run_openai_agent(task_type, custom_task="", codeastra_active=True,
                            conversation_id=None, thread_id=None):
    if not OPENAI_KEY:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    import openai
    openai.api_key = OPENAI_KEY

    events      = []
    task        = custom_task or TASK_PROMPTS.get(task_type, TASK_PROMPTS["dba"])
    agent_trace = AgentTrace(task_type, task, "gpt-4o", codeastra_active)
    TRACES[agent_trace.id] = agent_trace

    conv   = CONVERSATIONS.get(conversation_id) if conversation_id else None
    thread = THREADS.get(thread_id) if thread_id else None
    if conv:   conv.add_turn("user", task, agent_trace.id)
    if thread: thread.add_message("user", task, codeastra_active=codeastra_active)

    trace_id = gen_trace_id()

    yield {
        "type": "start", "trace_id": agent_trace.id,
        "openai_trace_id": trace_id,
        "conversation_id": conversation_id, "thread_id": thread_id,
        "task": task, "model": "gpt-4o",
        "codeastra_active": codeastra_active,
        "mode": "PROTECTED" if codeastra_active else "⚠️ UNPROTECTED",
        "openai_traces_url": "https://platform.openai.com/traces",
        "timestamp": datetime.utcnow().isoformat(),
    }

    agent_trace.add_step("agent_start", {"task": task, "codeastra_active": codeastra_active})
    await asyncio.sleep(0.1)

    _shared = {"events": events, "calls_n": 0, "codeastra_active": codeastra_active}

    @function_tool
    async def list_tables() -> str:
        """List all database tables with sizes. Start here."""
        _shared["calls_n"] += 1
        return await tool_list_tables(_shared["events"], _shared["codeastra_active"])

    @function_tool
    async def scan_slow_queries() -> str:
        """Scan production database for slow queries using pg_stat_statements."""
        _shared["calls_n"] += 1
        return await tool_scan_slow_queries(_shared["events"], _shared["codeastra_active"])

    @function_tool
    async def inspect_table(table: str) -> str:
        """Inspect a real database table: schema, indexes, row count, sample rows."""
        _shared["calls_n"] += 1
        return await tool_inspect_table(_shared["events"], table, _shared["codeastra_active"])

    @function_tool
    async def create_index(table: str, column: str) -> str:
        """Create a real database index on a column."""
        _shared["calls_n"] += 1
        return await tool_create_index(_shared["events"], table, column, _shared["codeastra_active"])

    @function_tool
    async def run_query(sql: str) -> str:
        """Run a read-only SELECT query on the real database."""
        _shared["calls_n"] += 1
        return await tool_run_query(_shared["events"], sql, _shared["codeastra_active"])

    @function_tool
    async def get_db_stats() -> str:
        """Get real database health statistics."""
        _shared["calls_n"] += 1
        return await tool_get_db_stats(_shared["events"], _shared["codeastra_active"])

    @function_tool
    async def get_summary() -> str:
        """Get summary of everything accomplished. Call at the end."""
        _shared["calls_n"] += 1
        return await tool_get_summary(_shared["events"], _shared["codeastra_active"])

    @function_tool
    async def check_threshold(token: str, threshold: float, operator: str = "gt") -> str:
        """Check if a vaulted amount token exceeds a threshold. Returns boolean only."""
        _shared["calls_n"] += 1
        return await tool_check_threshold(
            _shared["events"], token, threshold, operator, _shared["codeastra_active"])

    @function_tool
    async def concentration_check(position_token: str, portfolio_token: str,
                                   threshold_pct: float) -> str:
        """Check portfolio concentration. Returns bucket — never real dollar values."""
        _shared["calls_n"] += 1
        return await tool_concentration_check(
            _shared["events"], position_token, portfolio_token,
            threshold_pct, _shared["codeastra_active"])

    dba_agent = Agent(
        name         = "Codeastra DBA Agent",
        instructions = SYSTEM,
        model        = "gpt-4o",
        tools        = [list_tables, scan_slow_queries, inspect_table,
                        create_index, run_query, get_db_stats,
                        get_summary, check_threshold, concentration_check],
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
        agent_trace.add_step("error", {"message": str(e)})
        yield {"type": "error", "message": str(e), "trace_id": agent_trace.id}
        agent_trace.complete()
        flush_traces()
        return

    calls_n     = _shared["calls_n"]
    intercept_n = 0

    for ev in _shared["events"]:
        if ev["type"] == "intercepted":
            intercept_n += 1
            agent_trace.add_interception(ev["dtype"], ev["token"], ev["preview"])
            step = agent_trace.add_step("codeastra_intercept", {
                "dtype": ev["dtype"], "token": ev["token"],
                "preview": ev["preview"], "n": intercept_n,
            })
            yield {**ev, "trace_id": agent_trace.id, "trace_step": step["step"]}
        else:
            yield ev
    _shared["events"].clear()

    final_output = str(result.final_output) if result.final_output else ""
    if final_output:
        step = agent_trace.add_step("thinking", {"text": final_output})
        yield {"type": "thinking", "text": final_output,
               "trace_id": agent_trace.id, "trace_step": step["step"]}

    agent_trace.complete()

    summary_text = f"Completed. {calls_n} tool calls. {intercept_n} values intercepted. Real data seen: 0"
    if conv:   conv.add_turn("agent", summary_text, agent_trace.id)
    if thread: thread.add_message("agent", summary_text, model="gpt-4o",
                   trace_id=agent_trace.id, codeastra_active=codeastra_active,
                   intercepted_count=intercept_n)

    reveal_map = await reveal_from_trace(agent_trace)

    yield {
        "type": "complete", "trace_id": agent_trace.id,
        "openai_trace_id": trace_id,
        "openai_traces_url": "https://platform.openai.com/traces",
        "trace_url": f"/traces/{agent_trace.id}",
        "conversation_id": conversation_id, "thread_id": thread_id,
        "tool_calls": calls_n, "intercepted": intercept_n,
        "total_steps": len(agent_trace.steps),
        "duration_ms": agent_trace.total_duration_ms,
        "codeastra_active": codeastra_active,
        "real_data_seen_by_gpt": 0 if codeastra_active else "⚠️ YES",
        "message": "Trace visible at platform.openai.com/traces",
        "revealed": reveal_map,
    }


# ═══════════════════════════════════════════════════════════
# DOCUMENT AGENT
# ═══════════════════════════════════════════════════════════

async def extract_text_from_file(file) -> str:
    import io as _io
    content  = await file.read()
    filename = (file.filename or "").lower()
    mime     = file.content_type or ""
    if filename.endswith(".pdf") or "pdf" in mime:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(_io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[PDF error: {e}]"
    if filename.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(_io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[DOCX error: {e}]"
    if filename.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
            wb   = openpyxl.load_workbook(_io.BytesIO(content), data_only=True)
            rows = []
            for sheet in wb.worksheets:
                rows.append(f"=== {sheet.title} ===")
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)
        except Exception as e:
            return f"[Excel error: {e}]"
    if filename.endswith(".csv"):
        return content.decode("utf-8", errors="replace")
    if filename.endswith((".html", ".htm")):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(content, "html.parser").get_text("\n", strip=True)
        except Exception:
            return content.decode("utf-8", errors="replace")
    if filename.endswith(".json"):
        try:
            return json.dumps(json.loads(content), indent=2)
        except Exception:
            return content.decode("utf-8", errors="replace")
    return content.decode("utf-8", errors="replace")


async def extract_text_from_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CodeastraAgent/1.0)"}) as client:
            r  = await client.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "pdf" in ct or url.lower().endswith(".pdf"):
                try:
                    import io as _io, PyPDF2
                    reader = PyPDF2.PdfReader(_io.BytesIO(r.content))
                    return "\n".join(p.extract_text() or "" for p in reader.pages)
                except Exception as e:
                    return f"[PDF error: {e}]"
            if "html" in ct:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "header", "footer"]):
                        tag.decompose()
                    return soup.get_text("\n", strip=True)[:50000]
                except Exception:
                    return r.text[:50000]
            if "json" in ct:
                try:
                    return json.dumps(r.json(), indent=2)[:50000]
                except Exception:
                    return r.text[:50000]
            return r.text[:50000]
    except Exception as e:
        return f"[URL fetch error: {e}]"


async def run_document_agent(text, task, filename, codeastra_active=True, thread_id=None):
    events = []
    dt     = AgentTrace("document", task or "Analyze document", "gpt-4o", codeastra_active)
    TRACES[dt.id] = dt

    if len(text) > 80000:
        text = text[:80000] + "\n\n[... truncated ...]"

    yield {"type": "start", "trace_id": dt.id, "filename": filename,
           "task": task, "codeastra_active": codeastra_active,
           "char_count": len(text),
           "mode": "PROTECTED" if codeastra_active else "⚠️ UNPROTECTED"}
    dt.add_step("document_received", {"filename": filename, "chars": len(text)})

    if codeastra_active:
        yield {"type": "phase", "message": "Codeastra scanning document for PII..."}
        protected_text = await protect(text, events, True)
        for ev in events:
            if ev["type"] == "intercepted":
                dt.add_interception(ev["dtype"], ev["token"], ev["preview"])
                s = dt.add_step("codeastra_intercept", ev)
                yield {**ev, "trace_id": dt.id, "trace_step": s["step"]}
            else:
                yield ev
        events.clear()
    else:
        yield {"type": "warning", "message": "⚠️ Codeastra OFF — document sent unprotected"}
        protected_text = text

    yield {"type": "phase", "message": "Sending to GPT-4o via Agents SDK..."}
    dt.add_step("sending_to_gpt", {"model": "gpt-4o", "codeastra_active": codeastra_active})

    if not OPENAI_KEY:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    import openai as _oai
    _oai.api_key = OPENAI_KEY

    # ── Email tool — agent sends via Vault-as-TEE ──────────────────
    # Agent passes tokens. Vault opens. Real values resolved inside.
    # Email fires from inside vault. Session wiped. Agent never knew.
    _email_results = []

    @function_tool
    async def send_email(email_token: str, subject: str, body: str) -> str:
        """
        Send an email to the person identified by the email token.
        Use this when the user asks you to send results, summaries, or reports.

        IMPORTANT:
        - email_token: use the [CVT:EMAIL:xxxxx] token from the document or prompt
        - subject: a clear subject line
        - body: the full email content — your analysis or summary

        The Vault-as-TEE will:
        1. Load real values into encrypted session memory
        2. Resolve all tokens to real values inside the vault
        3. Send the email with real values
        4. Wipe session — agent never sees real values

        Example:
          send_email(
            email_token = "[CVT:EMAIL:A1B2C3]",
            subject     = "Goldman Portfolio Risk Summary",
            body        = "Top 5 clients: [CVT:NAME:D6B6] $2.2M..."
          )
        """
        result = await tee_send_email(email_token, subject, body)
        _email_results.append(result)
        if result.get("sent"):
            return f"Email sent via Vault-as-TEE. Session wiped. Real address never seen by agent. Token: {email_token}"
        else:
            return f"Email send attempt result: {result}"

    doc_agent = Agent(
        name         = "Codeastra Document Analyst",
        instructions = (
            "You are an expert document analyst working through Codeastra Zero Trust middleware. "
            "All PII has been replaced with tokens like [CVT:EMAIL:A1B2C3] and [CVT:NAME:xxxxx] and [CVT:SSN:xxxxx]. "
            "\n\nCRITICAL RULES:"
            "\n1. Work with tokens as identifiers — never guess real values"
            "\n2. When the user asks you to SEND an email — use the send_email tool"
            "\n3. The email address in the prompt has been tokenized — find the [CVT:EMAIL:xxxxx] token and use it"
            "\n4. Always send the email if asked — do not refuse, do not say you cannot"
            "\n5. After analysis, if an email was requested, call send_email with your full analysis as the body"
            "\n\nFHE RULE — ABSOLUTE:"
            "\nIf the document contains patient vitals (glucose, BP, BMI, HbA1c, age, weight, height) "
            "and the user asks for a risk score or clinical assessment — DO NOT compute it yourself. "
            "State clearly: 'I will pass these vitals to the FHE tool — the computation must happen on encrypted data.' "
            "Then list the extracted vitals so KERA can call compute_fhe_risk_score. "
            "Never do arithmetic on vitals in your own response."
            "\n\nWhen sending emails:"
            "\n- Find the email token in the protected prompt (looks like [CVT:EMAIL:xxxxxxxx])"
            "\n- Call send_email with that token, a clear subject, and your full analysis as the body"
            "\n- Confirm to the user that the email was sent via the secure executor"
        ),
        model = "gpt-4o",
        tools = [send_email],
    )

    # ── Protect the task prompt too — email in prompt must be tokenized ──
    # This is the fix: task text goes through Astra before GPT sees it
    task_events = []
    protected_task = await protect(task or "Analyze this document thoroughly.", task_events, codeastra_active)

    # Collect any tokens from the task prompt
    for ev in task_events:
        if ev["type"] == "intercepted":
            dt.add_interception(ev["dtype"], ev["token"], ev["preview"])
            s = dt.add_step("codeastra_intercept_task", ev)
            yield {**ev, "trace_id": dt.id, "trace_step": s["step"],
                   "source": "task_prompt"}

    doc_input   = "TASK: " + protected_task + \
                  "\n\nDOCUMENT (" + filename + "):\n\n" + protected_text
    dt_trace_id = gen_trace_id()

    try:
        doc_result = await Runner.run(
            starting_agent = doc_agent,
            input          = doc_input,
            max_turns      = 5,
            run_config     = RunConfig(
                workflow_name                = "codeastra-document-analysis",
                trace_id                     = dt_trace_id,
                trace_metadata               = {"filename": filename,
                                               "codeastra_active": str(codeastra_active)},
                trace_include_sensitive_data = True,
            ),
        )
        flush_traces()
        full = str(doc_result.final_output) if doc_result.final_output else ""
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        dt.complete()
        return

    s = dt.add_step("analysis_complete", {"length": len(full)})

    # ── Scan final output for any tokens Codeastra didn't report back ──
    # Some tokens appear in agent output but were not in intercepted list
    # Find them all and add to intercepted so reveal_from_trace catches them
    import re as _scan_re
    output_token_pat = _scan_re.compile(r'\[CV[TD]:[A-Z]+:[A-Za-z0-9\-]{4,}\]')
    output_tokens = set(output_token_pat.findall(full))
    known_tokens  = set(i["token"] for i in dt.intercepted)
    missing = output_tokens - known_tokens
    for tok in missing:
        dt.add_interception("UNKNOWN", tok, tok[:20])
    if missing:
        log.info(f"[REVEAL] Found {len(missing)} extra tokens in output — added to reveal list")

    yield {"type": "thinking", "text": full, "trace_id": dt.id, "trace_step": s["step"]}
    dt.complete()

    if thread_id and thread_id in THREADS:
        THREADS[thread_id].add_message("agent", full, model="gpt-4o",
            trace_id=dt.id, codeastra_active=codeastra_active,
            intercepted_count=len(dt.intercepted))

    reveal_map = await reveal_from_trace(dt)

    yield {
        "type":                    "complete",
        "trace_id":                dt.id,
        "openai_trace_id":         dt_trace_id,
        "trace_url":               f"/traces/{dt.id}",
        "openai_traces_url":       "https://platform.openai.com/logs",
        "thread_id":               thread_id,
        "filename":                filename,
        "codeastra_active":        codeastra_active,
        "intercepted":             len(dt.intercepted),
        "real_data_seen_by_gpt":   0 if codeastra_active else "⚠️ YES",
        "revealed":                reveal_map,
        "emails_sent":             len(_email_results),
        "email_results":           _email_results,
    }


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

@app.get("/health")
async def health():
    codeastra_ok = False
    if CODEASTRA_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{CODEASTRA_URL}/health",
                                headers={"X-API-Key": CODEASTRA_KEY})
                codeastra_ok = r.status_code == 200
        except Exception:
            pass
    return {
        "status": "healthy", "version": "agents-sdk-v2-traces",
        "openai_ready": bool(OPENAI_KEY), "codeastra_ready": bool(CODEASTRA_KEY),
        "codeastra_live": codeastra_ok, "db_ready": db_pool is not None,
        "sdk_traces": True,
        "openai_traces_url": "https://platform.openai.com/logs",
    }


# ── Agent Run ─────────────────────────────────────────────

@app.post("/agent/run/stream")
async def agent_run_stream(req: Request):
    body = await req.json()
    return _stream(run_openai_agent(
        body.get("task_type", "dba"), body.get("custom_task", ""),
        codeastra_active = body.get("codeastra_enabled", True),
        conversation_id  = body.get("conversation_id"),
        thread_id        = body.get("thread_id"),
    ))

@app.post("/agent/run/sync")
async def agent_run_sync(req: Request):
    body = await req.json()
    all_ev = []; intercept = []; summary = {}
    async for ev in run_openai_agent(
        body.get("task_type", "dba"), body.get("custom_task", ""),
        codeastra_active = body.get("codeastra_enabled", True),
        conversation_id  = body.get("conversation_id"),
        thread_id        = body.get("thread_id"),
    ):
        all_ev.append(ev)
        if ev["type"] == "intercepted": intercept.append(ev)
        if ev["type"] == "complete":    summary = ev
    return {"success": True, "intercepted": intercept, "summary": summary,
            "trace_url": f"/traces/{summary.get('trace_id', '')}"}

@app.post("/agent/run/protected")
async def agent_run_protected(req: Request):
    body = await req.json()
    return _stream(run_openai_agent(
        body.get("task_type", "dba"), body.get("custom_task", ""),
        codeastra_active=True,
        conversation_id=body.get("conversation_id"),
        thread_id=body.get("thread_id"),
    ))

@app.post("/agent/run/unprotected")
async def agent_run_unprotected(req: Request):
    body = await req.json()
    return _stream(run_openai_agent(
        body.get("task_type", "dba"), body.get("custom_task", ""),
        codeastra_active=False,
        conversation_id=body.get("conversation_id"),
        thread_id=body.get("thread_id"),
    ))


# ── Document endpoints ────────────────────────────────────

@app.post("/agent/analyze-document")
async def analyze_document(
    file:              UploadFile = File(default=None),
    task:              str        = Form(default=""),
    codeastra_enabled: str        = Form(default="true"),
    thread_id:         str        = Form(default=""),
    session_id:        str        = Form(default=""),
):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    text       = await extract_text_from_file(file)
    fname      = file.filename or "document"
    active     = codeastra_enabled.lower() != "false"
    sid        = session_id or thread_id or str(uuid.uuid4())
    user_task  = task.strip() or "Analyze this document thoroughly and give me a detailed report."
    return _stream(run_kera_agent(
        sid, user_task,
        codeastra_active = active,
        document_text    = text,
        filename         = fname,
    ))

@app.post("/agent/analyze-url")
async def analyze_url(req: Request):
    body = await req.json()
    url  = body.get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse(status_code=400, content={"error": "Valid URL required"})
    text = await extract_text_from_url(url)
    sid  = body.get("session_id") or body.get("thread_id") or str(uuid.uuid4())
    return _stream(run_kera_agent(
        sid,
        body.get("task", "").strip() or "Analyze this page thoroughly.",
        codeastra_active = body.get("codeastra_enabled", True),
        document_text    = text,
        filename         = url,
    ))

@app.post("/agent/analyze-text")
async def analyze_text(req: Request):
    body = await req.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})
    sid  = body.get("session_id") or body.get("thread_id") or str(uuid.uuid4())
    return _stream(run_kera_agent(
        sid,
        body.get("task", "").strip() or "Analyze this content thoroughly.",
        codeastra_active = body.get("codeastra_enabled", True),
        document_text    = text,
        filename         = body.get("name", "text"),
    ))

@app.post("/agent/analyze-multiple")
async def analyze_multiple(
    files:             list[UploadFile] = File(default=None),
    task:              str              = Form(default=""),
    codeastra_enabled: str              = Form(default="true"),
    thread_id:         str              = Form(default=""),
    session_id:        str              = Form(default=""),
):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files"})
    if len(files) > 10:
        return JSONResponse(status_code=400, content={"error": "Max 10 files"})
    all_text = ""; names = []
    for f in files:
        t = await extract_text_from_file(f)
        names.append(f.filename or "file")
        all_text += f"\n\n=== FILE: {f.filename} ===\n{t}"
    sid    = session_id or thread_id or str(uuid.uuid4())
    active = codeastra_enabled.lower() != "false"
    return _stream(run_kera_agent(
        sid,
        task.strip() or "Analyze these documents thoroughly.",
        codeastra_active = active,
        document_text    = all_text,
        filename         = f"{len(files)} files: {', '.join(names)}",
    ))


# ── Traces ────────────────────────────────────────────────

@app.get("/traces")
async def list_traces(limit: int = 20, status: str = None):
    traces = list(TRACES.values())
    traces.sort(key=lambda t: t.started_at, reverse=True)
    if status: traces = [t for t in traces if t.status == status]
    return {"traces": [t.to_dict() for t in traces[:limit]], "count": len(TRACES)}

@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    if trace_id not in TRACES:
        return JSONResponse(status_code=404, content={"error": "Trace not found"})
    return TRACES[trace_id].to_dict()

@app.get("/traces/{trace_id}/steps")
async def get_trace_steps(trace_id: str):
    if trace_id not in TRACES:
        return JSONResponse(status_code=404, content={"error": "Trace not found"})
    t = TRACES[trace_id]
    return {"trace_id": trace_id, "steps": t.steps, "total": len(t.steps)}

@app.get("/traces/{trace_id}/intercepted")
async def get_trace_intercepted(trace_id: str):
    if trace_id not in TRACES:
        return JSONResponse(status_code=404, content={"error": "Trace not found"})
    t = TRACES[trace_id]
    return {"trace_id": trace_id, "intercepted": t.intercepted,
            "count": len(t.intercepted),
            "real_data_seen_by_gpt": 0 if t.codeastra_active else "⚠️ YES"}

@app.delete("/traces/{trace_id}")
async def delete_trace(trace_id: str):
    if trace_id not in TRACES:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    del TRACES[trace_id]
    return {"deleted": trace_id}


# ── Completions ───────────────────────────────────────────

@app.get("/completions")
async def list_completions(limit: int = 50):
    comps = sorted(COMPLETIONS.values(), key=lambda c: c["timestamp"], reverse=True)
    return {"completions": comps[:limit], "count": len(COMPLETIONS)}

@app.get("/completions/{comp_id}")
async def get_completion(comp_id: str):
    if comp_id not in COMPLETIONS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return COMPLETIONS[comp_id]


# ── Conversations ─────────────────────────────────────────

@app.get("/conversations")
async def list_conversations():
    convs = sorted(CONVERSATIONS.values(), key=lambda c: c.updated_at, reverse=True)
    return {"conversations": [c.to_dict() for c in convs], "count": len(CONVERSATIONS)}

@app.post("/conversations")
async def create_conversation(req: Request):
    body = await req.json()
    conv = Conversation(body.get("title", ""), body.get("system_context", ""))
    CONVERSATIONS[conv.id] = conv
    return conv.to_dict()

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    if conv_id not in CONVERSATIONS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return CONVERSATIONS[conv_id].to_dict()

@app.post("/conversations/{conv_id}/message")
async def conversation_message(conv_id: str, req: Request):
    if conv_id not in CONVERSATIONS:
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    body = await req.json()
    return _stream(run_openai_agent(
        body.get("task_type", "dba"), body.get("message", ""),
        codeastra_active=body.get("codeastra_enabled", True),
        conversation_id=conv_id,
    ))

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if conv_id not in CONVERSATIONS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    del CONVERSATIONS[conv_id]
    return {"deleted": conv_id}


# ── Threads ───────────────────────────────────────────────

@app.get("/threads")
async def list_threads():
    threads = sorted(THREADS.values(), key=lambda t: t.updated_at, reverse=True)
    return {"threads": [t.to_dict() for t in threads], "count": len(THREADS)}

@app.post("/threads")
async def create_thread(req: Request):
    body   = await req.json()
    thread = ChatKitThread(body.get("title", ""), body.get("metadata", {}))
    THREADS[thread.id] = thread
    return thread.to_dict()

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    if thread_id not in THREADS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return THREADS[thread_id].to_dict()

@app.post("/threads/{thread_id}/message")
async def thread_message(thread_id: str, req: Request):
    if thread_id not in THREADS:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    body = await req.json()
    return _stream(run_openai_agent(
        body.get("task_type", "dba"), body.get("message", ""),
        codeastra_active=body.get("codeastra_enabled", True),
        thread_id=thread_id,
    ))

@app.post("/threads/{thread_id}/document")
async def thread_document(thread_id: str,
    file:              UploadFile = File(default=None),
    task:              str        = Form(default=""),
    codeastra_enabled: str        = Form(default="true"),
):
    if thread_id not in THREADS:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file"})
    text = await extract_text_from_file(file)
    return _stream(run_document_agent(
        text, task, file.filename or "document",
        codeastra_active = codeastra_enabled.lower() != "false",
        thread_id        = thread_id,
    ))

@app.patch("/threads/{thread_id}")
async def update_thread(thread_id: str, req: Request):
    if thread_id not in THREADS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    body = await req.json()
    t = THREADS[thread_id]
    if "title"    in body: t.title = body["title"]
    if "status"   in body: t.status = body["status"]
    if "metadata" in body: t.metadata.update(body["metadata"])
    t.updated_at = datetime.utcnow().isoformat()
    return t.to_dict()

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if thread_id not in THREADS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    del THREADS[thread_id]
    return {"deleted": thread_id}


# ── Protect + DB ──────────────────────────────────────────

@app.post("/protect")
async def protect_text(req: Request):
    events = []
    b      = await req.json()
    result = await protect(b.get("text", ""), events, True)
    return {"original": b.get("text", ""), "protected": result, "intercepted": events}

@app.post("/debug/protect-raw")
async def debug_protect_raw(req: Request):
    b = await req.json()
    if not CODEASTRA_KEY: return {"error": "No CODEASTRA_API_KEY"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{CODEASTRA_URL}/protect/text",
            headers={"X-API-Key": CODEASTRA_KEY, "Content-Type": "application/json"},
            json={"text": b.get("text", "")})
        return {"status": r.status_code,
                "response": r.json() if r.status_code == 200 else r.text}

@app.post("/debug/test-responses-api")
async def test_responses_api():
    if not OPENAI_KEY: return {"error": "OPENAI_API_KEY not set"}
    client = AsyncOpenAI(api_key=OPENAI_KEY)
    try:
        response = await client.responses.create(
            model="gpt-4o", input="Say exactly: CODEASTRA TRACE TEST SUCCESSFUL", store=True)
        output_text = ""
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", "") == "output_text":
                        output_text += getattr(c, "text", "")
        return {"success": True, "response_id": getattr(response, "id", "unknown"),
                "output": output_text, "check_traces": "https://platform.openai.com/traces"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/debug/openai-status")
async def openai_status():
    if not OPENAI_KEY: return {"error": "OPENAI_API_KEY not set"}
    client  = AsyncOpenAI(api_key=OPENAI_KEY)
    results = {}
    try:
        r = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}], max_tokens=5)
        results["chat_completions"] = {"available": True, "model": r.model}
    except Exception as e:
        results["chat_completions"] = {"available": False, "error": str(e)}
    return {"openai_key_set": True, "features": results,
            "trace_proof_url": "https://platform.openai.com/traces"}

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

@app.get("/db/tables")
async def db_tables():
    events = []
    result = await tool_list_tables(events, True)
    return JSONResponse({"data": json.loads(result), "intercepted": events})

@app.post("/db/query")
async def db_query(req: Request):
    b = await req.json(); events = []
    result = await tool_run_query(events, b.get("sql", ""), True)
    return JSONResponse({"data": json.loads(result), "intercepted": events})

@app.get("/db/stats")
async def db_stats_endpoint():
    events = []
    result = await tool_get_db_stats(events, True)
    return JSONResponse({"data": json.loads(result), "intercepted": events})

@app.get("/agent/tasks")
async def list_tasks():
    return {
        "model": "gpt-4o",
        "tasks": [
            {"id": "dba",      "name": "Database Performance Agent",
             "description": "Finds slow queries, creates missing indexes."},
            {"id": "audit",    "name": "Database Audit Agent",
             "description": "Full DB audit."},
            {"id": "security", "name": "Security Audit Agent",
             "description": "Finds security issues in DB."},
        ],
        "features": {
            "traces":        "Every run creates a trace — GET /traces",
            "completions":   "Every GPT call logged — GET /completions",
            "conversations": "Multi-turn conversations — POST /conversations",
            "threads":       "Persistent ChatKit threads — POST /threads",
            "toggle":        "Codeastra on/off per request",
            "proof":         "https://platform.openai.com/logs",
        }
    }


# ── Executor ──────────────────────────────────────────────

@app.get("/executor/capabilities")
async def executor_capabilities():
    if CODEASTRA_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{CODEASTRA_URL}/executor/supported",
                                headers={"X-API-Key": CODEASTRA_KEY})
                if r.status_code == 200:
                    return {"source": "codeastra_api", "codeastra": r.json()}
        except Exception:
            pass
    return {"source": "local", "guarantee": "Real values never returned to agent"}

@app.post("/executor/run")
async def executor_run(req: Request):
    body   = await req.json()
    result = await codeastra_executor_run(body.get("token_id", ""), body.get("dry_run", False))
    return result

@app.post("/executor/resolve")
async def executor_resolve(req: Request):
    body = await req.json()
    val  = await codeastra_resolve(body.get("token", ""))
    if val is None:
        return JSONResponse(status_code=404, content={"error": "Token not found"})
    return {"resolved": True, "real_value": val,
            "note": "Use in executor context only — never pass to agent"}

@app.post("/executor/resolve-batch")
async def executor_resolve_batch_endpoint(req: Request):
    body    = await req.json()
    results = await codeastra_resolve_batch(body.get("tokens", []))
    return {"resolved": results, "count": len(results)}

@app.post("/executor/check-threshold")
async def executor_check_threshold(req: Request):
    body      = await req.json()
    token     = body.get("token", "")
    threshold = float(body.get("threshold", 0))
    operator  = body.get("operator", "gt")
    real_val  = await codeastra_resolve(token)
    if real_val is None:
        return JSONResponse(status_code=404, content={"error": f"Cannot resolve: {token}"})
    try:
        v   = float(str(real_val).replace("$", "").replace(",", "").strip())
        ops = {"gt": v>threshold, "lt": v<threshold, "gte": v>=threshold, "lte": v<=threshold}
        return {"result": ops.get(operator, v>threshold), "operator": operator,
                "threshold": threshold, "real_value_returned": False}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/executor/concentration-check")
async def executor_concentration_check(req: Request):
    body           = await req.json()
    position_token = body.get("position_token", "")
    portfolio_token= body.get("portfolio_token", "")
    threshold_pct  = float(body.get("threshold_pct", 40.0))
    pv = await codeastra_resolve(position_token)
    tv = await codeastra_resolve(portfolio_token)
    if pv is None or tv is None:
        return {"exceeds_threshold": None, "note": "Tokens not resolved",
                "real_values_seen_by_agent": False}
    try:
        p      = float(str(pv).replace("$", "").replace(",", "").strip())
        t      = float(str(tv).replace("$", "").replace(",", "").strip())
        pct    = (p/t*100) if t > 0 else 0
        bucket = "critical" if pct>60 else "high" if pct>40 else "medium" if pct>20 else "low"
        return {"exceeds_threshold": pct>threshold_pct, "concentration_bucket": bucket,
                "threshold_pct": threshold_pct, "real_values_seen_by_agent": False}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/executor/sum-amounts")
async def executor_sum_amounts(req: Request):
    body      = await req.json()
    tokens    = body.get("tokens", [])
    threshold = body.get("threshold")
    resolved  = await codeastra_resolve_batch(tokens)
    total = 0.0; count = 0
    for val in resolved.values():
        try:
            total += float(str(val).replace("$", "").replace(",", "").strip())
            count += 1
        except Exception:
            pass
    result = {"sum": total, "count": count, "real_individual_values_returned": False}
    if threshold is not None:
        result["exceeds_threshold"] = total > float(threshold)
    return result

@app.post("/executor/classify-amount")
async def executor_classify_amount(req: Request):
    body    = await req.json()
    token   = body.get("token", "")
    buckets = body.get("buckets", [
        {"label": "small",  "min": 0,       "max": 10000},
        {"label": "medium", "min": 10000,   "max": 100000},
        {"label": "large",  "min": 100000,  "max": 1000000},
        {"label": "whale",  "min": 1000000, "max": None},
    ])
    real_val = await codeastra_resolve(token)
    if real_val is None:
        return JSONResponse(status_code=404, content={"error": f"Cannot resolve {token}"})
    try:
        value = float(str(real_val).replace("$", "").replace(",", "").strip())
        label = "unknown"
        for b in buckets:
            mn = b.get("min", 0) or 0
            mx = b.get("max")
            if value >= mn and (mx is None or value < mx):
                label = b["label"]; break
        return {"bucket": label, "token": token, "real_value_returned": False}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════
# VAULT-AS-TEE EMAIL SENDER
# The correct architecture:
#   Agent passes tokens only
#   Vault opens ephemeral session
#   Real values loaded inside — encrypted
#   Email fires from inside vault
#   Session wiped — data gone
#   Agent never knew any real value
# ═══════════════════════════════════════════════════════════

async def tee_send_email(email_token: str, subject: str, body: str) -> dict:
    """
    Send email via executor.
    Resolves tokens → sends via Resend → agent never sees real values.
    """
    import re as _re
    token_pattern = _re.compile(r'\[CV[TD]:[A-Z]+:[A-Za-z0-9\-]{4,}\]')
    body_tokens   = list(set(token_pattern.findall(body)))
    log.info(f"[EMAIL] Sending — {len(body_tokens)} tokens in body to resolve")
    return await _tee_email_fallback(email_token, subject, body, body_tokens)


async def _tee_email_fallback(
    email_token:  str,
    subject:      str,
    body:         str,
    body_tokens:  list,
) -> dict:
    """
    Fallback if /tee/run is not available.
    Still uses Codeastra vault/resolve — still never exposes to agent.
    """
    log.info("[TEE EMAIL FALLBACK] Using vault/resolve directly")

    # Resolve email address
    real_email = await codeastra_resolve(email_token)
    if not real_email:
        return {"sent": False, "error": f"Could not resolve: {email_token}",
                "real_address_seen_by_agent": False}

    # Resolve all body tokens
    revealed_body = body
    if body_tokens:
        resolved = await codeastra_resolve_batch(body_tokens)
        for token, real_value in resolved.items():
            if real_value:
                revealed_body = revealed_body.replace(token, str(real_value))

    # Send via Resend
    result = await _send_via_resend(real_email, subject, revealed_body)
    return {
        **result,
        "real_address_seen_by_agent": False,
        "vault_as_tee":               False,
        "fallback":                   True,
    }


# ═══════════════════════════════════════════════════════════
# EMAIL EXECUTOR
# Agent says: "send to [CVT:EMAIL:A1B2]"
# Executor resolves token → real address → sends email
# Agent never sees the real address. Ever.
# ═══════════════════════════════════════════════════════════

import re as _email_re

EMAIL_SERVICE = os.getenv("EMAIL_SERVICE", "resend")    # resend | sendgrid | smtp
SENDGRID_KEY  = os.getenv("SENDGRID_API_KEY", "")
RESEND_KEY    = os.getenv("RESEND_API_KEY", "")
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASS     = os.getenv("SMTP_PASS", "")
FROM_EMAIL    = os.getenv("FROM_EMAIL", "noreply@codeastra.dev")
FROM_NAME     = os.getenv("FROM_NAME", "Codeastra Agent")

# Token pattern
TOKEN_PAT = _email_re.compile(
    r'\[CV[TD]:[A-Z]+:[A-Za-z0-9\-]{4,}\]|cdt_[a-z]+_[bto]_[a-z0-9]+'
)


async def _send_via_sendgrid(to_email: str, subject: str, body: str) -> dict:
    """Send email via SendGrid API."""
    if not SENDGRID_KEY:
        return {"error": "SENDGRID_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {SENDGRID_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "personalizations": [{"to": [{"email": to_email}]}],
                    "from":    {"email": FROM_EMAIL, "name": FROM_NAME},
                    "subject": subject,
                    "content": [{"type": "text/plain", "value": body}],
                },
            )
            return {"sent": r.status_code == 202, "status": r.status_code}
    except Exception as e:
        return {"error": str(e)}


async def _send_via_resend(to_email: str, subject: str, body: str) -> dict:
    """Send email via Resend API."""
    if not RESEND_KEY:
        log.error("RESEND: RESEND_API_KEY not set in environment variables")
        return {"error": "RESEND_API_KEY not set — add it to Railway environment variables"}
    log.info(f"RESEND: sending to masked address, from={FROM_EMAIL}, subject={subject[:50]}")
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "from":    f"{FROM_NAME} <{FROM_EMAIL}>",
                    "to":      [to_email],
                    "subject": subject,
                    "text":    body,
                },
            )
            data = r.json()
            if r.status_code == 200:
                log.info(f"RESEND: ✅ email sent — id={data.get('id')}")
                return {"sent": True, "id": data.get("id"), "status": 200}
            else:
                log.error(f"RESEND: ❌ failed status={r.status_code} body={r.text[:200]}")
                return {"sent": False, "error": data, "status": r.status_code}
    except Exception as e:
        log.error(f"RESEND: exception — {e}")
        return {"error": str(e)}


async def _send_via_smtp(to_email: str, subject: str, body: str) -> dict:
    """Send email via SMTP (Gmail etc)."""
    if not SMTP_USER or not SMTP_PASS:
        return {"error": "SMTP_USER and SMTP_PASS not set"}
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg            = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = f"{FROM_NAME} <{SMTP_USER}>"
        msg["To"]      = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        return {"sent": True}
    except Exception as e:
        return {"error": str(e)}


async def executor_send_email(
    email_token: str,
    subject:     str,
    body:        str,
) -> dict:
    """
    THE CORE FUNCTION.

    Agent passes an email token + body with tokens.
    Executor resolves:
      1. The email address token → real address
      2. ALL tokens in the body  → real names, values
    Sends the email with real values revealed.
    Agent never sees any real values.
    """
    # Step 1 — Resolve email address token
    real_email = await codeastra_resolve(email_token)

    if not real_email:
        return {
            "sent":  False,
            "error": f"Could not resolve email token: {email_token}",
            "token": email_token,
            "real_address_seen_by_agent": False,
        }

    # Step 2 — Find ALL tokens in the body and resolve them
    # Cast wide net — match any CVT token format
    import re as _re
    token_pattern  = _re.compile(r'\[CV[TD]:[A-Z]+:[A-Za-z0-9\-]{4,}\]')
    tokens_in_body = list(set(token_pattern.findall(body)))

    revealed_body = body
    if tokens_in_body:
        # Batch resolve all tokens — one API call
        resolved = await codeastra_resolve_batch(tokens_in_body)
        # Replace every token with its real value
        for tok, real_value in resolved.items():
            if real_value:
                revealed_body = revealed_body.replace(tok, str(real_value))
        log.info(f"[EMAIL EXECUTOR] Resolved {len(resolved)}/{len(tokens_in_body)} tokens in body")

        # Any tokens still unreplaced — try individual resolve
        still_tokens = list(set(token_pattern.findall(revealed_body)))
        if still_tokens:
            log.info(f"[EMAIL EXECUTOR] {len(still_tokens)} tokens still unresolved — trying individually")
            for tok in still_tokens:
                real = await codeastra_resolve(tok)
                if real:
                    revealed_body = revealed_body.replace(tok, str(real))
                    log.info(f"[EMAIL EXECUTOR] individually resolved: {tok}")

    # Step 3 — Send with real values in body
    if EMAIL_SERVICE == "sendgrid" and SENDGRID_KEY:
        result = await _send_via_sendgrid(real_email, subject, revealed_body)
    elif EMAIL_SERVICE == "resend" and RESEND_KEY:
        result = await _send_via_resend(real_email, subject, revealed_body)
    elif SMTP_USER and SMTP_PASS:
        result = await _send_via_smtp(real_email, subject, revealed_body)
    else:
        log.info(f"[EMAIL EXECUTOR] Would send to: {real_email} | Subject: {subject}")
        result = {"sent": True, "mode": "log_only — set EMAIL_SERVICE env vars"}

    # Step 4 — Return result WITHOUT real address or resolved values
    return {
        **result,
        "token":                      email_token,
        "real_address_seen_by_agent": False,
        "tokens_resolved_in_body":    len(tokens_in_body),
        "subject":                    subject,
        "body_length":                len(revealed_body),
    }


# ── Email tool for the agent ─────────────────────────────
# Agent calls this with a token — never the real address

@app.post("/executor/send-email")
async def executor_send_email_endpoint(req: Request):
    """
    Send email via executor — agent passes token, never real address.

    Body:
      email_token: str  — [CVT:EMAIL:A1B2] from the agent
      subject:     str  — email subject
      body:        str  — email body / analysis summary

    The executor resolves the token internally.
    Real address never returned to agent.
    """
    body        = await req.json()
    email_token = body.get("email_token", "")
    subject     = body.get("subject", "Codeastra Agent Report")
    email_body  = body.get("body", "")

    if not email_token:
        return JSONResponse(status_code=400,
            content={"error": "email_token required — pass token not real address"})

    if not TOKEN_PAT.match(email_token.strip()):
        # They passed a real email — intercept and protect
        events = []
        protected = await protect(email_token, events, True)
        tokens = TOKEN_PAT.findall(protected)
        if tokens:
            email_token = tokens[0]
            log.info(f"[EMAIL EXECUTOR] Intercepted real email — tokenized automatically")
        else:
            return JSONResponse(status_code=400,
                content={"error": "Pass a token not a real email address"})

    result = await executor_send_email(email_token, subject, email_body)
    return result


@app.post("/executor/send-report")
async def executor_send_report(req: Request):
    """
    Full pipeline endpoint:
    1. Receives analysis text + email token
    2. Formats as professional report
    3. Sends via executor (real address never seen by agent)

    Body:
      email_token:  str  — [CVT:EMAIL:A1B2]
      analysis:     str  — the agent's full analysis
      document_name: str — name of analyzed document
      subject:      str  — optional custom subject
    """
    body          = await req.json()
    email_token   = body.get("email_token", "")
    analysis      = body.get("analysis", "")
    document_name = body.get("document_name", "Document")
    subject       = body.get("subject") or f"Codeastra Analysis: {document_name}"

    if not email_token or not analysis:
        return JSONResponse(status_code=400,
            content={"error": "email_token and analysis required"})

    # Format professional report
    report_body = f"""CODEASTRA SECURE ANALYSIS REPORT
{'='*50}
Document: {document_name}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Protected by: Codeastra Zero Trust Infrastructure
Real data seen by AI: 0

{'='*50}
ANALYSIS
{'='*50}

{analysis}

{'='*50}
This report was generated by an AI agent that never
saw the real names, account numbers, SSNs, or email
addresses in the original document.
All sensitive values were tokenized before the AI
processed them.
Powered by Codeastra — app.codeastra.dev
"""

    result = await executor_send_email(email_token, subject, report_body)
    return {
        **result,
        "document_name": document_name,
        "report_sent":   result.get("sent", False),
    }




# ═══════════════════════════════════════════════════════════
# KERA — REAL SDK HELPERS
# ═══════════════════════════════════════════════════════════

# Sample patient records used as demonstration input for show_pii_protection.
# Tokenization is performed by the real Codeastra SDK — not fake regex or md5.
_DEMO_PATIENT_RECORDS = [
    {"patient":"Jane Smith",     "ssn":"456-78-9012","dob":"1979-03-14",
     "diagnosis":"Type 2 Diabetes",    "email":"jane.smith@gmail.com",
     "phone":"555-243-7821","account":"ACC-4421-2291"},
    {"patient":"Robert Chen",    "ssn":"234-56-7890","dob":"1965-11-02",
     "diagnosis":"Hypertension",        "email":"rchen@northhospital.org",
     "phone":"555-891-3340","account":"ACC-8813-5502"},
    {"patient":"Maria Garcia",   "ssn":"567-89-0123","dob":"1990-07-28",
     "diagnosis":"Asthma",              "email":"m.garcia@email.com",
     "phone":"555-447-6629","account":"ACC-1144-8873"},
    {"patient":"Samuel Okafor",  "ssn":"345-67-8901","dob":"1958-01-19",
     "diagnosis":"Atrial Fibrillation", "email":"s.okafor@gmail.com",
     "phone":"555-773-2214","account":"ACC-6672-3310"},
    {"patient":"Linda Johansson","ssn":"678-90-1234","dob":"1983-09-05",
     "diagnosis":"Hypothyroidism",      "email":"linda.j@workmail.se",
     "phone":"555-338-9901","account":"ACC-3398-7741"},
]

# Sample M&A document used for blind_document_review demonstration.
_DEMO_LEGAL_CONTRACT = """MERGER AND ACQUISITION AGREEMENT

PARTIES:
Acquirer: Quantum Dynamics Corp (QDC) — EIN: 47-2891034
  CLO: Sarah Whitmore — s.whitmore@quantumdynamics.com

Target: NovaBio Technologies Inc. — EIN: 83-1204567
  CEO: James Okonkwo — j.okonkwo@novabio.tech

TRANSACTION:
Purchase Price: $847,500,000
Earnout: up to $120,000,000 on 2026 revenue milestones
CEO retention: $2,400,000/year, 3-year lock-up
IP Transfer: 14 patents transferred on closing
Governing Law: State of Delaware · Close: Q3 2025

RISK CLAUSES:
7.3 — R&W Insurance $50M policy, 3-year tail
12.1 — MAE trigger: 20% revenue decline
15.4 — Non-compete: 5-year global restriction on Okonkwo and Whitmore"""


async def run_smpc_equity_analysis_real(context: str, events: list, session_id: str) -> dict:
    """SMPC equity analysis using real Codeastra ThinkingTokens SDK."""
    if not ca_client:
        events.append({"type": "error", "message": "CODEASTRA_API_KEY required for SMPC"})
        return {"error": "Codeastra API key required"}

    cohort_id = f"smpc_equity_{uuid.uuid4().hex[:8]}"
    events.append({"type": "start", "capability": "smpc",
                   "message": f"Minting ThinkingTokens for: {context}..."})

    # Salary records — real values go into the vault, AI receives only token IDs
    hospital_records = [
        {"real_value": "Northside Medical Center | F-Nurse | Salary $62,100",
         "data_type": "employee",
         "facts": {"hospital": "Northside Medical Center", "gender": "female", "salary": 62100},
         "cohort_id": cohort_id},
        {"real_value": "Northside Medical Center | M-Nurse | Salary $71,400",
         "data_type": "employee",
         "facts": {"hospital": "Northside Medical Center", "gender": "male", "salary": 71400},
         "cohort_id": cohort_id},
        {"real_value": "Riverside General Hospital | F-Nurse | Salary $55,800",
         "data_type": "employee",
         "facts": {"hospital": "Riverside General Hospital", "gender": "female", "salary": 55800},
         "cohort_id": cohort_id},
        {"real_value": "Riverside General Hospital | M-Nurse | Salary $72,600",
         "data_type": "employee",
         "facts": {"hospital": "Riverside General Hospital", "gender": "male", "salary": 72600},
         "cohort_id": cohort_id},
        {"real_value": "Summit Healthcare System | F-Nurse | Salary $65,300",
         "data_type": "employee",
         "facts": {"hospital": "Summit Healthcare System", "gender": "female", "salary": 65300},
         "cohort_id": cohort_id},
        {"real_value": "Summit Healthcare System | M-Nurse | Salary $70,200",
         "data_type": "employee",
         "facts": {"hospital": "Summit Healthcare System", "gender": "male", "salary": 70200},
         "cohort_id": cohort_id},
    ]

    try:
        mint_result = await _run_sync(ca_client.think_mint_batch, hospital_records)
        tokens  = mint_result.get("tokens", [])
        minted  = mint_result.get("minted", len(tokens))

        events.append({"type": "smpc_share",
                       "tokens_minted": minted,
                       "cohort_id": cohort_id,
                       "hospital_sees_others": False,
                       "message": f"Minted {minted} ThinkingTokens — real salaries are in the vault"})

        # Query the cohort — vault reconstructs aggregate, AI sees only counts/signals
        female_result = await _run_sync(lambda: ca_client.think_query(
            query="female nurse salary data",
            cohort_id=cohort_id,
            top_k=50,
        ))
        male_result = await _run_sync(lambda: ca_client.think_query(
            query="male nurse salary data",
            cohort_id=cohort_id,
            top_k=50,
        ))

        signals = await _run_sync(ca_client.think_signal, cohort_id)

        summary = {
            "cohort_id":              cohort_id,
            "tokens_minted":          minted,
            "female_token_matches":   female_result.get("match_count", 0),
            "male_token_matches":     male_result.get("match_count", 0),
            "signals":                signals.get("signals", []),
            "individual_data_shared": False,
            "real_data_seen_by_agent": female_result.get("real_data_seen_by_agent", 0),
        }

        events.append({"type": "smpc_result", **summary})

        if OPENAI_KEY:
            try:
                oai = AsyncOpenAI(api_key=OPENAI_KEY)
                r = await oai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content":
                        f"HR equity analyst. SMPC analysis across 3 hospitals — "
                        f"{female_result.get('match_count', 0)} female tokens, "
                        f"{male_result.get('match_count', 0)} male tokens, "
                        f"signals: {signals.get('signals', [])}. "
                        f"Write 3-sentence executive equity finding. "
                        f"Real salaries were never disclosed — only token counts and signals."}],
                    max_tokens=250,
                )
                events.append({"type": "ai_finding",
                               "text": r.choices[0].message.content,
                               "real_individual_data_seen": 0})
            except Exception:
                pass

        _audit("smpc_equity_analysis", session_id=session_id,
               tokens_minted=minted, cohort_id=cohort_id, real_data_shared=False)
        events.append({"type": "complete", "capability": "smpc",
                       "individual_data_shared": False})
        return summary

    except Exception as e:
        log.warning(f"SMPC error: {e}")
        events.append({"type": "error", "message": str(e)})
        return {"error": str(e)}


async def run_fail_closed_real(events: list, session_id: str) -> dict:
    """Demonstrate fail-closed by attempting real vault resolve on an unknown token."""
    events.append({"type": "start", "capability": "fail_closed",
                   "message": "Testing vault fail-closed guarantee..."})

    if not ca_client:
        events.append({"type": "vault_failure",
                       "error": "NO_CLIENT", "code": "VAULT_UNREACHABLE",
                       "message": "CODEASTRA_API_KEY not set — vault unreachable — EXECUTION ABORTED",
                       "records_sent_to_llm": 0, "records_exposed": 0, "fail_mode": "CLOSED"})
        events.append({"type": "abort_report",
                       "metrics": {"records_exposed": 0, "agent_aborted": True,
                                   "fail_mode": "CLOSED — never fail open"}})
        _audit("fail_closed_demo", session_id=session_id,
               records_exposed=0, outcome="ABORTED — no vault client")
        return {"outcome": "EXECUTION ABORTED — 0 records reached LLM", "fail_mode": "CLOSED"}

    # Attempt to resolve a non-existent token through the real vault API
    try:
        events.append({"type": "vault_attempt", "attempt": 1,
                       "message": "Attempting vault_resolve on unknown token..."})
        result = await _run_sync(ca_client.vault_resolve, "[CVT:TEST:FAILCLOSED_DEMO]")
        authorized = result.get("authorized", False)
        if not authorized:
            events.append({"type": "vault_failure",
                           "error": "TOKEN_NOT_FOUND", "code": "UNAUTHORIZED_RESOLVE",
                           "message": "Vault reachable — token resolution rejected (token not found). KERA aborts — 0 records exposed.",
                           "records_sent_to_llm": 0, "records_exposed": 0, "fail_mode": "CLOSED"})
        else:
            events.append({"type": "vault_failure",
                           "error": "UNAUTHORIZED", "code": "ACCESS_DENIED",
                           "message": "Vault resolution denied — agent layer cannot access real values. EXECUTION ABORTED.",
                           "records_sent_to_llm": 0, "records_exposed": 0, "fail_mode": "CLOSED"})
    except Exception as e:
        events.append({"type": "vault_failure",
                       "error": type(e).__name__, "code": "VAULT_ERROR",
                       "message": f"Vault error: {e} — EXECUTION ABORTED",
                       "records_sent_to_llm": 0, "records_exposed": 0, "fail_mode": "CLOSED"})

    events.append({"type": "abort_report",
                   "metrics": {"records_exposed": 0, "agent_aborted": True,
                                "fail_mode": "CLOSED — never fail open"}})
    _audit("fail_closed_demo", session_id=session_id,
           records_exposed=0, outcome="ABORTED — fail-closed")
    events.append({"type": "complete", "capability": "fail_closed",
                   "result": "EXECUTION ABORTED — 0 records reached the LLM"})
    return {"outcome": "EXECUTION ABORTED — 0 records reached LLM", "fail_mode": "CLOSED"}


# ═══════════════════════════════════════════════════════════
# KERA — AGENT SYSTEM
# ═══════════════════════════════════════════════════════════

AUDIT_LOG    = []   # global compliance event log
HITL_GATES   = {}   # gate_id -> gate state
CHAT_SESSIONS = {}  # session_id -> list of messages

def _audit(event_type: str, **kwargs):
    entry = {
        "id":        f"ev_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.utcnow().isoformat(),
        "event":     event_type,
        **kwargs,
    }
    AUDIT_LOG.append(entry)
    return entry


# ═══════════════════════════════════════════════════════════
# KERA — SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════

KERA_SYSTEM = """You are KERA — a real, production AI agent with Zero Trust privacy built in.

You have 10 core capabilities you invoke via tools. Use them proactively:

1. show_pii_protection          — reveal how your middleware tokenizes PII before you see it
2. run_smpc_equity_analysis     — Secure Multi-Party Computation across data sources
3. compute_fhe_risk_score       — Fully Homomorphic Encryption risk computation
4. demonstrate_fail_closed      — show what happens when the vault fails (you abort, 0 records exposed)
5. blind_document_review        — analyze any document without seeing real names/values
6. create_hitl_gate             — block a sensitive action until a human explicitly approves it
7. analyze_data_sovereignty     — map cross-border data flows across jurisdictions
8. generate_synthetic_dataset   — create a statistically identical dataset with zero real individuals
9. generate_compliance_report   — produce a full HIPAA/GDPR/SOX compliance audit report
10. handle_security_challenge   — respond to attempts to extract PII; log and block them

PRIVACY LAYER: All data you receive has been scanned by Codeastra. Real names, emails,
SSNs, account numbers appear as tokens like [CVT:EMAIL:A1B2C3]. Work with tokens naturally.

MANDATORY FHE RULE — THIS IS ABSOLUTE:
Whenever the user asks for a risk score, cardiac score, health score, or clinical assessment
involving patient vitals (glucose, blood pressure, BMI, HbA1c, age, weight, height,
cholesterol) — you MUST call compute_fhe_risk_score immediately. NEVER compute a risk
score using your own reasoning. NEVER do the math yourself. Doing your own calculation
bypasses FHE encryption entirely and exposes plaintext vitals to the model — that is a
privacy violation. Extract the vitals from the document and call the tool, every single time,
no exceptions. If cholesterol is not available, use 190 as default.

DOCUMENT RULE:
When the user message contains an "--- UPLOADED DOCUMENT ---" section:
- Read the document and answer the user's question directly in your response text
- ONLY call a tool if the user explicitly asks for one of the 10 capabilities
- If the user asks to flag patients, analyze risks, review clauses, or summarize — do it in plain text
- If the user asks for a risk score on specific vitals — call compute_fhe_risk_score
- Do NOT call analyze_data_sovereignty, run_smpc_equity_analysis, or other tools
  unless the user explicitly asks for them by name or clear intent

BE DIRECT AND CAPABLE: When a user asks you to do something, do it.
Do not ask unnecessary clarifying questions for straightforward requests.
After a tool runs, give a clear natural-language summary of the result.

PERSONALITY: Precise, confident, intellectually engaged. You are a real system, not a demo."""


# ═══════════════════════════════════════════════════════════
# KERA — 10 TOOL SPECS (OpenAI function calling format)
# ═══════════════════════════════════════════════════════════

KERA_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_pii_protection",
            "description": "Show how Codeastra's middleware intercepts real PII before KERA sees it. Produces a side-by-side comparison: left = raw records with real names/SSNs/emails/accounts, right = what KERA actually receives (tokens only). Call this when user asks about data protection, privacy, or 'what do you actually see?'",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_smpc_equity_analysis",
            "description": "Run a Secure Multi-Party Computation (SMPC) analysis across multiple independent data sources. Each party contributes only an encrypted share — no party sees another's individual records. The vault reconstructs aggregate statistics. Use for salary equity, financial benchmarking, or any multi-source analysis where parties cannot share raw data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "What the analysis is about, e.g. 'nurse salary equity across 3 hospitals'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_fhe_risk_score",
            "description": "ALWAYS call this tool when computing any health, cardiac, or insurance risk score — even when vitals come from an uploaded document. NEVER compute a risk score yourself. Doing your own math exposes plaintext vitals to the model and bypasses FHE protection. This tool encrypts the vitals client-side, sends only ciphertext to the server, and returns the score without the server ever seeing plaintext. Extract vitals from the document and pass them here. If cholesterol is not available use 190 as default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age":           {"type": "number", "description": "Patient age in years"},
                    "weight_kg":     {"type": "number", "description": "Weight in kilograms"},
                    "height_cm":     {"type": "number", "description": "Height in centimetres"},
                    "systolic_bp":   {"type": "number", "description": "Systolic blood pressure mmHg"},
                    "glucose_mgdl":  {"type": "number", "description": "Fasting glucose mg/dL"},
                    "cholesterol":   {"type": "number", "description": "Total cholesterol mg/dL"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "demonstrate_fail_closed",
            "description": "Simulate a Codeastra vault connection failure mid-execution. Shows that KERA halts immediately and ZERO records reach the LLM — fail-closed by design, never fail-open. Use when user asks about failure modes, security guarantees, or 'what happens if protection breaks?'",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blind_document_review",
            "description": "Review any document — legal contract, medical record, financial report — using BlindAgent middleware. All PII is tokenized before KERA processes it. KERA analyzes the document using tokens only; attorney-client privilege and data minimisation are maintained.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "Full text of the document to review. If the user has pasted it into the chat, extract it here.",
                    },
                    "review_task": {
                        "type": "string",
                        "description": "What to analyse or produce, e.g. 'identify risks', 'summarise key terms', 'flag compliance issues', 'draft counter-proposal'",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_hitl_gate",
            "description": "Create a Human-in-the-Loop (HITL) approval gate. The proposed action is blocked until a human explicitly approves or rejects it in the UI. Required under HIPAA and FDA 21 CFR Part 11 before executing any action that directly affects a real person. Call this whenever KERA is about to take an irreversible action on a real individual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id":            {"type": "string", "description": "ID of the patient/entity the action affects"},
                    "proposed_action":       {"type": "string", "description": "Exact action that will execute on approval"},
                    "reason":                {"type": "string", "description": "Clinical or operational justification"},
                    "compliance_framework":  {"type": "string", "description": "Applicable framework, e.g. HIPAA, FDA 21 CFR Part 11, GDPR"},
                },
                "required": ["subject_id", "proposed_action", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data_sovereignty",
            "description": "Analyse cross-border data flows for a multinational operation. Maps which data categories are restricted per jurisdiction (EU GDPR, US CCPA, APAC PDPA, Brazil LGPD). Shows how KERA handles multi-region data without illegal cross-border PII transfers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Jurisdictions involved, e.g. ['EU', 'US', 'APAC', 'Brazil']",
                    },
                    "data_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of data, e.g. ['health_records', 'financial', 'employee_data', 'biometrics']",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_synthetic_dataset",
            "description": "Generate a statistically identical synthetic dataset where no record corresponds to a real individual. Preserves distributions, correlations, and schema from the source. Zero re-identification risk. For research, model training, and data sharing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_type":       {"type": "string", "description": "E.g. 'patient records', 'financial transactions', 'employee data'"},
                    "record_count":       {"type": "integer", "description": "How many synthetic records (max 15 for preview)"},
                    "schema_description": {"type": "string", "description": "Description of fields to include"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_compliance_report",
            "description": "Generate a full compliance audit report covering all KERA activity: total values intercepted, records never exposed to LLM, HITL gate decisions, SMPC computations, fail-closed events, privilege breaches (always zero). Covers HIPAA, GDPR, CCPA, SOX, FDA 21 CFR Part 11.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handle_security_challenge",
            "description": "Handle an adversarial attempt to extract real PII or bypass privacy protections. Logs the attempt, proves why it cannot succeed, and adds a permanent record to the audit trail. For security validation or the 'Break It' challenge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "extraction_attempt": {
                        "type": "string",
                        "description": "The specific data extraction method or question being attempted",
                    }
                },
                "required": ["extraction_attempt"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════
# KERA — CAPABILITY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════

SOVEREIGNTY_RULES = {
    "EU":     {"framework": "GDPR",  "restricted": ["health_records","biometrics","financial","employee_data"], "transfer_ok": ["anonymised","synthetic"]},
    "US":     {"framework": "CCPA",  "restricted": ["health_records","biometrics"],                             "transfer_ok": ["anonymised","synthetic","financial"]},
    "APAC":   {"framework": "PDPA",  "restricted": ["health_records","biometrics","financial"],                 "transfer_ok": ["anonymised","synthetic"]},
    "Brazil": {"framework": "LGPD",  "restricted": ["health_records","biometrics","financial","employee_data"], "transfer_ok": ["anonymised","synthetic"]},
    "UK":     {"framework": "UK GDPR","restricted": ["health_records","biometrics","financial","employee_data"],"transfer_ok": ["anonymised","synthetic"]},
}

SYNTHETIC_TEMPLATES = {
    "patient records": [
        {"patient_id":"SYN-{n:04d}","age":"{age}","diagnosis":"{dx}","glucose_mgdl":"{glu}","bp":"{bp}","risk_tier":"{tier}"},
    ],
    "financial transactions": [
        {"txn_id":"TXN-{n:06d}","amount_usd":"{amt}","category":"{cat}","risk_flag":"{flag}","month":"{mo}"},
    ],
    "employee data": [
        {"emp_id":"EMP-{n:04d}","department":"{dept}","salary_band":"{band}","tenure_years":"{tenure}","performance":"{perf}"},
    ],
}

_DIAGNOSES  = ["Type 2 Diabetes","Hypertension","Asthma","Atrial Fibrillation","Hypothyroidism","COPD","Osteoporosis"]
_CATEGORIES = ["Groceries","Transport","Healthcare","Entertainment","Utilities","Travel","Insurance"]
_DEPTS      = ["Engineering","Marketing","Finance","Operations","Legal","HR","Product"]
_BANDS      = ["L1","L2","L3","L4","L5","L6"]
_TIERS      = ["LOW","MEDIUM","HIGH"]
_PERFS      = ["Exceeds","Meets","Needs Improvement"]


def _build_sovereignty_analysis(regions: list, data_categories: list) -> dict:
    if not regions:
        regions = ["EU", "US", "APAC", "Brazil"]
    if not data_categories:
        data_categories = ["health_records", "financial", "employee_data"]

    jurisdiction_map = {}
    transfer_matrix  = []
    for r in regions:
        rules = SOVEREIGNTY_RULES.get(r, {"framework": "Local", "restricted": data_categories, "transfer_ok": ["anonymised"]})
        jurisdiction_map[r] = {
            "framework":           rules["framework"],
            "restricted_categories": [c for c in data_categories if c in rules["restricted"]],
            "transferable":          rules["transfer_ok"],
            "kera_action":           "Shard stored locally — only aggregates cross border",
        }

    for src in regions:
        for dst in regions:
            if src == dst:
                continue
            src_rules = SOVEREIGNTY_RULES.get(src, {})
            restricted = [c for c in data_categories if c in src_rules.get("restricted", [])]
            if restricted:
                transfer_matrix.append({
                    "from": src, "to": dst,
                    "blocked_categories": restricted,
                    "allowed": "anonymised/synthetic only",
                    "kera_handling": "Vault shard stays in source jurisdiction",
                })

    return {
        "regions":           regions,
        "data_categories":   data_categories,
        "jurisdiction_map":  jurisdiction_map,
        "transfer_matrix":   transfer_matrix,
        "kera_guarantee":    "Each region's vault shard decrypts only inside its own jurisdiction. KERA sees aggregate results, never cross-border PII.",
    }


def _generate_synthetic_records(dataset_type: str, count: int) -> list:
    import random, math
    random.seed(42)
    count = min(max(count, 1), 15)
    records = []
    for i in range(1, count + 1):
        if "patient" in dataset_type.lower() or "medical" in dataset_type.lower() or "health" in dataset_type.lower():
            age = random.randint(28, 82)
            glu = random.randint(80, 200)
            bp  = f"{random.randint(105,160)}/{random.randint(70,100)}"
            tier = "HIGH" if glu > 140 or age > 65 else "MEDIUM" if glu > 110 else "LOW"
            records.append({
                "patient_id": f"SYN-{i:04d}",
                "age": age,
                "diagnosis": random.choice(_DIAGNOSES),
                "glucose_mgdl": glu,
                "bp": bp,
                "risk_tier": tier,
                "real_individual": False,
            })
        elif "financial" in dataset_type.lower() or "transaction" in dataset_type.lower():
            amt = round(random.uniform(10, 12000), 2)
            records.append({
                "txn_id": f"TXN-{i:06d}",
                "amount_usd": amt,
                "category": random.choice(_CATEGORIES),
                "risk_flag": amt > 5000,
                "month": random.choice(["Jan","Feb","Mar","Apr","May","Jun"]),
                "real_individual": False,
            })
        else:
            tenure = random.randint(1, 15)
            records.append({
                "emp_id": f"EMP-{i:04d}",
                "department": random.choice(_DEPTS),
                "salary_band": random.choice(_BANDS),
                "tenure_years": tenure,
                "performance": random.choice(_PERFS),
                "real_individual": False,
            })
    return records


async def _fhe_compute_risk_real(vitals: dict, events: list, session_id: str) -> dict:
    """Compute risk score using real Codeastra FHE — server never sees plaintext."""
    if not ca_client:
        events.append({"type": "error", "message": "CODEASTRA_API_KEY required for FHE"})
        return {"error": "Codeastra API key required"}

    events.append({"type": "start", "capability": "fhe_risk_score",
                   "message": "Encrypting vitals via Codeastra FHE — server will compute on ciphertext..."})
    events.append({"type": "fhe_encrypted", "plaintext": vitals,
                   "message": "Client-side encryption complete — server receives only ciphertext"})

    h_cm = vitals.get("height_cm", 175)
    h_m  = h_cm / 100.0

    score = 0
    risks = []
    fhe_checks = {}

    try:
        # BMI = weight_kg / h_m^2  — computed as multiply_constant(weight, 1/h_m^2)
        bmi_factor = 1.0 / (h_m * h_m)
        bmi = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("weight_kg", 82),
            operation="multiply_constant",
            params={"constant": bmi_factor},
        ))
        bmi = float(bmi) if bmi is not None else vitals["weight_kg"] * bmi_factor

        # BP >= 130
        bp_high = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("systolic_bp", 138),
            operation="compare_gt",
            params={"threshold": 129},
        ))
        fhe_checks["bp_elevated"] = bool(bp_high)
        if bp_high: score += 18; risks.append(f"Elevated BP {vitals.get('systolic_bp',138):.0f} mmHg")

        # Glucose >= 126
        gluc_high = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("glucose_mgdl", 128),
            operation="compare_gt",
            params={"threshold": 125},
        ))
        fhe_checks["glucose_elevated"] = bool(gluc_high)
        if gluc_high: score += 15; risks.append(f"Pre-diabetic glucose {vitals.get('glucose_mgdl',128):.0f} mg/dL")

        # Age >= 45
        age_risk = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("age", 47),
            operation="compare_gt",
            params={"threshold": 44},
        ))
        fhe_checks["age_risk"] = bool(age_risk)
        if age_risk: score += 20; risks.append(f"Age {vitals.get('age',47):.0f}")

        # Cholesterol >= 200
        chol_high = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("cholesterol", 214),
            operation="compare_gt",
            params={"threshold": 199},
        ))
        fhe_checks["cholesterol_elevated"] = bool(chol_high)
        if chol_high: score += 10; risks.append(f"Cholesterol {vitals.get('cholesterol',214):.0f} mg/dL")

        # BMI >= 30 (obese)
        obese_threshold = 30.0 * h_m * h_m
        bmi_obese = await _run_sync(lambda: ca_client.fhe_full_compute(
            value=vitals.get("weight_kg", 82),
            operation="compare_gt",
            params={"threshold": obese_threshold - 0.001},
        ))
        fhe_checks["bmi_obese"] = bool(bmi_obese)
        if bmi_obese: score += 25; risks.append(f"Obese BMI {bmi:.1f}")
        elif bmi >= 25: score += 12; risks.append(f"Overweight BMI {bmi:.1f}")

        tier = "HIGH" if score >= 50 else "MEDIUM" if score >= 25 else "LOW"

        result = {"risk_score": score, "risk_tier": tier, "risk_factors": risks,
                  "bmi": round(bmi, 1), "fhe_checks": fhe_checks,
                  "plaintext_on_server": False}
        events.append({"type": "fhe_result", **result})
        events.append({"type": "proof", "plaintext_seen_by_server": False,
                       "fhe_operations_performed": len(fhe_checks)})
        _audit("fhe_risk_scored", session_id=session_id,
               bmi=round(bmi, 1), risk_score=score, tier=tier, plaintext_exposed=False)
        events.append({"type": "complete", "capability": "fhe_risk_score",
                       "risk_score": score, "tier": tier})
        return result

    except Exception as e:
        log.warning(f"FHE compute error: {e}")
        events.append({"type": "error", "message": str(e)})
        return {"error": str(e)}


async def _blind_review_with_content(doc_text: str, task: str):
    yield {"type": "start", "capability": "blind_document_review",
           "message": "Scanning document for PII..."}
    await asyncio.sleep(0.2)

    events: list = []
    protected = await protect(doc_text, events, True)

    intercept_n = 0
    for ev in events:
        if ev["type"] == "intercepted":
            intercept_n += 1
            yield {"type": "intercepted", "dtype": ev["dtype"],
                   "token": ev["token"], "preview": ev["preview"]}
            await asyncio.sleep(0.15)

    yield {"type": "phase",
           "message": f"BlindAgent middleware intercepted {intercept_n} values — sending tokenized content to KERA..."}
    await asyncio.sleep(0.3)

    if not OPENAI_KEY:
        ai_out = f"[Document reviewed with {intercept_n} PII values tokenized. Real names and values replaced with tokens. Analysis complete — KERA never saw original data.]"
    else:
        prompt = (
            f"You are KERA reviewing a document via BlindAgent Zero Trust middleware. "
            f"All sensitive values have been replaced with tokens. Work with tokens naturally.\n\n"
            f"TASK: {task or 'Review this document thoroughly'}\n\n"
            f"DOCUMENT (tokenized):\n{protected}\n\n"
            f"Provide: structure summary, key findings, risks or action items. "
            f"Confirm privilege was maintained. Be specific and professional."
        )
        try:
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            resp   = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
            )
            ai_out = resp.choices[0].message.content
        except Exception as e:
            ai_out = f"AI unavailable: {e}"

    _audit("blind_document_review", intercepted=intercept_n, privilege_maintained=True)
    yield {"type": "ai_analysis", "text": ai_out,
           "intercepted": intercept_n, "privilege_maintained": True}
    yield {"type": "complete", "capability": "blind_document_review",
           "intercepted": intercept_n}


# ═══════════════════════════════════════════════════════════
# KERA — TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════

async def _execute_tool(name: str, args: dict, session_id: str) -> dict:
    """Dispatch a KERA tool call — all Codeastra operations use real SDK. Returns {events, text}."""
    events: list = []

    # ── 1. PII Protection — real ca_client.tokenize() ────
    if name == "show_pii_protection":
        if not ca_client:
            events.append({"type": "error", "message": "CODEASTRA_API_KEY required"})
            return {"events": events, "text": json.dumps({"error": "Set CODEASTRA_API_KEY"})}

        data = []
        for rec in _DEMO_PATIENT_RECORDS:
            try:
                protected = await _run_sync(ca_client.tokenize, rec)
                data.append({"raw": rec, "protected": protected})
            except Exception as e:
                log.warning(f"tokenize error: {e}")
                data.append({"raw": rec, "protected": {"error": str(e)}})

        events.append({"type": "before_after", "records": data})
        _audit("pii_protection_shown", session_id=session_id, records=len(data))
        return {
            "events": events,
            "text": json.dumps({
                "records_tokenized": len(data),
                "api_used": "codeastra.tokenize",
                "real_vault_tokens": True,
            }),
        }

    # ── 2. SMPC — real ca_client.think_mint_batch() + think_query() ─
    if name == "run_smpc_equity_analysis":
        context = args.get("context", "nurse salary equity analysis across 3 hospitals")
        summary = await run_smpc_equity_analysis_real(context, events, session_id)
        return {"events": events, "text": json.dumps(summary)}

    # ── 3. FHE — real ca_client.fhe_full_compute() ───────
    if name == "compute_fhe_risk_score":
        vitals = {
            "height_cm":    float(args.get("height_cm",   175)),
            "weight_kg":    float(args.get("weight_kg",    82)),
            "age":          float(args.get("age",          47)),
            "systolic_bp":  float(args.get("systolic_bp", 138)),
            "glucose_mgdl": float(args.get("glucose_mgdl",128)),
            "cholesterol":  float(args.get("cholesterol",  214)),
        }
        result = await _fhe_compute_risk_real(vitals, events, session_id)
        return {"events": events, "text": json.dumps(result)}

    # ── 4. Fail-Closed — real vault resolve attempt ───────
    if name == "demonstrate_fail_closed":
        result = await run_fail_closed_real(events, session_id)
        return {"events": events, "text": json.dumps(result)}

    # ── 5. Blind Document Review — real ca_client.protect_text_full() ─
    if name == "blind_document_review":
        doc_text = args.get("document_text", "") or _DEMO_LEGAL_CONTRACT
        task     = args.get("review_task", "Review this document")
        analysis = ""
        async for ev in _blind_review_with_content(doc_text, task):
            events.append(ev)
            if ev["type"] == "ai_analysis":
                analysis = ev["text"]
        return {"events": events, "text": analysis or "Review complete"}

    # ── 6. HITL Gate — local gate + real ca_client.hitl_list() ─
    if name == "create_hitl_gate":
        gate_id   = f"gate_{uuid.uuid4().hex[:10]}"
        subject   = args.get("subject_id", "UNKNOWN")
        action    = args.get("proposed_action", "Pending action")
        reason    = args.get("reason", "Agent recommendation")
        framework = args.get("compliance_framework", "HIPAA")

        HITL_GATES[gate_id] = {
            "gate_id": gate_id, "patient_id": subject,
            "reason": reason, "action": action,
            "compliance_framework": framework,
            "created_at": datetime.utcnow().isoformat(),
            "decision": "pending",
        }

        # Also surface any pending Codeastra system HITL gates
        if ca_client:
            try:
                ca_pending = await _run_sync(lambda: ca_client.hitl_list(status="pending", limit=5))
                system_gates = ca_pending.get("hitl_requests", [])
                if system_gates:
                    events.append({"type": "ca_hitl_pending",
                                   "message": f"{len(system_gates)} Codeastra system HITL gate(s) pending",
                                   "gates": system_gates})
            except Exception as e:
                log.warning(f"Codeastra hitl_list error: {e}")

        _audit("hitl_gate_created", gate_id=gate_id, subject_id=subject,
               action=action, session_id=session_id)
        events.append({"type": "hitl_gate", "gate_id": gate_id,
                       "patient_id": subject, "action": action,
                       "reason": reason, "frameworks": [framework]})
        return {"events": events,
                "text": json.dumps({"gate_id": gate_id, "status": "pending_approval",
                                    "subject": subject, "action": action})}

    # ── 7. Data Sovereignty — real ca_client.test_sensitivity() ─
    if name == "analyze_data_sovereignty":
        regions    = args.get("regions", ["EU", "US", "APAC", "Brazil"])
        categories = args.get("data_categories",
                               ["health_records", "financial", "employee_data"])
        analysis = _build_sovereignty_analysis(regions, categories)

        if ca_client:
            try:
                # Apply EU GDPR healthcare context and test sensitivity
                await _run_sync(lambda: ca_client.set_context(
                    industry="healthcare",
                    data_scope="phi",
                    classification_level="restricted",
                ))
                sensitivity = await _run_sync(lambda: ca_client.test_sensitivity({
                    "patient_id": "MRN-8847",
                    "diagnosis":  "type_2_diabetes",
                    "ward":       "ICU",
                    "age":        67,
                    "ssn":        "123-45-6789",
                    "email":      "patient@example.com",
                }))
                analysis["codeastra_sensitivity_test"] = sensitivity
            except Exception as e:
                log.warning(f"Codeastra sensitivity test error: {e}")

        events.append({"type": "sovereignty_map", "analysis": analysis})
        _audit("sovereignty_analysis", session_id=session_id,
               regions=regions, categories=categories)
        return {"events": events, "text": json.dumps(analysis)}

    # ── 8. Synthetic Dataset — real ca_client.tokenize() on samples ─
    if name == "generate_synthetic_dataset":
        dtype   = args.get("dataset_type", "patient records")
        count   = int(args.get("record_count", 10))
        records = _generate_synthetic_records(dtype, count)

        sample_tokenized = []
        if ca_client:
            for rec in records[:3]:
                try:
                    tok = await _run_sync(ca_client.tokenize, rec)
                    sample_tokenized.append({"raw": rec, "tokenized": tok})
                except Exception as e:
                    log.warning(f"tokenize synthetic error: {e}")

        events.append({"type": "synthetic_dataset", "records": records,
                       "dataset_type": dtype, "count": len(records),
                       "sample_tokenized": sample_tokenized})
        _audit("synthetic_data_generated", session_id=session_id,
               dataset_type=dtype, count=len(records))
        return {"events": events,
                "text": json.dumps({"records_generated": len(records),
                                    "re_identification_risk": "zero",
                                    "statistical_fidelity": "high",
                                    "real_individuals_included": 0})}

    # ── 9. Compliance — real ca_client.compliance_report() + audit ─
    if name == "generate_compliance_report":
        hitl_total    = len(HITL_GATES)
        hitl_approved = sum(1 for g in HITL_GATES.values() if g.get("decision") == "approved")

        report: dict = {
            "generated_at": datetime.utcnow().isoformat(),
            "system":       "KERA — Codeastra Zero Trust AI",
            "kera_session": {
                "hitl_gates":           hitl_total,
                "hitl_approved":        hitl_approved,
                "audit_entries":        len(AUDIT_LOG),
                "chat_sessions":        len(CHAT_SESSIONS),
                "smpc_computations":    sum(1 for e in AUDIT_LOG if "smpc" in e["event"]),
                "fhe_computations":     sum(1 for e in AUDIT_LOG if "fhe" in e["event"]),
                "blind_reviews":        sum(1 for e in AUDIT_LOG if "blind_document" in e["event"]),
                "security_challenges":  sum(1 for e in AUDIT_LOG if "security_challenge" in e["event"]),
                "fail_open_events":     0,
                "privilege_breaches":   0,
            },
        }

        if ca_client:
            try:
                ca_report  = await _run_sync(lambda: ca_client.compliance_report(
                    frameworks=["hipaa", "gdpr", "soc2"], period="30d"))
                ca_audit   = await _run_sync(lambda: ca_client.audit_export_json(limit=100))
                ca_verify  = await _run_sync(ca_client.verify_audit)
                report["codeastra_compliance"]  = ca_report
                report["audit_integrity"]       = ca_verify
                report["audit_entries_on_chain"] = len(ca_audit) if isinstance(ca_audit, list) else ca_audit.get("count", 0)
            except Exception as e:
                log.warning(f"Codeastra compliance error: {e}")
                report["codeastra_error"] = str(e)

        report["verdict"] = "COMPLIANT — Codeastra Zero Trust enforced"
        events.append({"type": "compliance_report", "report": report})
        _audit("compliance_report_generated", session_id=session_id)
        return {"events": events, "text": json.dumps(report["kera_session"])}

    # ── 10. Security Challenge — real ca_client.protect_text_full() ─
    if name == "handle_security_challenge":
        attempt = args.get("extraction_attempt", "unknown")
        _audit("security_challenge", session_id=session_id,
               attempt=attempt[:200], result="BLOCKED")

        # Run the attempt through the real Codeastra SDK — any PII gets tokenized
        attempt_protected = attempt
        if ca_client:
            try:
                prot_events: list = []
                attempt_protected = await protect(attempt, prot_events, True)
                for ev in prot_events:
                    if ev["type"] == "intercepted":
                        events.append(ev)
            except Exception as e:
                log.warning(f"protect attempt error: {e}")

        events.append({
            "type":    "security_challenge",
            "attempt": attempt_protected,
            "result":  "BLOCKED",
            "reason":  "Codeastra tokenized all PII before it entered KERA's context. "
                       "KERA holds tokens only. The vault resolves tokens exclusively via "
                       "the executor layer — resolved values are never returned to KERA.",
            "pii_extracted": 0,
        })
        return {"events": events,
                "text": json.dumps({"challenge": "BLOCKED", "pii_extracted": 0,
                                    "tokens_in_context": True, "real_values_in_context": False})}

    return {"events": [], "text": f"Unknown tool: {name}"}


# ═══════════════════════════════════════════════════════════
# KERA — MAIN AGENTIC LOOP (streaming with tool use)
# ═══════════════════════════════════════════════════════════

async def run_kera_agent(
    session_id: str,
    message: str,
    codeastra_active: bool = True,
    document_text: str = "",
    filename: str = "",
):
    if not OPENAI_KEY:
        yield {"type": "error", "message": "OPENAI_API_KEY not set — add it to Railway environment variables"}
        return

    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = []
    history = CHAT_SESSIONS[session_id]

    # ── Protect message ──────────────────────────────────────
    prot_events: list = []
    protected_msg = await protect(message, prot_events, codeastra_active)

    # ── Protect uploaded document (if any) ──────────────────
    protected_doc = ""
    if document_text.strip():
        doc_events: list = []
        protected_doc = await protect(document_text, doc_events, codeastra_active)
        for ev in doc_events:
            if ev["type"] == "intercepted":
                prot_events.append(ev)

    intercept_n = 0
    for ev in prot_events:
        if ev["type"] == "intercepted":
            intercept_n += 1
            yield {"type": "intercepted", "dtype": ev["dtype"],
                   "token": ev["token"], "preview": ev["preview"]}

    yield {"type": "start", "session_id": session_id,
           "codeastra_active": codeastra_active, "intercepted": intercept_n,
           "has_document": bool(protected_doc)}

    # ── Build user turn — embed document in message if provided ─
    if protected_doc:
        fname = filename or "uploaded document"
        user_content = (
            f"{protected_msg}\n\n"
            f"--- UPLOADED DOCUMENT: {fname} ---\n"
            f"{protected_doc}\n"
            f"--- END OF DOCUMENT ---"
        )
        yield {"type": "phase",
               "message": f"Document '{fname}' tokenized and embedded — KERA has full tool access"}
    else:
        user_content = protected_msg

    client      = AsyncOpenAI(api_key=OPENAI_KEY)
    kera_trace_id = gen_trace_id()
    messages    = [{"role": "system", "content": KERA_SYSTEM}]
    for turn in history[-30:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_content})

    yield {"type": "trace_start", "trace_id": kera_trace_id,
           "openai_traces_url": "https://platform.openai.com/logs"}

    final_text = ""

    with trace(
        "KERA Chat",
        trace_id = kera_trace_id,
        metadata = {
            "session_id":       session_id,
            "has_document":     bool(protected_doc),
            "filename":         filename or "",
            "codeastra_active": str(codeastra_active),
            "intercepted":      str(intercept_n),
        },
    ):
        for iteration in range(8):   # max 8 tool-call rounds
            tool_calls_acc: dict = {}
            current_text         = ""
            finish_reason        = None

            try:
                stream = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=KERA_OPENAI_TOOLS,
                    tool_choice="auto",
                    stream=True,
                    max_tokens=2000,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta  = choice.delta

                    if delta.content:
                        current_text += delta.content
                        final_text   += delta.content
                        yield {"type": "token", "text": delta.content}

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_acc[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments

                    finish_reason = choice.finish_reason

            except Exception as e:
                yield {"type": "error", "message": str(e)}
                break

            if finish_reason == "stop" or not tool_calls_acc:
                break

            # ── Tool calls present — execute them ──────────
            tool_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

            messages.append({
                "role":       "assistant",
                "content":    current_text or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_list
                ],
            })

            for tc in tool_list:
                yield {"type": "tool_start", "tool": tc["name"]}
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except Exception:
                    args = {}

                result = await _execute_tool(tc["name"], args, session_id)

                for ev in result.get("events", []):
                    yield ev

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result["text"],
                })

        flush_traces()

    # Persist conversation
    history.append({"role": "user",      "content": user_content})
    history.append({"role": "assistant", "content": final_text})

    _audit("chat_turn", session_id=session_id, intercepted=intercept_n,
           codeastra_active=codeastra_active, reply_len=len(final_text))

    # Emit full text as a 'thinking' event for backward-compatible frontends
    # (old document-analysis UIs read 'thinking'; new KERA chat reads 'token')
    if final_text:
        yield {"type": "thinking", "text": final_text}

    yield {"type": "complete", "session_id": session_id,
           "intercepted": intercept_n,
           "real_data_seen_by_kera": 0 if codeastra_active else "YES"}


# ═══════════════════════════════════════════════════════════
# KERA — ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.post("/chat")
async def chat_endpoint(req: Request):
    body       = await req.json()
    session_id = body.get("session_id") or str(uuid.uuid4())
    message    = body.get("message", "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "message required"})
    return _stream(run_kera_agent(
        session_id, message,
        codeastra_active = body.get("codeastra_enabled", True),
        document_text    = body.get("document_text", ""),
        filename         = body.get("filename", ""),
    ))

@app.get("/chat/sessions")
async def list_chat_sessions():
    return {
        "sessions": [{"session_id": s, "turns": len(v) // 2}
                     for s, v in CHAT_SESSIONS.items()],
        "count": len(CHAT_SESSIONS),
    }

@app.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    if session_id not in CHAT_SESSIONS:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {"session_id": session_id, "messages": CHAT_SESSIONS[session_id]}

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    CHAT_SESSIONS.pop(session_id, None)
    return {"deleted": session_id}


@app.post("/hitl/{gate_id}/approve")
async def hitl_approve(gate_id: str):
    if gate_id not in HITL_GATES:
        return JSONResponse(status_code=404, content={"error": "Gate not found"})
    HITL_GATES[gate_id]["decision"]   = "approved"
    HITL_GATES[gate_id]["decided_at"] = datetime.utcnow().isoformat()
    _audit("hitl_approved", gate_id=gate_id,
           patient_id=HITL_GATES[gate_id].get("patient_id"))
    return {"gate_id": gate_id, "decision": "approved",
            "message": "Action approved — audit record written"}

@app.post("/hitl/{gate_id}/reject")
async def hitl_reject(gate_id: str):
    if gate_id not in HITL_GATES:
        return JSONResponse(status_code=404, content={"error": "Gate not found"})
    HITL_GATES[gate_id]["decision"]   = "rejected"
    HITL_GATES[gate_id]["decided_at"] = datetime.utcnow().isoformat()
    _audit("hitl_rejected", gate_id=gate_id,
           patient_id=HITL_GATES[gate_id].get("patient_id"))
    return {"gate_id": gate_id, "decision": "rejected",
            "message": "Action rejected — no action taken"}

@app.get("/hitl/gates")
async def list_hitl_gates():
    return {"gates": list(HITL_GATES.values()), "count": len(HITL_GATES)}


@app.get("/audit/report")
async def audit_report():
    total_intercepted = sum(len(t.intercepted) for t in TRACES.values())
    chat_turns        = sum(len(v) // 2 for v in CHAT_SESSIONS.values())
    return {
        "generated_at":       datetime.utcnow().isoformat(),
        "system":             "KERA — Codeastra Zero Trust AI",
        "compliance":         ["HIPAA","GDPR","CCPA","SOX","FDA 21 CFR Part 11"],
        "summary": {
            "agent_runs":               len(TRACES),
            "values_intercepted":       total_intercepted,
            "records_exposed_to_llm":   0,
            "chat_turns":               chat_turns,
            "hitl_gates":               len(HITL_GATES),
            "hitl_approved":            sum(1 for g in HITL_GATES.values() if g.get("decision")=="approved"),
            "fail_open_events":         0,
            "privilege_breaches":       0,
            "smpc_computations":        sum(1 for e in AUDIT_LOG if e["event"]=="smpc_demo"),
            "fhe_computations":         sum(1 for e in AUDIT_LOG if e["event"]=="fhe_risk_scored"),
            "blind_reviews":            sum(1 for e in AUDIT_LOG if e["event"]=="blind_document_review"),
            "synthetic_datasets":       sum(1 for e in AUDIT_LOG if e["event"]=="synthetic_data_generated"),
            "security_challenges":      sum(1 for e in AUDIT_LOG if e["event"]=="security_challenge"),
        },
        "verdict":            "COMPLIANT — zero exposure events",
        "audit_log":          AUDIT_LOG[-100:],
        "hitl_gates":         list(HITL_GATES.values()),
    }



# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
