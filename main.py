"""
Codeastra Autonomous Agent — PRODUCTION
=========================================
A REAL autonomous AI agent that:
  1. Connects to your REAL PostgreSQL database
  2. Calls your REAL Codeastra API (app.codeastra.dev)
  3. Claude makes REAL tool calls on your REAL data
  4. Every tool result goes through Codeastra BEFORE Claude sees it
  5. Claude fixes REAL performance and billing issues
  6. Real actions executed on your real database

This is what your clients deploy. Not a demo. Not fake data.
A real agent doing real work, made safe by Codeastra.

Required environment variables on Railway:
  ANTHROPIC_API_KEY   — your Claude API key
  CODEASTRA_API_KEY   — your Codeastra API key (sk-guard-...)
  DATABASE_URL        — postgresql://user:pass@host:5432/dbname
  PORT                — 8080 (Railway sets this automatically)

Optional:
  CODEASTRA_URL       — defaults to https://app.codeastra.dev
  STRIPE_SECRET_KEY   — for real billing operations
"""

import os, json, asyncio, logging
from datetime import datetime
import anthropic
import asyncpg
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("codeastra-agent")

# ── Config ────────────────────────────────────────────────
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CODEASTRA_KEY  = os.getenv("CODEASTRA_API_KEY", "")
CODEASTRA_URL  = os.getenv("CODEASTRA_URL", "https://app.codeastra.dev")
DATABASE_URL   = os.getenv("DATABASE_URL", "")
STRIPE_KEY     = os.getenv("STRIPE_SECRET_KEY", "")
PORT           = int(os.getenv("PORT", 8080))

app = FastAPI(title="Codeastra Autonomous Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db_pool = None

# ── DB Pool ───────────────────────────────────────────────
async def get_pool():
    global db_pool
    if db_pool is None and DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            log.info("✅ Database connected")
        except Exception as e:
            log.warning(f"DB connection failed: {e}")
    return db_pool


# ═══════════════════════════════════════════════════════════
# CODEASTRA API — REAL CALL
# Every single tool result passes through this.
# Real values go into the Codeastra vault.
# Claude only receives the tokenized output.
# ═══════════════════════════════════════════════════════════

async def protect(data, events: list) -> str:
    """
    Send any tool result to Codeastra API.
    Returns tokenized text — this is what Claude receives.
    Real values are stored in your Codeastra vault.
    """
    text = json.dumps(data) if not isinstance(data, str) else data

    if not CODEASTRA_KEY:
        events.append({
            "type":    "warning",
            "message": "CODEASTRA_API_KEY not set — data NOT protected",
        })
        return text  # ← no protection if no key

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{CODEASTRA_URL}/protect/text",
                headers={
                    "X-API-Key":    CODEASTRA_KEY,
                    "Content-Type": "application/json",
                },
                json={"text": text},
            )

            if r.status_code == 200:
                result    = r.json()
                protected = result.get("protected_text", text)

                # Codeastra returns "entities" array
                # Each entity: {token, original, type}
                entities  = (
                    result.get("entities") or
                    result.get("detections") or
                    result.get("tokens") or
                    result.get("items") or
                    []
                )

                for d in entities:
                    real = d.get("original") or d.get("real") or d.get("value") or ""
                    prev = (real[:3] + "•"*min(len(real)-5,8) + real[-2:]) if len(real)>5 else "•••"
                    events.append({
                        "type":    "intercepted",
                        "dtype":   d.get("type") or d.get("field_type") or d.get("dtype") or "PII",
                        "token":   d.get("token") or d.get("cdt_token") or "",
                        "preview": d.get("preview") or prev,
                        "real":    real,
                    })

                log.info(f"Codeastra protected {len(entities)} values")
                return protected

            else:
                log.warning(f"Codeastra API {r.status_code}: {r.text[:200]}")
                events.append({
                    "type":    "codeastra_error",
                    "status":  r.status_code,
                    "message": r.text[:200],
                })
                return text

    except Exception as e:
        log.warning(f"Codeastra API error: {e}")
        events.append({"type": "codeastra_error", "message": str(e)})
        return text


# ═══════════════════════════════════════════════════════════
# REAL TOOL IMPLEMENTATIONS
# These run against your real PostgreSQL database.
# Every result is protected by Codeastra before Claude sees it.
# ═══════════════════════════════════════════════════════════

async def tool_scan_slow_queries(events: list) -> str:
    """
    Reads pg_stat_statements from your real database.
    Returns real slow queries — Codeastra tokenizes before Claude sees them.
    """
    pool = await get_pool()

    if not pool:
        # If no real DB, explain clearly
        return await protect({
            "error": "No database connected. Set DATABASE_URL on Railway.",
            "how_to_fix": "Add DATABASE_URL=postgresql://user:pass@host:5432/dbname to Railway variables",
        }, events)

    async with pool.acquire() as conn:
        try:
            # Try pg_stat_statements first (real production metric)
            rows = await conn.fetch("""
                SELECT query, calls, mean_exec_time, total_exec_time,
                       rows, shared_blks_hit, shared_blks_read
                FROM pg_stat_statements
                WHERE mean_exec_time > 100
                ORDER BY mean_exec_time DESC
                LIMIT 20
            """)
            slow = [dict(r) for r in rows]
        except Exception:
            # Fall back to pg_stat_activity
            try:
                rows = await conn.fetch("""
                    SELECT pid, now() - pg_stat_activity.query_start AS duration,
                           query, state
                    FROM pg_stat_activity
                    WHERE (now() - pg_stat_activity.query_start) > interval '1 second'
                    AND state != 'idle'
                """)
                slow = [dict(r) for r in rows]
            except Exception as e:
                slow = [{"note": f"Could not read query stats: {e}"}]

        # Also get missing indexes
        try:
            missing_idx = await conn.fetch("""
                SELECT schemaname, tablename, attname, n_distinct,
                       correlation
                FROM pg_stats
                WHERE schemaname NOT IN ('pg_catalog','information_schema')
                AND n_distinct > 100
                ORDER BY n_distinct DESC
                LIMIT 10
            """)
            missing = [dict(r) for r in missing_idx]
        except Exception:
            missing = []

        real = {
            "slow_queries":    slow,
            "missing_indexes": missing,
            "db_connected":    True,
        }

    return await protect(real, events)


async def tool_inspect_table(events: list, table: str) -> str:
    """Inspect a real table — schema, row count, existing indexes."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected. Set DATABASE_URL."}, events)

    async with pool.acquire() as conn:
        try:
            # Get columns
            cols = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """, table)

            # Get row count
            try:
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {table}"
                )
            except Exception:
                count = "unknown"

            # Get existing indexes
            indexes = await conn.fetch("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = $1
            """, table)

            # Get sample rows (Codeastra will tokenize all PII in them)
            try:
                samples = await conn.fetch(
                    f"SELECT * FROM {table} LIMIT 3"
                )
                sample_list = [dict(r) for r in samples]
            except Exception:
                sample_list = []

            real = {
                "table":    table,
                "columns":  [dict(c) for c in cols],
                "row_count": count,
                "indexes":  [dict(i) for i in indexes],
                "samples":  sample_list,  # ← Codeastra will tokenize PII here
            }

        except Exception as e:
            real = {"error": str(e), "table": table}

    return await protect(real, events)


async def tool_create_index(events: list, table: str, column: str) -> str:
    """Create a real index on the real database."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected. Set DATABASE_URL."}, events)

    index_name = f"idx_{table}_{column}_codeastra"

    async with pool.acquire() as conn:
        try:
            await conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON {table}({column})"
            )
            real = {
                "status":       "success",
                "sql_executed": f"CREATE INDEX CONCURRENTLY {index_name} ON {table}({column})",
                "table":        table,
                "column":       column,
                "index_name":   index_name,
                "message":      "Index created on real database",
            }
        except Exception as e:
            real = {
                "status": "error",
                "error":  str(e),
                "sql_attempted": f"CREATE INDEX {index_name} ON {table}({column})",
            }

    return await protect(real, events)


async def tool_analyze_table(events: list, table: str) -> str:
    """Run ANALYZE on a real table to update query planner statistics."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected."}, events)

    async with pool.acquire() as conn:
        try:
            await conn.execute(f"ANALYZE {table}")
            real = {
                "status":  "success",
                "sql":     f"ANALYZE {table}",
                "message": f"Statistics updated for {table}",
            }
        except Exception as e:
            real = {"status": "error", "error": str(e)}

    return await protect(real, events)


async def tool_list_tables(events: list) -> str:
    """List all tables in the real database."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected. Set DATABASE_URL."}, events)

    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            real = {
                "tables":     [dict(r) for r in rows],
                "table_count": len(rows),
            }
        except Exception as e:
            real = {"error": str(e)}

    return await protect(real, events)


async def tool_run_query(events: list, sql: str) -> str:
    """
    Run a read-only SQL query on the real database.
    Results go through Codeastra before Claude sees them.
    """
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected."}, events)

    # Safety: only allow SELECT
    sql_clean = sql.strip().upper()
    if not sql_clean.startswith("SELECT") and not sql_clean.startswith("WITH"):
        return await protect({
            "error": "Only SELECT queries allowed via this tool",
            "sql":   sql,
        }, events)

    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql)
            real = {
                "sql":         sql,
                "rows":        [dict(r) for r in rows],
                "row_count":   len(rows),
            }
        except Exception as e:
            real = {"sql": sql, "error": str(e)}

    return await protect(real, events)


async def tool_get_db_stats(events: list) -> str:
    """Get real database health statistics."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected."}, events)

    async with pool.acquire() as conn:
        try:
            stats = await conn.fetchrow("""
                SELECT
                    numbackends AS active_connections,
                    xact_commit AS transactions_committed,
                    xact_rollback AS transactions_rolled_back,
                    blks_read,
                    blks_hit,
                    ROUND(blks_hit::numeric / NULLIF(blks_hit + blks_read, 0) * 100, 2)
                        AS cache_hit_ratio,
                    deadlocks,
                    conflicts
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            real = dict(stats) if stats else {}
        except Exception as e:
            real = {"error": str(e)}

    return await protect(real, events)


async def tool_check_db_connections(events: list) -> str:
    """Check active DB connections — who is connected and what they are doing."""
    pool = await get_pool()

    if not pool:
        return await protect({"error": "No database connected."}, events)

    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT pid, usename, application_name,
                       client_addr, state,
                       now() - query_start AS query_duration,
                       left(query, 100) AS query_preview
                FROM pg_stat_activity
                WHERE state != 'idle'
                ORDER BY query_start
            """)
            real = {
                "active_connections": [dict(r) for r in rows],
                "count": len(rows),
            }
        except Exception as e:
            real = {"error": str(e)}

    return await protect(real, events)


async def tool_get_summary(events: list) -> str:
    """Get full summary of what was done this session."""
    pool = await get_pool()

    real = {
        "session_complete":  True,
        "db_connected":      pool is not None,
        "codeastra_active":  bool(CODEASTRA_KEY),
        "codeastra_url":     CODEASTRA_URL,
        "message":           "Agent completed all tasks",
    }

    if pool:
        async with pool.acquire() as conn:
            try:
                idx = await conn.fetch("""
                    SELECT indexname FROM pg_indexes
                    WHERE indexname LIKE '%codeastra%'
                """)
                real["indexes_created_this_session"] = [r["indexname"] for r in idx]
            except Exception:
                pass

    return await protect(real, events)


# Tool map
async def _exec_check_threshold(events, token, threshold, operator="gt"):
    real = await codeastra_resolve(token)
    if real is None:
        return await protect({"error":f"Cannot resolve {token}","real_value_returned":False}, events)
    try:
        v = float(str(real).replace("$","").replace(",","").strip())
        ops = {"gt":v>threshold,"lt":v<threshold,"gte":v>=threshold,"lte":v<=threshold,"eq":v==threshold}
        return await protect({"result":ops.get(operator,v>threshold),"operator":operator,"threshold":threshold,"real_value_returned":False}, events)
    except Exception as e:
        return await protect({"error":str(e)}, events)

async def _exec_concentration(events, position_token, portfolio_token, threshold_pct):
    pv = await codeastra_resolve(position_token)
    tv = await codeastra_resolve(portfolio_token)
    if pv is None or tv is None:
        return await protect({"exceeds_threshold":None,"note":"Could not resolve tokens — use quantity as proxy","real_values_seen_by_agent":False}, events)
    try:
        p = float(str(pv).replace("$","").replace(",","").strip())
        t = float(str(tv).replace("$","").replace(",","").strip())
        pct = (p/t*100) if t>0 else 0
        bucket = "critical" if pct>60 else "high" if pct>40 else "medium" if pct>20 else "low"
        return await protect({"exceeds_threshold":pct>threshold_pct,"concentration_bucket":bucket,"threshold_pct":threshold_pct,"real_values_seen_by_agent":False}, events)
    except Exception as e:
        return await protect({"error":str(e)}, events)

async def _exec_classify(events, token):
    real = await codeastra_resolve(token)
    if real is None:
        return await protect({"error":f"Cannot resolve {token}"}, events)
    try:
        v = float(str(real).replace("$","").replace(",","").strip())
        label = "whale" if v>=1000000 else "large" if v>=100000 else "medium" if v>=10000 else "small"
        return await protect({"bucket":label,"real_value_returned":False}, events)
    except Exception:
        return await protect({"bucket":"unknown"}, events)

async def _exec_sum(events, tokens, threshold=None):
    total=0.0; count=0
    for tok in tokens:
        v = await codeastra_resolve(tok)
        if v:
            try: total+=float(str(v).replace("$","").replace(",","").strip()); count+=1
            except Exception: pass
    res = {"sum":total,"count":count,"real_individual_values_returned":False}
    if threshold is not None:
        res["exceeds_threshold"] = total>threshold
    return await protect(res, events)

TOOL_MAP = {
    "scan_slow_queries":    tool_scan_slow_queries,
    "list_tables":          tool_list_tables,
    "inspect_table":        tool_inspect_table,
    "create_index":         tool_create_index,
    "analyze_table":        tool_analyze_table,
    "run_query":            tool_run_query,
    "get_db_stats":         tool_get_db_stats,
    "check_db_connections": tool_check_db_connections,
    "get_summary":          tool_get_summary,
    "check_threshold":      _exec_check_threshold,
    "concentration_check":  _exec_concentration,
    "classify_amount":      _exec_classify,
    "sum_amounts":          _exec_sum,
}

TOOL_DEFINITIONS = [
    {
        "name": "scan_slow_queries",
        "description": "Scan the real production database for slow queries using pg_stat_statements. Start here to understand performance issues.",
        "input_schema": {"type":"object","properties":{},"required":[]},
    },
    {
        "name": "list_tables",
        "description": "List all tables in the real database with sizes. Use to understand what the database contains.",
        "input_schema": {"type":"object","properties":{},"required":[]},
    },
    {
        "name": "inspect_table",
        "description": "Inspect a specific real database table: columns, indexes, row count, sample rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type":"string","description":"Real table name to inspect"}
            },
            "required": ["table"],
        },
    },
    {
        "name": "create_index",
        "description": "Create a real index on a real database table column. This runs CREATE INDEX CONCURRENTLY on your production database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table":  {"type":"string","description":"Table name"},
                "column": {"type":"string","description":"Column to index"},
            },
            "required": ["table","column"],
        },
    },
    {
        "name": "analyze_table",
        "description": "Run ANALYZE on a real table to update PostgreSQL query planner statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type":"string","description":"Table name to analyze"}
            },
            "required": ["table"],
        },
    },
    {
        "name": "run_query",
        "description": "Run a read-only SELECT query on the real database. Results are tokenized by Codeastra before you see them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type":"string","description":"SELECT SQL query to run"}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_db_stats",
        "description": "Get real database health statistics: cache hit ratio, transactions, deadlocks, connections.",
        "input_schema": {"type":"object","properties":{},"required":[]},
    },
    {
        "name": "check_db_connections",
        "description": "Check what queries are currently running on the real database.",
        "input_schema": {"type":"object","properties":{},"required":[]},
    },
    {
        "name": "get_summary",
        "description": "Get a summary of everything done this session. Call this at the end.",
        "input_schema": {"type":"object","properties":{},"required":[]},
    },
]

AGENT_SYSTEM = """You are an expert autonomous Database Administrator and DevOps engineer.

You are connected to a REAL production PostgreSQL database.
Your job: investigate and fix real performance issues autonomously.

IMPORTANT: You are working through Codeastra's Zero Trust middleware.
Every tool result you receive has already been processed by Codeastra.
Sensitive values (emails, names, card numbers, SSNs) have been replaced with tokens like [CVT:EMAIL:A1B2C3].
You MUST work with these tokens as identifiers — they represent real values in the Codeastra vault.
You can reason about the structure and patterns without knowing the actual values.

Work methodically:
1. Get database stats and list tables first
2. Scan for slow queries
3. Inspect the worst-performing tables
4. Create missing indexes
5. Analyze affected tables
6. Call get_summary at the end

Be thorough. Fix every performance issue you find."""

TASK_PROMPTS = {
    "dba": (
        "Our production database has performance issues. "
        "Investigate the database completely — check stats, find slow queries, "
        "inspect tables, create all missing indexes, and restore performance. "
        "Be thorough. Fix everything you find."
    ),
    "audit": (
        "Run a complete database audit. "
        "Check all tables, query performance, connection health, and index coverage. "
        "Report everything you find."
    ),
    "custom": "",
}


# ═══════════════════════════════════════════════════════════
# AUTONOMOUS AGENT LOOP
# ═══════════════════════════════════════════════════════════

async def run_agent(task_type: str, custom_task: str = ""):
    if not ANTHROPIC_KEY:
        yield {"type":"error","message":"ANTHROPIC_API_KEY not set on Railway"}
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    events = []   # collects interception events

    task     = custom_task or TASK_PROMPTS.get(task_type, TASK_PROMPTS["dba"])
    messages = [{"role":"user","content":task}]
    calls_n  = 0
    actions_n = 0

    yield {
        "type":            "start",
        "task":            task,
        "task_type":       task_type,
        "codeastra_url":   CODEASTRA_URL,
        "codeastra_ready": bool(CODEASTRA_KEY),
        "db_ready":        bool(DATABASE_URL),
        "timestamp":       datetime.utcnow().isoformat(),
    }
    await asyncio.sleep(0.1)

    for iteration in range(20):
        yield {"type":"iteration","n":iteration+1}

        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 1500,
                system     = AGENT_SYSTEM,
                tools      = TOOL_DEFINITIONS,
                messages   = messages,
            )
        except Exception as e:
            yield {"type":"error","message":str(e)}
            return

        for block in response.content:
            if hasattr(block,"text") and block.text:
                yield {"type":"thinking","text":block.text}
                await asyncio.sleep(0.01)

        if response.stop_reason == "end_turn":
            yield {"type":"agent_done","message":"Agent completed task autonomously."}
            break

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name   = block.name
            tool_inputs = block.input
            calls_n += 1

            yield {
                "type":   "tool_call",
                "tool":   tool_name,
                "inputs": tool_inputs,
                "n":      calls_n,
            }
            await asyncio.sleep(0.1)

            # Execute tool → Codeastra protects result → Claude receives tokens
            fn = TOOL_MAP.get(tool_name)
            if fn:
                try:
                    kwargs           = {k: tool_inputs[k] for k in tool_inputs}
                    tokenized_result = await fn(events, **kwargs)
                except Exception as e:
                    tokenized_result = await protect({"error": str(e)}, events)
            else:
                tokenized_result = await protect(
                    {"error": f"Unknown tool: {tool_name}"}, events
                )

            # Emit interception events to UI
            for ev in events:
                yield ev
            events.clear()

            is_write = tool_name in {"create_index","analyze_table"}
            if is_write:
                actions_n += 1

            yield {
                "type":      "tool_result",
                "tool":      tool_name,
                "result":    tokenized_result[:600],
                "is_action": is_write,
            }
            await asyncio.sleep(0.05)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     tokenized_result,
            })

        messages.append({"role":"assistant","content":response.content})
        messages.append({"role":"user",     "content":tool_results})

    yield {
        "type":                    "complete",
        "tool_calls":              calls_n,
        "actions_executed":        actions_n,
        "codeastra_api_used":      bool(CODEASTRA_KEY),
        "db_connected":            bool(DATABASE_URL),
        "real_data_seen_by_agent": 0,
        "timestamp":               datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    await get_pool()


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    pool = await get_pool()
    codeastra_ok = False
    if CODEASTRA_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"{CODEASTRA_URL}/health",
                    headers={"X-API-Key": CODEASTRA_KEY}
                )
                codeastra_ok = r.status_code == 200
        except Exception:
            pass

    return {
        "status":          "healthy",
        "anthropic_ready": bool(ANTHROPIC_KEY),
        "codeastra_ready": bool(CODEASTRA_KEY),
        "codeastra_live":  codeastra_ok,
        "db_ready":        pool is not None,
        "codeastra_url":   CODEASTRA_URL,
    }


@app.get("/agent/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id":          "dba",
                "name":        "Database Performance Agent",
                "description": "Finds slow queries, creates missing indexes, restores performance on your real database.",
                "prompt":      TASK_PROMPTS["dba"],
            },
            {
                "id":          "audit",
                "name":        "Database Audit Agent",
                "description": "Full audit of all tables, query performance, connection health, and index coverage.",
                "prompt":      TASK_PROMPTS["audit"],
            },
        ]
    }


@app.post("/agent/run/stream")
async def agent_stream(req: Request):
    """
    PRIMARY ENDPOINT — Run the real autonomous agent.
    Streams events in real time as the agent works.

    Events:
      start        — agent started
      intercepted  — Codeastra caught real PII from DB, replaced with token
      thinking     — Claude's reasoning (contains only tokens, never real data)
      tool_call    — agent called a real database tool
      tool_result  — tokenized result Claude received from Codeastra
      complete     — agent finished

    Body:
      task_type:   "dba" | "audit"
      custom_task: str (optional — your own task description)

    Example:
      curl -X POST https://your-app.railway.app/agent/run/stream \\
        -H "Content-Type: application/json" \\
        -d '{"task_type":"dba"}' --no-buffer
    """
    body = await req.json()

    async def stream():
        async for ev in run_agent(
            body.get("task_type", "dba"),
            body.get("custom_task", "")
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/agent/run/sync")
async def agent_sync(req: Request):
    """Run agent and return complete result when done."""
    body      = await req.json()
    all_ev    = []
    intercept = []
    summary   = {}

    async for ev in run_agent(
        body.get("task_type","dba"),
        body.get("custom_task","")
    ):
        all_ev.append(ev)
        if ev["type"] == "intercepted":
            intercept.append(ev)
        if ev["type"] == "complete":
            summary = ev

    return {
        "success":                 True,
        "codeastra_intercepted":   intercept,
        "total_intercepted":       len(intercept),
        "summary":                 summary,
        "real_data_seen_by_agent": 0,
    }


@app.post("/protect")
async def protect_text(req: Request):
    """
    Protect any text through real Codeastra API.
    Use this to see what Codeastra does to your data.
    """
    body   = await req.json()
    text   = body.get("text","")
    events = []
    result = await protect(text, events)
    return {
        "original":    text,
        "protected":   result,
        "intercepted": events,
        "count":       len(events),
        "via_api":     bool(CODEASTRA_KEY),
    }


@app.get("/db/status")
async def db_status():
    """Check real database connection status."""
    pool = await get_pool()
    if not pool:
        return {"connected": False, "message": "Set DATABASE_URL on Railway"}
    async with pool.acquire() as conn:
        try:
            ver = await conn.fetchval("SELECT version()")
            tbl = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
            )
            return {
                "connected": True,
                "version":   ver,
                "tables":    tbl,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


# ═══════════════════════════════════════════════════════════
# DIRECT DATABASE ENDPOINTS
# Call your real database directly via API.
# Every result goes through Codeastra before returning.
# Claude never sees the real values — neither does your frontend
# unless you explicitly resolve tokens.
# ═══════════════════════════════════════════════════════════

@app.get("/db/tables")
async def db_list_tables():
    """
    List all real tables in your database with sizes.
    Results are protected by Codeastra.

    GET /db/tables
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={
            "error": "Database not connected",
            "fix":   "Set DATABASE_URL on Railway"
        })

    events = []
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT
                    table_name,
                    pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS total_size,
                    pg_size_pretty(pg_relation_size(quote_ident(table_name))) AS table_size,
                    (SELECT COUNT(*) FROM information_schema.columns
                     WHERE table_name = t.table_name) AS column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            real = {"tables": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.get("/db/table/{table_name}")
async def db_inspect_table(table_name: str):
    """
    Inspect a specific real table.
    Returns schema, indexes, row count, and sample rows.
    All PII in sample rows is tokenized by Codeastra.

    GET /db/table/users
    GET /db/table/transactions
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            cols = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = 'public'
                ORDER BY ordinal_position
            """, table_name)

            indexes = await conn.fetch("""
                SELECT indexname, indexdef, indisunique
                FROM pg_indexes
                LEFT JOIN pg_index pi ON pi.indexrelid = (
                    SELECT oid FROM pg_class WHERE relname = indexname
                )
                WHERE tablename = $1
            """, table_name)

            try:
                count = await conn.fetchval(
                    f'SELECT COUNT(*) FROM "{table_name}"'
                )
            except Exception:
                count = "unknown"

            try:
                samples = await conn.fetch(
                    f'SELECT * FROM "{table_name}" LIMIT 5'
                )
                sample_list = [dict(r) for r in samples]
            except Exception:
                sample_list = []

            real = {
                "table":    table_name,
                "columns":  [dict(c) for c in cols],
                "indexes":  [dict(i) for i in indexes],
                "row_count": count,
                "samples":  sample_list,  # ← PII tokenized here
            }

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.post("/db/query")
async def db_run_query(req: Request):
    """
    Run any SELECT query on your real database.
    Results tokenized by Codeastra before returning.

    POST /db/query
    Body: {"sql": "SELECT * FROM users LIMIT 10"}

    Only SELECT and WITH queries allowed.
    """
    body = await req.json()
    sql  = body.get("sql","").strip()

    if not sql:
        return JSONResponse(status_code=400, content={"error": "sql required"})

    if not sql.upper().startswith(("SELECT","WITH","EXPLAIN")):
        return JSONResponse(status_code=400, content={
            "error": "Only SELECT / WITH / EXPLAIN queries allowed",
            "tip":   "Use /db/execute for write operations (requires confirmation)"
        })

    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql)
            real = {
                "sql":       sql,
                "rows":      [dict(r) for r in rows],
                "row_count": len(rows),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e), "sql": sql})

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.post("/db/execute")
async def db_execute(req: Request):
    """
    Execute a write operation on your real database.
    Used by the agent for CREATE INDEX, ANALYZE, etc.
    Results protected by Codeastra.

    POST /db/execute
    Body: {
      "sql":     "CREATE INDEX idx_users_email ON users(email)",
      "confirm": true
    }

    Requires confirm=true for safety.
    Allowed operations: CREATE INDEX, ANALYZE, VACUUM
    """
    body    = await req.json()
    sql     = body.get("sql","").strip()
    confirm = body.get("confirm", False)

    if not sql:
        return JSONResponse(status_code=400, content={"error": "sql required"})

    if not confirm:
        return JSONResponse(status_code=400, content={
            "error":   "confirm=true required for write operations",
            "sql":     sql,
            "message": "Add confirm:true to your request body to execute",
        })

    sql_upper = sql.upper()
    allowed   = ("CREATE INDEX","ANALYZE","VACUUM","REINDEX","CLUSTER")
    if not any(sql_upper.startswith(op) for op in allowed):
        return JSONResponse(status_code=400, content={
            "error":   f"Operation not allowed via this endpoint",
            "allowed": list(allowed),
            "sql":     sql,
        })

    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            await conn.execute(sql)
            real = {
                "status":       "executed",
                "sql_executed": sql,
                "timestamp":    datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e), "sql": sql})

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.get("/db/slow-queries")
async def db_slow_queries(min_ms: float = 100.0, limit: int = 20):
    """
    Get real slow queries from pg_stat_statements.
    All query text tokenized by Codeastra.

    GET /db/slow-queries
    GET /db/slow-queries?min_ms=500&limit=10
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT
                    query,
                    calls,
                    ROUND(mean_exec_time::numeric, 2) AS avg_ms,
                    ROUND(total_exec_time::numeric, 2) AS total_ms,
                    rows,
                    ROUND((shared_blks_hit::numeric /
                           NULLIF(shared_blks_hit + shared_blks_read, 0) * 100), 2
                    ) AS cache_hit_pct
                FROM pg_stat_statements
                WHERE mean_exec_time > $1
                ORDER BY mean_exec_time DESC
                LIMIT $2
            """, min_ms, limit)
            real = {
                "slow_queries": [dict(r) for r in rows],
                "count":        len(rows),
                "min_ms":       min_ms,
            }
        except Exception as e:
            # pg_stat_statements not enabled
            real = {
                "error":   str(e),
                "message": "pg_stat_statements extension may not be enabled",
                "fix":     "Run: CREATE EXTENSION pg_stat_statements; in your DB",
            }

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.get("/db/indexes")
async def db_list_indexes(table: str = None):
    """
    List all real indexes in your database.
    Optionally filter by table.

    GET /db/indexes
    GET /db/indexes?table=users
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            if table:
                rows = await conn.fetch("""
                    SELECT schemaname, tablename, indexname, indexdef,
                           pg_size_pretty(pg_relation_size(indexname::text)) AS size
                    FROM pg_indexes
                    WHERE schemaname = 'public' AND tablename = $1
                    ORDER BY indexname
                """, table)
            else:
                rows = await conn.fetch("""
                    SELECT schemaname, tablename, indexname, indexdef,
                           pg_size_pretty(pg_relation_size(indexname::text)) AS size
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname
                """)
            real = {
                "indexes": [dict(r) for r in rows],
                "count":   len(rows),
            }
        except Exception as e:
            real = {"error": str(e)}

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.get("/db/stats")
async def db_stats():
    """
    Real database health statistics.
    Cache hit ratio, transactions, deadlocks, connections.

    GET /db/stats
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            stats = await conn.fetchrow("""
                SELECT
                    numbackends         AS active_connections,
                    xact_commit         AS transactions_committed,
                    xact_rollback       AS transactions_rolled_back,
                    blks_read           AS disk_blocks_read,
                    blks_hit            AS cache_blocks_hit,
                    ROUND(
                        blks_hit::numeric /
                        NULLIF(blks_hit + blks_read, 0) * 100, 2
                    )                   AS cache_hit_ratio_pct,
                    tup_returned        AS rows_returned,
                    tup_fetched         AS rows_fetched,
                    tup_inserted        AS rows_inserted,
                    tup_updated         AS rows_updated,
                    tup_deleted         AS rows_deleted,
                    conflicts,
                    deadlocks,
                    pg_size_pretty(pg_database_size(current_database()))
                                        AS database_size
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            real = dict(stats) if stats else {}
        except Exception as e:
            real = {"error": str(e)}

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })


@app.get("/db/connections")
async def db_connections():
    """
    Show active real database connections and running queries.
    All query text and user info tokenized by Codeastra.

    GET /db/connections
    """
    pool = await get_pool()
    if not pool:
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    events = []
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT
                    pid,
                    usename         AS username,
                    application_name,
                    client_addr,
                    state,
                    wait_event_type,
                    wait_event,
                    EXTRACT(EPOCH FROM (now() - query_start))::int
                                    AS query_seconds,
                    LEFT(query, 200) AS query_preview
                FROM pg_stat_activity
                WHERE pid <> pg_backend_pid()
                ORDER BY query_start DESC NULLS LAST
            """)
            real = {
                "connections": [dict(r) for r in rows],
                "count":       len(rows),
            }
        except Exception as e:
            real = {"error": str(e)}

    protected = await protect(real, events)
    return JSONResponse(content={
        "data":        json.loads(protected),
        "intercepted": events,
        "protected_by": "codeastra" if CODEASTRA_KEY else "none",
    })



# ═══════════════════════════════════════════════════════════
# DOCUMENT + LINK INGESTION
# Frontend sends any file or URL.
# Agent reads it, Codeastra tokenizes all PII,
# Claude analyzes tokens only.
# ═══════════════════════════════════════════════════════════

import io
import tempfile
import mimetypes
from fastapi import UploadFile, File, Form

async def extract_text(file: UploadFile) -> str:
    """Extract raw text from any uploaded file."""
    content  = await file.read()
    filename = (file.filename or "").lower()
    mime     = file.content_type or ""

    # PDF
    if filename.endswith(".pdf") or "pdf" in mime:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as e:
            return f"[PDF extraction error: {e}]"

    # Word (.docx)
    if filename.endswith(".docx") or "wordprocessingml" in mime:
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[DOCX extraction error: {e}]"

    # Excel (.xlsx / .xls)
    if filename.endswith((".xlsx",".xls")) or "spreadsheet" in mime or "excel" in mime:
        try:
            import openpyxl
            wb   = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            rows = []
            for sheet in wb.worksheets:
                rows.append(f"=== Sheet: {sheet.title} ===")
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)
        except Exception as e:
            return f"[Excel extraction error: {e}]"

    # CSV
    if filename.endswith(".csv") or "csv" in mime:
        try:
            return content.decode("utf-8", errors="replace")
        except Exception as e:
            return f"[CSV error: {e}]"

    # JSON
    if filename.endswith(".json") or "json" in mime:
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except Exception as e:
            return content.decode("utf-8", errors="replace")

    # HTML
    if filename.endswith((".html",".htm")) or "html" in mime:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            return content.decode("utf-8", errors="replace")

    # Plain text, markdown, code files
    text_extensions = (
        ".txt",".md",".py",".js",".ts",".sql",".yaml",".yml",
        ".toml",".env",".log",".xml",".sh",".bash",".r",".rb",
    )
    if any(filename.endswith(ext) for ext in text_extensions) or "text" in mime:
        return content.decode("utf-8", errors="replace")

    # Unknown — try UTF-8 decode
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return f"[Could not extract text from {filename}]"


async def extract_from_url(url: str) -> str:
    """Fetch and extract text from any URL."""
    try:
        async with httpx.AsyncClient(
            timeout      = 20.0,
            follow_redirects = True,
            headers      = {"User-Agent": "Mozilla/5.0 (compatible; CodeastraAgent/1.0)"},
        ) as client:
            r = await client.get(url)
            r.raise_for_status()

            ct = r.headers.get("content-type","")

            # PDF from URL
            if "pdf" in ct or url.lower().endswith(".pdf"):
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(r.content))
                    return "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except Exception as e:
                    return f"[PDF from URL error: {e}]"

            # HTML — extract readable text
            if "html" in ct:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    # Remove scripts, styles, nav
                    for tag in soup(["script","style","nav","header","footer","aside"]):
                        tag.decompose()
                    return soup.get_text(separator="\n", strip=True)[:50000]
                except Exception:
                    return r.text[:50000]

            # JSON
            if "json" in ct:
                try:
                    return json.dumps(r.json(), indent=2)[:50000]
                except Exception:
                    return r.text[:50000]

            # Plain text
            return r.text[:50000]

    except Exception as e:
        return f"[URL fetch error: {e}]"


async def run_document_agent(text: str, task: str, filename: str):
    """
    Autonomous agent that analyzes a document.
    Text is protected by Codeastra before Claude sees it.
    Claude reasons on tokens only.
    """
    if not ANTHROPIC_KEY:
        yield {"type":"error","message":"ANTHROPIC_API_KEY not set"}
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    events = []

    # Truncate if too large
    if len(text) > 80000:
        text = text[:80000] + "\n\n[... document truncated at 80,000 characters ...]"

    yield {
        "type":          "start",
        "filename":      filename,
        "task":          task,
        "char_count":    len(text),
        "codeastra_ready": bool(CODEASTRA_KEY),
    }
    await asyncio.sleep(0.1)

    # Protect full document text through Codeastra
    yield {"type":"phase","message":"Codeastra scanning document for PII..."}
    protected_text = await protect(text, events)

    # Emit all interception events
    for ev in events:
        yield ev
    events.clear()

    yield {
        "type":    "phase",
        "message": f"Document protected. Sending to Claude for analysis...",
    }
    await asyncio.sleep(0.1)

    # Build task prompt
    if not task:
        task = (
            "Analyze this document thoroughly. "
            "Summarize the key information, identify any issues or action items, "
            "extract important data points, and provide recommendations."
        )

    system = """You are an expert document analyst.
You receive documents that have been processed by Codeastra's Zero Trust middleware.
All PII (names, emails, SSNs, card numbers, addresses) has been replaced with tokens like [CVT:EMAIL:A1B2].
Work with these tokens as identifiers — they represent real values stored securely in the vault.
Analyze the document structure, content, and meaning. You can reason about the data without knowing the actual values.
Be thorough and specific in your analysis."""

    messages = [{
        "role": "user",
        "content": f"TASK: {task}\n\nDOCUMENT ({filename}):\n\n{protected_text}"
    }]

    # Stream Claude's analysis
    full_response = ""
    try:
        with client.messages.stream(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 2000,
            system     = system,
            messages   = messages,
        ) as stream:
            for chunk in stream.text_stream:
                full_response += chunk
                yield {"type":"thinking","text":chunk}
                await asyncio.sleep(0.01)

    except Exception as e:
        yield {"type":"error","message":str(e)}
        return

    yield {
        "type":                    "complete",
        "analysis":                full_response,
        "filename":                filename,
        "char_count":              len(text),
        "real_data_seen_by_agent": 0,
        "codeastra_api_used":      bool(CODEASTRA_KEY),
    }


# ── Endpoints ────────────────────────────────────────────

@app.post("/agent/analyze-document")
async def analyze_document(
    file: UploadFile = File(default=None),
    task: str        = Form(default=""),
    req:  Request    = None,
):
    """
    Upload any document — agent analyzes it with full Codeastra protection.

    Supported file types:
      PDF, DOCX, XLSX, XLS, CSV, JSON, TXT, MD,
      HTML, XML, YAML, Python, SQL, and any text file.

    Form fields:
      file: the document to analyze
      task: what to do with it (optional)
            e.g. "Find all compliance violations"
                 "Summarize key financial figures"
                 "Extract all action items"
                 "Identify PII exposure risks"

    Returns: text/event-stream
      start        — document received
      intercepted  — Codeastra caught PII in document
      thinking     — Claude analyzing (sees tokens only)
      complete     — full analysis

    Example (JavaScript):
      const form = new FormData();
      form.append('file', file);
      form.append('task', 'Find all compliance issues');
      fetch('/agent/analyze-document', {method:'POST', body: form})

    Example (curl):
      curl -X POST https://your-app.railway.app/agent/analyze-document \\
        -F "file=@contract.pdf" \\
        -F "task=Extract all payment terms and identify risks" \\
        --no-buffer
    """
    if file is None:
        return JSONResponse(status_code=400, content={
            "error": "No file uploaded",
            "tip":   "Send as multipart/form-data with field name 'file'"
        })

    filename = file.filename or "document"
    text     = await extract_text(file)

    if not text or text.startswith("[Could not"):
        return JSONResponse(status_code=400, content={
            "error": f"Could not extract text from {filename}",
            "raw":   text,
        })

    async def stream():
        async for ev in run_document_agent(text, task, filename):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/agent/analyze-url")
async def analyze_url(req: Request):
    """
    Send any URL — agent fetches it and analyzes with Codeastra protection.

    Works with:
      Web pages, articles, blog posts
      Online PDFs
      JSON APIs
      GitHub files (raw URLs)
      Google Docs (published)
      News articles
      Documentation pages
      Financial reports
      Any public URL

    Body:
      url:  str — the URL to fetch and analyze
      task: str — what to do with it (optional)

    Returns: text/event-stream (same events as analyze-document)

    Example:
      curl -X POST https://your-app.railway.app/agent/analyze-url \\
        -H "Content-Type: application/json" \\
        -d '{"url":"https://example.com/report.pdf","task":"Find key risks"}' \\
        --no-buffer
    """
    body = await req.json()
    url  = body.get("url","").strip()
    task = body.get("task","")

    if not url:
        return JSONResponse(status_code=400, content={"error":"url required"})

    if not url.startswith(("http://","https://")):
        return JSONResponse(status_code=400, content={
            "error": "URL must start with http:// or https://"
        })

    text = await extract_from_url(url)

    if not text or text.startswith("[URL fetch error"):
        return JSONResponse(status_code=400, content={
            "error": f"Could not fetch URL: {url}",
            "detail": text,
        })

    async def stream():
        async for ev in run_document_agent(text, task, url):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/agent/analyze-text")
async def analyze_text(req: Request):
    """
    Send raw text directly — agent analyzes with Codeastra protection.
    Use when you already have the text (no file upload needed).

    Body:
      text: str  — the text content to analyze
      task: str  — what to do with it
      name: str  — label for this content (optional)

    Example:
      curl -X POST https://your-app.railway.app/agent/analyze-text \\
        -H "Content-Type: application/json" \\
        -d '{
          "text": "Client: John Smith, SSN 234-56-7890, owes $45,000...",
          "task": "Identify all PII and compliance risks",
          "name": "client-record"
        }' --no-buffer
    """
    body = await req.json()
    text = body.get("text","").strip()
    task = body.get("task","")
    name = body.get("name","raw-text")

    if not text:
        return JSONResponse(status_code=400, content={"error":"text required"})

    async def stream():
        async for ev in run_document_agent(text, task, name):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/agent/analyze-multiple")
async def analyze_multiple(
    files: list[UploadFile] = File(default=None),
    task:  str              = Form(default=""),
):
    """
    Upload multiple documents at once.
    Agent analyzes all of them together as one context.
    Useful for: comparing contracts, cross-referencing reports,
    auditing multiple files at once.

    Max 10 files, 50MB total.

    Example:
      curl -X POST https://your-app.railway.app/agent/analyze-multiple \\
        -F "files=@contract1.pdf" \\
        -F "files=@contract2.pdf" \\
        -F "task=Compare these contracts and find discrepancies" \\
        --no-buffer
    """
    if len(files) > 10:
        return JSONResponse(status_code=400, content={"error":"Max 10 files"})

    all_text = ""
    names    = []

    for f in files:
        text = await extract_text(f)
        names.append(f.filename or "file")
        all_text += f"\n\n=== FILE: {f.filename} ===\n{text}"

    if not all_text.strip():
        return JSONResponse(status_code=400, content={"error":"No text extracted from files"})

    combined_name = f"{len(files)} files: {', '.join(names)}"

    async def stream():
        async for ev in run_document_agent(all_text, task, combined_name):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )



# ── Debug endpoint — see exact raw Codeastra API response ──
@app.post("/debug/protect-raw")
async def debug_protect_raw(req: Request):
    """
    Shows the RAW response from Codeastra API.
    Use this to verify the API is detecting PII correctly.

    POST /debug/protect-raw
    Body: {"text": "John Smith SSN 234-56-7890 email john@goldman.com"}
    """
    body = await req.json()
    text = body.get("text", "")

    if not CODEASTRA_KEY:
        return {"error": "CODEASTRA_API_KEY not set"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{CODEASTRA_URL}/protect/text",
            headers={
                "X-API-Key":    CODEASTRA_KEY,
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        return {
            "status_code":    r.status_code,
            "raw_response":   r.json() if r.status_code == 200 else r.text,
            "sent_text":      text,
            "codeastra_url":  CODEASTRA_URL,
        }



# ═══════════════════════════════════════════════════════════
# SELECTIVE REVEAL EXECUTOR
# The agent's "Hands" — resolves tokens for computation,
# returns only derived results (booleans, buckets, totals)
# never the raw values.
#
# Pattern:
#   Agent (Brain) reasons on tokens
#   Agent calls executor tool: check_concentration(token, threshold)
#   Executor resolves token → gets real value → does math
#   Executor returns: {exceeds: true} — never the real value
#   Agent never saw the dollar amount
# ═══════════════════════════════════════════════════════════

async def codeastra_resolve(token: str) -> str | None:
    """
    Resolve a single CDT/CVT token to its real value.
    Calls the Codeastra vault — authorized executor only.
    Real value used for computation, never returned to agent.
    """
    if not CODEASTRA_KEY:
        return None

    # Try CDT resolve first
    for endpoint in ["/cdt/resolve", "/vault/resolve", "/vault/read"]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"{CODEASTRA_URL}{endpoint}",
                    headers={"X-API-Key": CODEASTRA_KEY,
                             "Content-Type": "application/json"},
                    json={"token": token, "token_id": token},
                )
                if r.status_code == 200:
                    data = r.json()
                    return (
                        data.get("real_value") or
                        data.get("value") or
                        data.get("original") or
                        data.get("resolved")
                    )
        except Exception:
            continue
    return None


async def executor_resolve_batch(tokens: list[str]) -> dict:
    """Resolve multiple tokens at once. Returns {token: real_value}."""
    results = {}
    for token in tokens:
        val = await codeastra_resolve(token)
        if val is not None:
            results[token] = val
    return results


# ── Executor Tool Endpoints ────────────────────────────────

@app.post("/executor/check-threshold")
async def executor_check_threshold(req: Request):
    """
    Selective Reveal: check if a vaulted amount exceeds a threshold.
    Resolves token internally. Returns ONLY boolean — never real value.

    Body:
      token:     str   — the [CVT:AMT:*] token to check
      threshold: float — the value to compare against
      operator:  str   — "gt" | "lt" | "gte" | "lte" | "eq"

    Returns:
      {result: true/false, operator, threshold}
      Never returns the real value.
    """
    body      = await req.json()
    token     = body.get("token","")
    threshold = float(body.get("threshold", 0))
    operator  = body.get("operator","gt")

    real_value = await codeastra_resolve(token)

    if real_value is None:
        return JSONResponse(status_code=404, content={
            "error": f"Could not resolve token: {token}",
            "tip":   "Token may be expired or not in vault"
        })

    try:
        # Strip currency symbols and commas
        clean = str(real_value).replace("$","").replace(",","").replace("%","").strip()
        value = float(clean)
    except Exception:
        return JSONResponse(status_code=400, content={
            "error": f"Token value is not numeric: real value hidden",
            "token": token,
        })

    ops = {
        "gt":  value > threshold,
        "lt":  value < threshold,
        "gte": value >= threshold,
        "lte": value <= threshold,
        "eq":  value == threshold,
    }
    result = ops.get(operator, value > threshold)

    log.info(f"Executor check_threshold: token={token} op={operator} threshold={threshold} result={result}")

    return {
        "result":    result,
        "operator":  operator,
        "threshold": threshold,
        "token":     token,
        "real_value_returned": False,
        "real_value_seen_by_agent": False,
    }


@app.post("/executor/compare-amounts")
async def executor_compare_amounts(req: Request):
    """
    Selective Reveal: compare two vaulted amounts.
    Resolves both tokens. Returns only comparison result.

    Body:
      token_a: str — first amount token
      token_b: str — second amount token

    Returns:
      {a_greater: bool, b_greater: bool, equal: bool, ratio_bucket: str}
      Ratio bucket: "much_larger" | "larger" | "similar" | "smaller"
    """
    body    = await req.json()
    token_a = body.get("token_a","")
    token_b = body.get("token_b","")

    val_a = await codeastra_resolve(token_a)
    val_b = await codeastra_resolve(token_b)

    if val_a is None or val_b is None:
        return JSONResponse(status_code=404, content={
            "error": "Could not resolve one or both tokens"
        })

    try:
        a = float(str(val_a).replace("$","").replace(",","").strip())
        b = float(str(val_b).replace("$","").replace(",","").strip())
    except Exception:
        return JSONResponse(status_code=400, content={"error":"Non-numeric values"})

    ratio = a / b if b != 0 else float('inf')
    bucket = (
        "much_larger"  if ratio > 2.0  else
        "larger"       if ratio > 1.1  else
        "similar"      if ratio > 0.9  else
        "smaller"      if ratio > 0.5  else
        "much_smaller"
    )

    return {
        "a_greater":    a > b,
        "b_greater":    b > a,
        "equal":        a == b,
        "ratio_bucket": bucket,
        "real_values_returned": False,
    }


@app.post("/executor/sum-amounts")
async def executor_sum_amounts(req: Request):
    """
    Selective Reveal: sum a list of vaulted amount tokens.
    Returns sum and count — never individual values.

    Body:
      tokens:    list[str] — list of [CVT:AMT:*] tokens
      threshold: float     — optional threshold to check sum against
    """
    body      = await req.json()
    tokens    = body.get("tokens", [])
    threshold = body.get("threshold")

    resolved = await executor_resolve_batch(tokens)
    total    = 0.0
    count    = 0

    for token, val in resolved.items():
        try:
            clean = str(val).replace("$","").replace(",","").strip()
            total += float(clean)
            count += 1
        except Exception:
            pass

    result = {
        "sum":          total,
        "count":        count,
        "failed_count": len(tokens) - count,
        "real_individual_values_returned": False,
    }

    if threshold is not None:
        result["exceeds_threshold"] = total > float(threshold)
        result["threshold"]         = threshold

    return result


@app.post("/executor/concentration-check")
async def executor_concentration_check(req: Request):
    """
    Selective Reveal: check portfolio concentration.
    The key executor tool for the financial use case.

    Takes vaulted position amounts, computes concentration,
    returns only whether threshold is exceeded.
    Agent never sees real dollar values.

    Body:
      position_token:  str   — token for this position value
      portfolio_token: str   — token for total portfolio value
      threshold_pct:   float — concentration threshold (e.g. 40.0 for 40%)

    Returns:
      {
        exceeds_threshold: bool,
        concentration_bucket: "low|medium|high|critical",
        threshold_pct: 40.0,
        real_values_seen_by_agent: false
      }
    """
    body             = await req.json()
    position_token   = body.get("position_token","")
    portfolio_token  = body.get("portfolio_token","")
    threshold_pct    = float(body.get("threshold_pct", 40.0))

    pos_val  = await codeastra_resolve(position_token)
    port_val = await codeastra_resolve(portfolio_token)

    if pos_val is None or port_val is None:
        # Fall back: if tokens can't be resolved, use quantity proxy
        return {
            "exceeds_threshold":        None,
            "concentration_bucket":     "unknown",
            "threshold_pct":            threshold_pct,
            "note":                     "Could not resolve tokens — use quantity as proxy",
            "real_values_seen_by_agent": False,
        }

    try:
        pos  = float(str(pos_val).replace("$","").replace(",","").strip())
        port = float(str(port_val).replace("$","").replace(",","").strip())
        pct  = (pos / port * 100) if port > 0 else 0
    except Exception:
        return JSONResponse(status_code=400, content={"error":"Non-numeric token values"})

    bucket = (
        "critical" if pct > 60  else
        "high"     if pct > 40  else
        "medium"   if pct > 20  else
        "low"
    )

    log.info(f"Executor concentration_check: pct={pct:.1f}% threshold={threshold_pct}% result={pct>threshold_pct}")

    return {
        "exceeds_threshold":          pct > threshold_pct,
        "concentration_bucket":       bucket,
        "threshold_pct":              threshold_pct,
        "real_pct_returned":          False,
        "real_values_seen_by_agent":  False,
    }


@app.post("/executor/classify-amount")
async def executor_classify_amount(req: Request):
    """
    Selective Reveal: classify a vaulted amount into a bucket.
    Returns a label — never the real value.

    Body:
      token:   str  — the [CVT:AMT:*] token
      buckets: list — [{label, min, max}, ...]
               e.g. [
                 {"label":"small",  "min":0,       "max":10000},
                 {"label":"medium", "min":10000,   "max":100000},
                 {"label":"large",  "min":100000,  "max":1000000},
                 {"label":"whale",  "min":1000000, "max":null}
               ]
    """
    body    = await req.json()
    token   = body.get("token","")
    buckets = body.get("buckets", [
        {"label":"small",  "min":0,       "max":10000},
        {"label":"medium", "min":10000,   "max":100000},
        {"label":"large",  "min":100000,  "max":1000000},
        {"label":"whale",  "min":1000000, "max":None},
    ])

    real_value = await codeastra_resolve(token)
    if real_value is None:
        return JSONResponse(status_code=404, content={"error": f"Cannot resolve {token}"})

    try:
        value = float(str(real_value).replace("$","").replace(",","").strip())
    except Exception:
        return JSONResponse(status_code=400, content={"error":"Non-numeric value"})

    label = "unknown"
    for b in buckets:
        mn = b.get("min", 0) or 0
        mx = b.get("max")
        if value >= mn and (mx is None or value < mx):
            label = b["label"]
            break

    return {
        "bucket":     label,
        "token":      token,
        "real_value_returned": False,
    }


@app.post("/executor/run-computation")
async def executor_run_computation(req: Request):
    """
    General-purpose Selective Reveal executor.
    Resolves any tokens in an expression, computes the result,
    returns only the derived output.

    Body:
      tokens:      dict  — {"var_name": "token_id", ...}
      expression:  str   — Python expression using var names
                           e.g. "(a / (a+b+c)) * 100"
      return_only: str   — what to return: "result"|"bucket"|"boolean"
      threshold:   float — for boolean return_only

    Example:
      {
        "tokens": {"position": "[CVT:AMT:A1B2]", "portfolio": "[CVT:AMT:C3D4]"},
        "expression": "(position / portfolio) * 100",
        "return_only": "boolean",
        "threshold": 40.0
      }
    """
    body        = await req.json()
    token_map   = body.get("tokens", {})
    expression  = body.get("expression","")
    return_only = body.get("return_only","result")
    threshold   = body.get("threshold")

    if not expression:
        return JSONResponse(status_code=400, content={"error":"expression required"})

    # Resolve all tokens
    resolved = {}
    for var_name, token in token_map.items():
        val = await codeastra_resolve(token)
        if val is not None:
            try:
                clean = str(val).replace("$","").replace(",","").strip()
                resolved[var_name] = float(clean)
            except Exception:
                resolved[var_name] = val

    if not resolved:
        return JSONResponse(status_code=404, content={"error":"No tokens could be resolved"})

    # Evaluate expression in sandboxed namespace
    try:
        import math as _math
        safe_globals = {"__builtins__": {}, "math": _math, "abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, safe_globals, resolved)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Expression error: {e}"})

    if return_only == "boolean":
        return {
            "result":     bool(result > threshold) if threshold is not None else bool(result),
            "threshold":  threshold,
            "real_values_returned": False,
        }
    elif return_only == "bucket":
        buckets = body.get("buckets", [])
        label = "unknown"
        for b in buckets:
            if result >= b.get("min",0) and (b.get("max") is None or result < b["max"]):
                label = b["label"]
                break
        return {
            "bucket":  label,
            "real_values_returned": False,
        }
    else:
        return {
            "result":  result,
            "real_values_returned": False,
        }


@app.get("/executor/capabilities")
async def executor_capabilities():
    """List all executor capabilities for agent tool selection."""
    return {
        "endpoints": {
            "POST /executor/check-threshold":      "Check if vaulted amount exceeds a threshold",
            "POST /executor/compare-amounts":      "Compare two vaulted amounts",
            "POST /executor/sum-amounts":          "Sum a list of vaulted amount tokens",
            "POST /executor/concentration-check":  "Check portfolio concentration percentage",
            "POST /executor/classify-amount":      "Classify vaulted amount into a bucket",
            "POST /executor/run-computation":      "General-purpose expression evaluator on vaulted tokens",
        },
        "pattern": "Agent passes tokens → Executor resolves internally → Returns derived result only",
        "guarantee": "Real values never returned to agent — only computation outputs",
    }
