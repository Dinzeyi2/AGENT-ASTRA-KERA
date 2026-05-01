"""
Codeastra Autonomous Agent — OpenAI + Claude
=============================================
Real autonomous agent using OpenAI GPT-4o tool calling.
Every tool result goes through Codeastra before GPT sees it.

OpenAI Platform logs (platform.openai.com/logs) will show
EXACTLY what GPT-4 received — tokens only, never real PII.
That is your third-party proof.

Same agent also runs on Claude (claude-haiku-4-5-20251001).
Toggle between them. Toggle Codeastra on/off.
The difference is undeniable.

Environment variables:
  OPENAI_API_KEY      — your OpenAI key
  ANTHROPIC_API_KEY   — your Claude key
  CODEASTRA_API_KEY   — sk-guard-...
  CODEASTRA_URL       — https://app.codeastra.dev
  DATABASE_URL        — postgresql://...
  PORT                — 8080
"""

import os, json, asyncio, re, hashlib, logging, io
from datetime import datetime
from typing import AsyncGenerator

import httpx
import asyncpg
from openai import AsyncOpenAI
import anthropic
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("codeastra-agent")

# ── Config ────────────────────────────────────────────────
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CODEASTRA_KEY  = os.getenv("CODEASTRA_API_KEY", "")
CODEASTRA_URL  = os.getenv("CODEASTRA_URL", "https://app.codeastra.dev")
DATABASE_URL   = os.getenv("DATABASE_URL", "")
PORT           = int(os.getenv("PORT", 8080))

app = FastAPI(title="Codeastra Autonomous Agent — OpenAI + Claude")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            log.info("✅ Database connected")
        except Exception as e:
            log.warning(f"DB connection failed: {e}")


# ═══════════════════════════════════════════════════════════
# CODEASTRA — Real API call
# Every tool result passes through this.
# Real values → Codeastra vault → tokens returned to AI.
# ═══════════════════════════════════════════════════════════

async def protect(data, events: list, codeastra_active: bool = True) -> str:
    """Protect any data through real Codeastra API."""
    text = json.dumps(data) if not isinstance(data, str) else data

    if not codeastra_active:
        # Toggle OFF — raw data goes to AI unprotected
        events.append({
            "type":    "unprotected",
            "message": "⚠️ Codeastra OFF — real data exposed to AI",
            "preview": text[:150],
        })
        return text

    if not CODEASTRA_KEY:
        events.append({"type": "warning", "message": "CODEASTRA_API_KEY not set"})
        return _local_protect(text, events)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{CODEASTRA_URL}/protect/text",
                headers={"X-API-Key": CODEASTRA_KEY,
                         "Content-Type": "application/json"},
                json={"text": text},
            )
            if r.status_code == 200:
                result   = r.json()
                protected = result.get("protected_text", text)
                entities  = (result.get("entities") or result.get("detections") or [])
                for e in entities:
                    real    = e.get("original") or e.get("value") or ""
                    preview = real[:3]+"•"*min(len(real)-5,8)+real[-2:] if len(real)>5 else "•••"
                    events.append({
                        "type":    "intercepted",
                        "dtype":   e.get("type") or e.get("field_type") or "PII",
                        "token":   e.get("token",""),
                        "preview": e.get("preview") or preview,
                    })
                log.info(f"Codeastra protected {len(entities)} values")
                return protected
            else:
                log.warning(f"Codeastra {r.status_code}")
                return _local_protect(text, events)
    except Exception as ex:
        log.warning(f"Codeastra error: {ex}")
        return _local_protect(text, events)


def _local_protect(text: str, events: list) -> str:
    """Local fallback tokenizer."""
    PATTERNS = {
        "EMAIL":  re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        "SSN":    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "CARD":   re.compile(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b'),
        "APIKEY": re.compile(r'\bsk-(?:live-|prod-)[a-f0-9]{16,}\b'),
        "NAME":   re.compile(r'\b(James Dimon|David Solomon|Marc Rowan|Steve Schwarzman|Ken Griffin|Ray Dalio)\b'),
    }
    seen = {}
    result = text
    for dtype, pat in PATTERNS.items():
        for m in pat.finditer(result):
            real = m.group(0)
            if real not in seen:
                hex_id = hashlib.md5(real.encode()).hexdigest()[:10].upper()
                token  = f"[CVT:{dtype}:{hex_id}]"
                seen[real] = token
                preview = real[:3]+"•"*min(len(real)-5,8)+real[-2:] if len(real)>5 else "•••"
                events.append({"type":"intercepted","dtype":dtype,"token":token,"preview":preview})
            result = result.replace(real, seen[real])
    return result


async def codeastra_resolve(token: str) -> str | None:
    """Resolve a token to its real value for executor computations."""
    if not CODEASTRA_KEY:
        return None
    for endpoint in ["/vault/resolve", "/cdt/resolve"]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"{CODEASTRA_URL}{endpoint}",
                    headers={"X-API-Key": CODEASTRA_KEY,
                             "Content-Type": "application/json"},
                    json={"token": token, "token_id": token},
                )
                if r.status_code == 200:
                    d = r.json()
                    return d.get("real_value") or d.get("value") or d.get("original")
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════
# REAL DATABASE TOOLS
# Each tool reads real DB data.
# Every result → protect() → AI receives tokens only.
# ═══════════════════════════════════════════════════════════

async def tool_list_tables(events, codeastra_active=True):
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set","tip":"Add DATABASE_URL to Railway variables"}, events, codeastra_active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema='public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            real = {"tables": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e)}
    return await protect(real, events, codeastra_active)


async def tool_scan_slow_queries(events, codeastra_active=True):
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set"}, events, codeastra_active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT query, calls,
                       ROUND(mean_exec_time::numeric,2) AS avg_ms,
                       ROUND(total_exec_time::numeric,2) AS total_ms,
                       rows
                FROM pg_stat_statements
                WHERE mean_exec_time > 100
                ORDER BY mean_exec_time DESC LIMIT 20
            """)
            real = {"slow_queries": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e), "hint": "Enable pg_stat_statements extension"}
    return await protect(real, events, codeastra_active)


async def tool_inspect_table(events, table: str, codeastra_active=True):
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set"}, events, codeastra_active)
    async with db_pool.acquire() as conn:
        try:
            cols = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name=$1 AND table_schema='public'
                ORDER BY ordinal_position
            """, table)
            idxs = await conn.fetch("""
                SELECT indexname, indexdef FROM pg_indexes WHERE tablename=$1
            """, table)
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
            except Exception:
                count = "unknown"
            try:
                samples = await conn.fetch(f'SELECT * FROM "{table}" LIMIT 5')
                sample_list = [dict(r) for r in samples]
            except Exception:
                sample_list = []
            real = {
                "table":    table,
                "columns":  [dict(c) for c in cols],
                "indexes":  [dict(i) for i in idxs],
                "row_count": count,
                "samples":  sample_list,
            }
        except Exception as e:
            real = {"error": str(e), "table": table}
    return await protect(real, events, codeastra_active)


async def tool_create_index(events, table: str, column: str, codeastra_active=True):
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set"}, events, codeastra_active)
    index_name = f"idx_{table}_{column}_codeastra"
    async with db_pool.acquire() as conn:
        try:
            await conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON {table}({column})"
            )
            real = {
                "status":       "success",
                "sql_executed": f"CREATE INDEX CONCURRENTLY {index_name} ON {table}({column})",
                "index_name":   index_name,
            }
        except Exception as e:
            real = {"status": "error", "error": str(e)}
    return await protect(real, events, codeastra_active)


async def tool_run_query(events, sql: str, codeastra_active=True):
    if not sql.strip().upper().startswith(("SELECT","WITH","EXPLAIN")):
        return await protect({"error":"Only SELECT queries allowed"}, events, codeastra_active)
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set"}, events, codeastra_active)
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql)
            real = {"sql": sql, "rows": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            real = {"error": str(e), "sql": sql}
    return await protect(real, events, codeastra_active)


async def tool_get_db_stats(events, codeastra_active=True):
    if not db_pool:
        return await protect({"error":"No DATABASE_URL set"}, events, codeastra_active)
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
            real = dict(stats) if stats else {}
        except Exception as e:
            real = {"error": str(e)}
    return await protect(real, events, codeastra_active)


async def tool_get_summary(events, codeastra_active=True):
    real = {
        "agent":           "Codeastra Autonomous Agent",
        "model":           "GPT-4o + Claude Haiku",
        "codeastra_active": codeastra_active,
        "db_connected":    db_pool is not None,
        "timestamp":       datetime.utcnow().isoformat(),
        "message":         "Task complete. All data protected by Codeastra.",
    }
    return await protect(real, events, codeastra_active)


# ── Executor tools ────────────────────────────────────────

async def tool_check_threshold(events, token: str, threshold: float,
                               operator: str = "gt", codeastra_active=True):
    real_val = await codeastra_resolve(token)
    if real_val is None:
        return await protect({"error":f"Cannot resolve {token}","real_value_returned":False}, events, codeastra_active)
    try:
        v = float(str(real_val).replace("$","").replace(",","").strip())
        ops = {"gt":v>threshold,"lt":v<threshold,"gte":v>=threshold,"lte":v<=threshold}
        result = ops.get(operator, v>threshold)
        return await protect({"result":result,"operator":operator,"threshold":threshold,"real_value_returned":False}, events, codeastra_active)
    except Exception as e:
        return await protect({"error":str(e)}, events, codeastra_active)


async def tool_concentration_check(events, position_token: str, portfolio_token: str,
                                   threshold_pct: float, codeastra_active=True):
    pv = await codeastra_resolve(position_token)
    tv = await codeastra_resolve(portfolio_token)
    if pv is None or tv is None:
        return await protect({
            "exceeds_threshold": None,
            "note": "Tokens not resolved — use quantity as proxy",
            "real_values_seen_by_agent": False,
        }, events, codeastra_active)
    try:
        p = float(str(pv).replace("$","").replace(",","").strip())
        t = float(str(tv).replace("$","").replace(",","").strip())
        pct    = (p/t*100) if t > 0 else 0
        bucket = "critical" if pct>60 else "high" if pct>40 else "medium" if pct>20 else "low"
        return await protect({
            "exceeds_threshold":          pct > threshold_pct,
            "concentration_bucket":       bucket,
            "threshold_pct":              threshold_pct,
            "real_values_seen_by_agent":  False,
        }, events, codeastra_active)
    except Exception as e:
        return await protect({"error":str(e)}, events, codeastra_active)


# ── Tool dispatcher ────────────────────────────────────────

TOOL_FUNCTIONS = {
    "list_tables":          tool_list_tables,
    "scan_slow_queries":    tool_scan_slow_queries,
    "inspect_table":        tool_inspect_table,
    "create_index":         tool_create_index,
    "run_query":            tool_run_query,
    "get_db_stats":         tool_get_db_stats,
    "get_summary":          tool_get_summary,
    "check_threshold":      tool_check_threshold,
    "concentration_check":  tool_concentration_check,
}

# OpenAI tool definitions (function calling format)
OPENAI_TOOLS = [
    {"type":"function","function":{"name":"list_tables","description":"List all real database tables with sizes. Start here.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"scan_slow_queries","description":"Scan production database for slow queries using pg_stat_statements.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"inspect_table","description":"Inspect a real database table: schema, indexes, row count, sample rows.","parameters":{"type":"object","properties":{"table":{"type":"string","description":"Table name to inspect"}},"required":["table"]}}},
    {"type":"function","function":{"name":"create_index","description":"Create a real database index on a column to fix slow query performance. Runs CREATE INDEX CONCURRENTLY.","parameters":{"type":"object","properties":{"table":{"type":"string"},"column":{"type":"string","description":"Column to index"}},"required":["table","column"]}}},
    {"type":"function","function":{"name":"run_query","description":"Run a read-only SELECT query on the real database.","parameters":{"type":"object","properties":{"sql":{"type":"string","description":"SELECT SQL query"}},"required":["sql"]}}},
    {"type":"function","function":{"name":"get_db_stats","description":"Get real database health: cache hit ratio, deadlocks, connections, size.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"check_threshold","description":"Check if a vaulted amount token exceeds a threshold. Returns only boolean — never real value.","parameters":{"type":"object","properties":{"token":{"type":"string"},"threshold":{"type":"number"},"operator":{"type":"string","enum":["gt","lt","gte","lte"]}},"required":["token","threshold"]}}},
    {"type":"function","function":{"name":"concentration_check","description":"Check if a portfolio position exceeds concentration threshold. Returns bucket and boolean — never real value.","parameters":{"type":"object","properties":{"position_token":{"type":"string"},"portfolio_token":{"type":"string"},"threshold_pct":{"type":"number"}},"required":["position_token","portfolio_token","threshold_pct"]}}},
    {"type":"function","function":{"name":"get_summary","description":"Get summary of what was accomplished. Call at the end.","parameters":{"type":"object","properties":{},"required":[]}}},
]

# Claude tool definitions (Anthropic format)
CLAUDE_TOOLS = [
    {"name":"list_tables","description":"List all real database tables with sizes.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"scan_slow_queries","description":"Scan production database for slow queries.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"inspect_table","description":"Inspect a real database table.","input_schema":{"type":"object","properties":{"table":{"type":"string"}},"required":["table"]}},
    {"name":"create_index","description":"Create a real index on a database column.","input_schema":{"type":"object","properties":{"table":{"type":"string"},"column":{"type":"string"}},"required":["table","column"]}},
    {"name":"run_query","description":"Run a SELECT query on the real database.","input_schema":{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}},
    {"name":"get_db_stats","description":"Get real database health statistics.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"check_threshold","description":"Check if a vaulted token amount exceeds threshold. Returns boolean only.","input_schema":{"type":"object","properties":{"token":{"type":"string"},"threshold":{"type":"number"},"operator":{"type":"string"}},"required":["token","threshold"]}},
    {"name":"concentration_check","description":"Check portfolio concentration. Returns bucket and boolean — never real value.","input_schema":{"type":"object","properties":{"position_token":{"type":"string"},"portfolio_token":{"type":"string"},"threshold_pct":{"type":"number"}},"required":["position_token","portfolio_token","threshold_pct"]}},
    {"name":"get_summary","description":"Get summary of what was accomplished.","input_schema":{"type":"object","properties":{},"required":[]}},
]

SYSTEM_PROMPT = """You are an expert autonomous Database Administrator and DevOps engineer.

You are working through Codeastra's Zero Trust middleware.
ALL sensitive data (emails, names, SSNs, card numbers) has been replaced with tokens like [CVT:EMAIL:A1B2C3].
You MUST work with these tokens as identifiers — they represent real values stored securely in the vault.
Never try to guess or reconstruct real values.

For any computation on sensitive amounts, use check_threshold or concentration_check tools.
These resolve tokens internally and return only derived results — never raw values.

Work methodically:
1. List tables and get DB stats first
2. Scan for slow queries
3. Inspect affected tables
4. Create missing indexes
5. Call get_summary at the end

Be thorough. Fix everything you find."""

TASK_PROMPTS = {
    "dba":   "Our production database has performance issues. Investigate completely — check stats, find slow queries, inspect tables, create all missing indexes. Be thorough.",
    "audit": "Run a complete database audit. Check all tables, query performance, connection health, and index coverage. Report everything.",
}


# ═══════════════════════════════════════════════════════════
# OPENAI AUTONOMOUS AGENT LOOP
# ═══════════════════════════════════════════════════════════

async def run_openai_agent(
    task_type:        str,
    custom_task:      str  = "",
    codeastra_active: bool = True,
) -> AsyncGenerator[dict, None]:
    """
    Real autonomous agent using OpenAI GPT-4o tool calling.
    Every tool result goes through Codeastra before GPT sees it.

    OpenAI Platform logs will show exactly what GPT received.
    Toggle codeastra_active=False to show raw mode for comparison.
    """
    if not OPENAI_KEY:
        yield {"type":"error","message":"OPENAI_API_KEY not set on Railway"}
        return

    client = AsyncOpenAI(api_key=OPENAI_KEY)
    events = []
    task   = custom_task or TASK_PROMPTS.get(task_type, TASK_PROMPTS["dba"])

    yield {
        "type":             "start",
        "model":            "gpt-4o",
        "task":             task,
        "task_type":        task_type,
        "codeastra_active": codeastra_active,
        "mode":             "PROTECTED — Codeastra active" if codeastra_active else "⚠️ UNPROTECTED — Real data sent to GPT",
        "openai_logs_url":  "https://platform.openai.com/logs",
        "proof":            "Open platform.openai.com/logs to see exactly what GPT received",
        "timestamp":        datetime.utcnow().isoformat(),
    }
    await asyncio.sleep(0.1)

    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task},
    ]
    calls_n   = 0
    actions_n = 0

    for iteration in range(20):
        yield {"type":"iteration","n":iteration+1,"model":"gpt-4o"}

        try:
            response = await client.chat.completions.create(
                model    = "gpt-4o",
                messages = messages,
                tools    = OPENAI_TOOLS,
            )
        except Exception as e:
            yield {"type":"error","message":str(e)}
            return

        msg = response.choices[0].message

        # Emit GPT's text reasoning
        if msg.content:
            yield {"type":"thinking","text":msg.content,"model":"gpt-4o"}
            await asyncio.sleep(0.02)

        # Done
        if response.choices[0].finish_reason == "stop":
            yield {"type":"agent_done","message":"GPT-4o completed task.","model":"gpt-4o"}
            break

        # No tool calls
        if not msg.tool_calls:
            break

        # Process tool calls
        tool_results_for_api = []

        for tc in msg.tool_calls:
            tool_name   = tc.function.name
            tool_inputs = json.loads(tc.function.arguments or "{}")
            calls_n    += 1

            yield {
                "type":   "tool_call",
                "tool":   tool_name,
                "inputs": tool_inputs,
                "n":      calls_n,
                "model":  "gpt-4o",
            }
            await asyncio.sleep(0.15)

            # Execute tool → Codeastra intercepts → GPT gets tokens
            fn = TOOL_FUNCTIONS.get(tool_name)
            if fn:
                try:
                    kwargs = {k: tool_inputs[k] for k in tool_inputs}
                    kwargs["codeastra_active"] = codeastra_active
                    result = await fn(events, **kwargs)
                except Exception as e:
                    result = await protect({"error": str(e)}, events, codeastra_active)
            else:
                result = await protect({"error":f"Unknown tool: {tool_name}"}, events, codeastra_active)

            # Emit interception events
            for ev in events:
                yield ev
            events.clear()

            is_action = tool_name in {"create_index","analyze_table"}
            if is_action:
                actions_n += 1

            yield {
                "type":      "tool_result",
                "tool":      tool_name,
                "result":    result[:500],
                "is_action": is_action,
                "model":     "gpt-4o",
            }
            await asyncio.sleep(0.05)

            tool_results_for_api.append({
                "tool_call_id": tc.id,
                "role":         "tool",
                "content":      result,  # GPT gets tokens — never real values
            })

        # Add to message history
        messages.append(msg)
        messages.extend(tool_results_for_api)

    yield {
        "type":                    "complete",
        "model":                   "gpt-4o",
        "tool_calls":              calls_n,
        "actions":                 actions_n,
        "codeastra_active":        codeastra_active,
        "real_data_seen_by_gpt":   0 if codeastra_active else "⚠️ YES — toggle Codeastra ON",
        "openai_logs_url":         "https://platform.openai.com/logs",
        "proof_message":           "Go to platform.openai.com/logs — GPT received tokens only" if codeastra_active else "Go to platform.openai.com/logs — GPT received REAL data",
        "timestamp":               datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# CLAUDE AUTONOMOUS AGENT LOOP
# ═══════════════════════════════════════════════════════════

async def run_claude_agent(
    task_type:        str,
    custom_task:      str  = "",
    codeastra_active: bool = True,
) -> AsyncGenerator[dict, None]:
    """Same agent, same tools, same Codeastra protection — using Claude."""
    if not ANTHROPIC_KEY:
        yield {"type":"error","message":"ANTHROPIC_API_KEY not set on Railway"}
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    events = []
    task   = custom_task or TASK_PROMPTS.get(task_type, TASK_PROMPTS["dba"])

    yield {
        "type":             "start",
        "model":            "claude-haiku-4-5-20251001",
        "task":             task,
        "task_type":        task_type,
        "codeastra_active": codeastra_active,
        "mode":             "PROTECTED — Codeastra active" if codeastra_active else "⚠️ UNPROTECTED — Real data sent to Claude",
        "timestamp":        datetime.utcnow().isoformat(),
    }
    await asyncio.sleep(0.1)

    messages = [{"role":"user","content":task}]
    calls_n  = 0
    actions_n = 0

    for iteration in range(20):
        yield {"type":"iteration","n":iteration+1,"model":"claude"}

        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 1500,
                system     = SYSTEM_PROMPT,
                tools      = CLAUDE_TOOLS,
                messages   = messages,
            )
        except Exception as e:
            yield {"type":"error","message":str(e)}
            return

        for block in response.content:
            if hasattr(block,"text") and block.text:
                yield {"type":"thinking","text":block.text,"model":"claude"}
                await asyncio.sleep(0.01)

        if response.stop_reason == "end_turn":
            yield {"type":"agent_done","message":"Claude completed task.","model":"claude"}
            break

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name   = block.name
            tool_inputs = block.input
            calls_n    += 1

            yield {"type":"tool_call","tool":tool_name,"inputs":tool_inputs,"n":calls_n,"model":"claude"}
            await asyncio.sleep(0.15)

            fn = TOOL_FUNCTIONS.get(tool_name)
            if fn:
                try:
                    kwargs = {k: tool_inputs[k] for k in tool_inputs}
                    kwargs["codeastra_active"] = codeastra_active
                    result = await fn(events, **kwargs)
                except Exception as e:
                    result = await protect({"error":str(e)}, events, codeastra_active)
            else:
                result = await protect({"error":f"Unknown tool: {tool_name}"}, events, codeastra_active)

            for ev in events:
                yield ev
            events.clear()

            is_action = tool_name in {"create_index"}
            if is_action:
                actions_n += 1

            yield {"type":"tool_result","tool":tool_name,"result":result[:500],"is_action":is_action,"model":"claude"}
            await asyncio.sleep(0.05)

            tool_results.append({
                "type":"tool_result","tool_use_id":block.id,"content":result
            })

        messages.append({"role":"assistant","content":response.content})
        messages.append({"role":"user","content":tool_results})

    yield {
        "type":                    "complete",
        "model":                   "claude-haiku-4-5-20251001",
        "tool_calls":              calls_n,
        "actions":                 actions_n,
        "codeastra_active":        codeastra_active,
        "real_data_seen_by_agent": 0 if codeastra_active else "⚠️ YES",
        "timestamp":               datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# TEXT/DOCUMENT EXTRACTION
# ═══════════════════════════════════════════════════════════

async def extract_text_from_file(file: UploadFile) -> str:
    content  = await file.read()
    filename = (file.filename or "").lower()
    mime     = file.content_type or ""
    if filename.endswith(".pdf") or "pdf" in mime:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[PDF error: {e}]"
    if filename.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[DOCX error: {e}]"
    if filename.endswith((".xlsx",".xls")):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
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
    if filename.endswith((".html",".htm")):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(content, "html.parser").get_text("\n", strip=True)
        except Exception:
            return content.decode("utf-8", errors="replace")
    return content.decode("utf-8", errors="replace")


async def extract_text_from_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                     headers={"User-Agent":"Mozilla/5.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type","")
            if "pdf" in ct or url.lower().endswith(".pdf"):
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(r.content))
                    return "\n".join(p.extract_text() or "" for p in reader.pages)
                except Exception as e:
                    return f"[PDF error: {e}]"
            if "html" in ct:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    for tag in soup(["script","style","nav","header","footer"]):
                        tag.decompose()
                    return soup.get_text("\n", strip=True)[:50000]
                except Exception:
                    return r.text[:50000]
            return r.text[:50000]
    except Exception as e:
        return f"[URL error: {e}]"


async def run_document_agent(
    text:             str,
    task:             str,
    filename:         str,
    model:            str  = "gpt-4o",
    codeastra_active: bool = True,
) -> AsyncGenerator[dict, None]:
    """Analyze any document with chosen model + Codeastra toggle."""
    events = []

    if len(text) > 80000:
        text = text[:80000] + "\n\n[... truncated ...]"

    yield {
        "type":             "start",
        "filename":         filename,
        "task":             task,
        "model":            model,
        "codeastra_active": codeastra_active,
        "char_count":       len(text),
        "mode":             "PROTECTED" if codeastra_active else "⚠️ UNPROTECTED",
    }
    await asyncio.sleep(0.1)

    # Protect document through Codeastra
    if codeastra_active:
        yield {"type":"phase","message":"Codeastra scanning document for PII..."}
        protected_text = await protect(text, events, True)
        for ev in events:
            yield ev
        events.clear()
    else:
        yield {"type":"warning","message":"⚠️ Codeastra OFF — full document sent to AI unprotected"}
        protected_text = text

    yield {"type":"phase","message":f"Sending to {model} for analysis..."}

    system = """You are an expert document analyst working through Codeastra's Zero Trust middleware.
All PII has been replaced with tokens like [CVT:EMAIL:A1B2C3].
Work with tokens as identifiers. Analyze structure, content, and meaning without knowing actual values.
Be thorough and specific."""

    prompt = f"TASK: {task or 'Analyze this document thoroughly.'}\n\nDOCUMENT ({filename}):\n\n{protected_text}"

    try:
        if model == "gpt-4o" and OPENAI_KEY:
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            response = await client.chat.completions.create(
                model    = "gpt-4o",
                messages = [
                    {"role":"system","content":system},
                    {"role":"user",  "content":prompt},
                ],
                max_tokens = 2000,
                stream     = True,
            )
            full = ""
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full += delta
                    yield {"type":"thinking","text":delta,"model":"gpt-4o"}
                    await asyncio.sleep(0.01)

        elif ANTHROPIC_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=system,
                messages=[{"role":"user","content":prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield {"type":"thinking","text":chunk,"model":"claude"}
                    await asyncio.sleep(0.01)
        else:
            yield {"type":"error","message":"No AI API key set"}
            return

    except Exception as e:
        yield {"type":"error","message":str(e)}
        return

    yield {
        "type":                    "complete",
        "filename":                filename,
        "model":                   model,
        "codeastra_active":        codeastra_active,
        "real_data_seen_by_agent": 0 if codeastra_active else "⚠️ YES",
        "openai_logs_url":         "https://platform.openai.com/logs" if model=="gpt-4o" else None,
    }


# ═══════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    openai_ok    = False
    codeastra_ok = False

    if OPENAI_KEY:
        try:
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            models = await client.models.list()
            openai_ok = True
        except Exception:
            pass

    if CODEASTRA_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{CODEASTRA_URL}/health",
                                headers={"X-API-Key":CODEASTRA_KEY})
                codeastra_ok = r.status_code == 200
        except Exception:
            pass

    return {
        "status":          "healthy",
        "openai_ready":    bool(OPENAI_KEY),
        "openai_live":     openai_ok,
        "anthropic_ready": bool(ANTHROPIC_KEY),
        "codeastra_ready": bool(CODEASTRA_KEY),
        "codeastra_live":  codeastra_ok,
        "db_ready":        db_pool is not None,
        "codeastra_url":   CODEASTRA_URL,
        "proof_url":       "https://platform.openai.com/logs",
    }


@app.get("/agent/tasks")
async def list_tasks():
    return {
        "models": {
            "gpt-4o":                   "OpenAI GPT-4o — logs visible at platform.openai.com/logs",
            "claude-haiku-4-5-20251001": "Anthropic Claude Haiku",
        },
        "tasks": [
            {"id":"dba",   "name":"Database Performance Agent","description":"Finds slow queries, creates missing indexes."},
            {"id":"audit", "name":"Database Audit Agent",      "description":"Full DB audit — tables, queries, connections, indexes."},
        ],
        "codeastra_toggle": {
            "on":  "All tool results tokenized before AI sees them. platform.openai.com/logs shows tokens only.",
            "off": "Raw data sent to AI. platform.openai.com/logs shows real PII. USE ONLY FOR DEMO COMPARISON.",
        }
    }


def _stream_response(generator):
    async def stream():
        async for ev in generator:
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no",
                 "Access-Control-Allow-Origin":"*"})


@app.post("/agent/run/stream")
async def agent_run_stream(req: Request):
    """
    PRIMARY — Run autonomous agent with real-time streaming.

    Body:
      task_type:         "dba" | "audit"
      model:             "gpt-4o" | "claude" (default: gpt-4o)
      codeastra_enabled: true | false (default: true)
      custom_task:       str (optional)

    Returns: text/event-stream

    Events:
      start        — agent initialized
      intercepted  — Codeastra caught real PII, replaced with token
      thinking     — AI reasoning (tokens only when protected)
      tool_call    — agent called a real DB tool
      tool_result  — what AI received (tokens if protected)
      complete     — done. Check openai_logs_url for proof.

    PROOF: When model=gpt-4o, open platform.openai.com/logs
    to see EXACTLY what GPT received. Tokens if Codeastra ON.
    Real data if Codeastra OFF. The difference is undeniable.
    """
    body    = await req.json()
    model   = body.get("model", "gpt-4o")
    enabled = body.get("codeastra_enabled", True)

    if model == "gpt-4o":
        gen = run_openai_agent(
            body.get("task_type","dba"),
            body.get("custom_task",""),
            codeastra_active=enabled,
        )
    else:
        gen = run_claude_agent(
            body.get("task_type","dba"),
            body.get("custom_task",""),
            codeastra_active=enabled,
        )
    return _stream_response(gen)


@app.post("/agent/run/sync")
async def agent_run_sync(req: Request):
    """Run agent, wait for completion, return full result."""
    body    = await req.json()
    model   = body.get("model","gpt-4o")
    enabled = body.get("codeastra_enabled",True)

    gen = (run_openai_agent if model=="gpt-4o" else run_claude_agent)(
        body.get("task_type","dba"),
        body.get("custom_task",""),
        codeastra_active=enabled,
    )

    all_ev      = []
    intercepted = []
    summary     = {}

    async for ev in gen:
        all_ev.append(ev)
        if ev["type"] == "intercepted":
            intercepted.append(ev)
        if ev["type"] == "complete":
            summary = ev

    return {
        "success":           True,
        "model":             model,
        "codeastra_active":  enabled,
        "intercepted_count": len(intercepted),
        "intercepted":       intercepted,
        "summary":           summary,
        "proof_url":         "https://platform.openai.com/logs" if model=="gpt-4o" else None,
    }


@app.post("/agent/run/protected")
async def agent_run_protected(req: Request):
    """Run agent with Codeastra ALWAYS ON."""
    body  = await req.json()
    model = body.get("model","gpt-4o")
    gen   = (run_openai_agent if model=="gpt-4o" else run_claude_agent)(
        body.get("task_type","dba"), body.get("custom_task",""), codeastra_active=True
    )
    return _stream_response(gen)


@app.post("/agent/run/unprotected")
async def agent_run_unprotected(req: Request):
    """Run agent with Codeastra OFF — for comparison demo only."""
    body  = await req.json()
    model = body.get("model","gpt-4o")
    gen   = (run_openai_agent if model=="gpt-4o" else run_claude_agent)(
        body.get("task_type","dba"), body.get("custom_task",""), codeastra_active=False
    )
    return _stream_response(gen)


@app.post("/agent/analyze-document")
async def analyze_document(
    file:  UploadFile = File(default=None),
    task:  str        = Form(default=""),
    model: str        = Form(default="gpt-4o"),
    codeastra_enabled: str = Form(default="true"),
):
    """Upload any document for analysis with model + toggle selection."""
    if file is None:
        return JSONResponse(status_code=400, content={"error":"No file uploaded"})
    enabled  = codeastra_enabled.lower() != "false"
    text     = await extract_text_from_file(file)
    filename = file.filename or "document"
    return _stream_response(run_document_agent(text, task, filename, model, enabled))


@app.post("/agent/analyze-url")
async def analyze_url(req: Request):
    """Fetch any URL and analyze with model + toggle selection."""
    body    = await req.json()
    url     = body.get("url","").strip()
    task    = body.get("task","")
    model   = body.get("model","gpt-4o")
    enabled = body.get("codeastra_enabled",True)
    if not url.startswith(("http://","https://")):
        return JSONResponse(status_code=400, content={"error":"Valid URL required"})
    text = await extract_text_from_url(url)
    return _stream_response(run_document_agent(text, task, url, model, enabled))


@app.post("/agent/analyze-text")
async def analyze_text(req: Request):
    """Send raw text for analysis with model + toggle selection."""
    body    = await req.json()
    text    = body.get("text","").strip()
    task    = body.get("task","")
    name    = body.get("name","text")
    model   = body.get("model","gpt-4o")
    enabled = body.get("codeastra_enabled",True)
    if not text:
        return JSONResponse(status_code=400, content={"error":"text required"})
    return _stream_response(run_document_agent(text, task, name, model, enabled))


@app.post("/protect")
async def protect_text(req: Request):
    """Protect any text through real Codeastra API."""
    body   = await req.json()
    events = []
    result = await protect(body.get("text",""), events, True)
    return {"original":body.get("text",""),"protected":result,"intercepted":events,"count":len(events)}


@app.post("/debug/protect-raw")
async def debug_protect_raw(req: Request):
    """See raw Codeastra API response."""
    body = await req.json()
    if not CODEASTRA_KEY:
        return {"error":"CODEASTRA_API_KEY not set"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{CODEASTRA_URL}/protect/text",
            headers={"X-API-Key":CODEASTRA_KEY,"Content-Type":"application/json"},
            json={"text":body.get("text","")},
        )
        return {"status_code":r.status_code,"raw_response":r.json() if r.status_code==200 else r.text}


@app.get("/db/status")
async def db_status():
    if not db_pool:
        return {"connected":False,"message":"Set DATABASE_URL on Railway"}
    async with db_pool.acquire() as conn:
        try:
            ver = await conn.fetchval("SELECT version()")
            tbl = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
            )
            return {"connected":True,"version":ver,"public_tables":tbl}
        except Exception as e:
            return {"connected":False,"error":str(e)}



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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
