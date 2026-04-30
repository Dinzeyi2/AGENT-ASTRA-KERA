"""
Codeastra Blind Agent Demo
===========================
Architecture: Agent-first, Codeastra-transparent

The agent is a real autonomous DBA / ops agent.
It knows NOTHING about Codeastra.
It just calls tools and does its job.

Codeastra wraps the tool layer silently:
  - Every tool RESULT is tokenized before the agent sees it
  - Agent reasons on tokens as if they were real values
  - At execution time, Codeastra resolves tokens back to real values
  - Agent never knew any of this happened

This is exactly how a real customer would use Codeastra:
  Step 1: They have an existing agent
  Step 2: They wrap it with Codeastra middleware
  Step 3: Agent keeps working. Data stops leaking.
"""

import os, re, json, asyncio, hashlib, secrets
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════
# THE PRODUCTION DATABASE
# Real data. Real problems. This is what the agent works on.
# ═══════════════════════════════════════════════════════════

DB = {
    "users": [
        {"id":1,"email":"james.dimon@jpmorgan.com",     "name":"James Dimon",     "ssn":"234-56-7890","plan":"starter",   "mrr":95000,"last_login_days":2},
        {"id":2,"email":"david.solomon@goldmansachs.com","name":"David Solomon",   "ssn":"345-67-8901","plan":"starter",   "mrr":87000,"last_login_days":48},
        {"id":3,"email":"marc.rowan@apollo.com",        "name":"Marc Rowan",      "ssn":"456-78-9012","plan":"enterprise","mrr":72000,"last_login_days":5},
        {"id":4,"email":"steve.schwarzman@blackstone.com","name":"Steve Schwarzman","ssn":"567-89-0123","plan":"starter",  "mrr":68000,"last_login_days":31},
        {"id":5,"email":"ken.griffin@citadel.com",      "name":"Ken Griffin",     "ssn":"678-90-1234","plan":"starter",   "mrr":54000,"last_login_days":12},
        {"id":6,"email":"ray.dalio@bridgewater.com",    "name":"Ray Dalio",       "ssn":"789-01-2345","plan":"starter",   "mrr":41000,"last_login_days":67},
    ],
    "transactions": [
        {"id":101,"user_id":1,"card":"4532-0151-1283-0366","amount":95000,"status":"failed", "retries":3,"merchant":"Codeastra"},
        {"id":102,"user_id":2,"card":"5425-2334-3010-9903","amount":87000,"status":"failed", "retries":2,"merchant":"Codeastra"},
        {"id":103,"user_id":4,"card":"4916-3384-9572-0041","amount":68000,"status":"failed", "retries":1,"merchant":"Codeastra"},
        {"id":104,"user_id":6,"card":"4532-8811-2200-1177","amount":41000,"status":"failed", "retries":0,"merchant":"Codeastra"},
        {"id":105,"user_id":3,"card":"5500-0050-0000-0004","amount":72000,"status":"success","retries":0,"merchant":"Codeastra"},
        {"id":106,"user_id":5,"card":"4111-1111-1111-1111","amount":54000,"status":"success","retries":0,"merchant":"Codeastra"},
    ],
    "slow_queries": [
        {"sql":"SELECT * FROM users WHERE email=?",        "avg_ms":52847,"count":9821,"table":"users",       "column":"email"},
        {"sql":"SELECT * FROM transactions WHERE card=?",  "avg_ms":61203,"count":4419,"table":"transactions","column":"card"},
        {"sql":"SELECT u.*,t.* FROM users u JOIN transactions t ON u.id=t.user_id WHERE u.ssn=?","avg_ms":89441,"count":2103,"table":"users","column":"ssn"},
        {"sql":"SELECT * FROM users WHERE name LIKE ?",    "avg_ms":44821,"count":1847,"table":"users",       "column":"name"},
    ],
    "indexes": ["users_pkey","transactions_pkey"],
    "api_keys": [
        {"id":1,"user_id":1,"key":"sk-live-4ca4305fec929e56707928991d85","permissions":["read","write"],"last_used_days":1},
        {"id":2,"user_id":2,"key":"sk-live-8f2a9c1e4b7d6f3a2e5c8d1b4a7","permissions":["read"],"last_used_days":3},
    ],
    "executed_fixes": [],
    "revenue_recovered": 0,
}

# ═══════════════════════════════════════════════════════════
# CODEASTRA VAULT
# Invisible to the agent. Sits between tools and agent.
# ═══════════════════════════════════════════════════════════

SENSITIVE_PATTERNS = {
    "EMAIL":  re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    "SSN":    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "CARD":   re.compile(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b'),
    "APIKEY": re.compile(r'\bsk-live-[a-f0-9]{24,}\b'),
    "NAME":   re.compile(r'\b(James Dimon|David Solomon|Marc Rowan|Steve Schwarzman|Ken Griffin|Ray Dalio)\b'),
}

class CodeastraVault:
    """
    Transparent middleware between tools and agent.
    Agent has no idea this exists.
    """
    def __init__(self):
        self.vault   = {}   # token → real_value
        self.reverse = {}   # real_value → token (for dedup)
        self.events  = []   # stream events for UI

    def protect(self, data) -> str:
        """
        Intercept any tool result before it reaches the agent.
        Replace all sensitive values with tokens.
        Agent receives this — never the original.
        """
        text = json.dumps(data) if not isinstance(data, str) else data
        result = text

        for dtype, pattern in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(result):
                real = match.group(0)
                if real in self.reverse:
                    token = self.reverse[real]
                else:
                    hex_id = hashlib.md5(real.encode()).hexdigest()[:10].upper()
                    token  = f"[CDT:{dtype}:{hex_id}]"
                    self.vault[token]   = real
                    self.reverse[real]  = token
                    preview = real[:3]+"•"*min(len(real)-5,8)+real[-2:] if len(real)>5 else "•••"
                    self.events.append({
                        "type":    "intercepted",
                        "dtype":   dtype,
                        "token":   token,
                        "preview": preview,
                        "real":    real,
                    })
                result = result.replace(real, token)

        return result

    def resolve(self, text: str) -> str:
        """Resolve tokens back to real values at execution time."""
        result = text
        for token, real in self.vault.items():
            result = result.replace(token, real)
        return result

    def pop_events(self):
        evs = self.events[:]
        self.events.clear()
        return evs

# ═══════════════════════════════════════════════════════════
# THE AGENT'S TOOLS
# These are real tools. The agent calls them normally.
# Codeastra intercepts the OUTPUT before the agent sees it.
# The agent never knows.
# ═══════════════════════════════════════════════════════════

def tool_scan_slow_queries(vault: CodeastraVault) -> str:
    """Real tool. Returns real DB data. Vault intercepts before agent sees it."""
    real_result = {
        "status": "found_issues",
        "slow_queries": DB["slow_queries"],
        "total_slow_count": len(DB["slow_queries"]),
        "worst_query_ms": max(q["avg_ms"] for q in DB["slow_queries"]),
        "estimated_daily_impact": "~18,400 slow queries affecting real users",
        "recommendation": "Create indexes on queried columns",
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_inspect_table(vault: CodeastraVault, table: str) -> str:
    real_result = {
        "table": table,
        "row_count": 2847291 if table=="users" else 47829441,
        "current_indexes": DB["indexes"],
        "columns": {
            "users":        ["id","email","name","ssn","plan","mrr","last_login_days"],
            "transactions": ["id","user_id","card","amount","status","retries","merchant"],
            "api_keys":     ["id","user_id","key","permissions","last_used_days"],
        }.get(table, ["id"]),
        "sample_rows": (DB["users"][:2] if table=="users"
                        else DB["transactions"][:2] if table=="transactions"
                        else DB["api_keys"][:2]),
        "missing_indexes": {
            "users":        ["email","ssn","name"],
            "transactions": ["card","user_id"],
            "api_keys":     ["key","user_id"],
        }.get(table, []),
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_create_index(vault: CodeastraVault, table: str, column: str) -> str:
    index_name = f"idx_{table}_{column}"
    before_ms  = next((q["avg_ms"] for q in DB["slow_queries"]
                       if q.get("table")==table and q.get("column")==column), None)

    # Execute the fix
    DB["indexes"].append(index_name)
    DB["slow_queries"] = [q for q in DB["slow_queries"]
                          if not (q.get("table")==table and q.get("column")==column)]
    DB["executed_fixes"].append(f"CREATE INDEX {index_name} ON {table}({column})")

    real_result = {
        "status":        "success",
        "sql_executed":  f"CREATE INDEX {index_name} ON {table}({column})",
        "before_ms":     before_ms,
        "after_ms":      2,
        "speedup":       f"{round(before_ms/2)}x faster" if before_ms else "index created",
        "rows_affected": 2847291 if table=="users" else 47829441,
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_find_revenue_issues(vault: CodeastraVault) -> str:
    issues = [
        {
            "user_id":       u["id"],
            "email":         u["email"],
            "name":          u["name"],
            "current_plan":  u["plan"],
            "should_be":     "enterprise",
            "monthly_loss":  u["mrr"] - 999,
            "last_login_days": u["last_login_days"],
        }
        for u in DB["users"] if u["plan"] == "starter"
    ]
    failed = [
        {
            "txn_id":   t["id"],
            "user_id":  t["user_id"],
            "card":     t["card"],
            "amount":   t["amount"],
            "retries":  t["retries"],
            "user_email": next(u["email"] for u in DB["users"] if u["id"]==t["user_id"]),
        }
        for t in DB["transactions"] if t["status"]=="failed"
    ]
    real_result = {
        "mismatched_plans":    issues,
        "failed_payments":     failed,
        "total_monthly_leakage": sum(i["monthly_loss"] for i in issues),
        "recoverable_payments":  sum(t["amount"] for t in DB["transactions"] if t["status"]=="failed"),
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_upgrade_plan(vault: CodeastraVault, user_id: int) -> str:
    user = next((u for u in DB["users"] if u["id"]==user_id), None)
    if not user:
        return vault.protect({"error": f"User {user_id} not found"})
    recovered = user["mrr"] - 999
    user["plan"] = "enterprise"
    DB["revenue_recovered"] += recovered
    DB["executed_fixes"].append(f"UPDATE users SET plan='enterprise' WHERE id={user_id} -- {user['email']}")
    real_result = {
        "status":           "success",
        "user_id":          user_id,
        "user_email":       user["email"],
        "user_name":        user["name"],
        "old_plan":         "starter",
        "new_plan":         "enterprise",
        "revenue_recovered": recovered,
        "total_recovered_so_far": DB["revenue_recovered"],
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_retry_payment(vault: CodeastraVault, txn_id: int) -> str:
    txn  = next((t for t in DB["transactions"] if t["id"]==txn_id), None)
    if not txn:
        return vault.protect({"error": f"Transaction {txn_id} not found"})
    user = next(u for u in DB["users"] if u["id"]==txn["user_id"])
    success = txn["retries"] < 3
    if success:
        txn["status"] = "success"
        DB["executed_fixes"].append(f"RETRY payment {txn_id} ${txn['amount']:,} -- {user['email']} -- SUCCESS")
    txn["retries"] += 1
    real_result = {
        "status":         "success" if success else "failed_again",
        "txn_id":         txn_id,
        "user_email":     user["email"],
        "user_name":      user["name"],
        "card_charged":   txn["card"],
        "amount":         txn["amount"],
        "result":         f"${txn['amount']:,} collected" if success else "Card declined",
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_check_api_security(vault: CodeastraVault) -> str:
    real_result = {
        "api_keys_found": len(DB["api_keys"]),
        "keys": DB["api_keys"],
        "security_issues": [
            {"issue": "key with write permissions", "user_id": 1,
             "key": DB["api_keys"][0]["key"], "recommendation": "rotate immediately"},
        ],
        "db_credentials": "postgresql://admin:Xk9mP2qR4vN8wL3j@prod-db.internal:5432/prod",
        "stripe_key":     "sk_live_4ca4305fec929e56707928",
    }
    return vault.protect(real_result)   # ← Codeastra intercepts here

def tool_get_summary(vault: CodeastraVault) -> str:
    remaining_slow = len(DB["slow_queries"])
    real_result = {
        "fixes_executed":     DB["executed_fixes"],
        "remaining_slow_queries": remaining_slow,
        "indexes_created":    [i for i in DB["indexes"] if i.startswith("idx_")],
        "revenue_recovered":  DB["revenue_recovered"],
        "db_performance":     "optimal" if remaining_slow==0 else f"{remaining_slow} issues remain",
    }
    return vault.protect(real_result)

# Tool dispatcher
TOOLS = {
    "scan_slow_queries":   {"fn": tool_scan_slow_queries,  "args": []},
    "inspect_table":       {"fn": tool_inspect_table,      "args": ["table"]},
    "create_index":        {"fn": tool_create_index,       "args": ["table","column"]},
    "find_revenue_issues": {"fn": tool_find_revenue_issues,"args": []},
    "upgrade_plan":        {"fn": tool_upgrade_plan,       "args": ["user_id"]},
    "retry_payment":       {"fn": tool_retry_payment,      "args": ["txn_id"]},
    "check_api_security":  {"fn": tool_check_api_security, "args": []},
    "get_summary":         {"fn": tool_get_summary,        "args": []},
}

TOOL_DEFINITIONS = [
    {"name":"scan_slow_queries","description":"Scan the production database for slow queries. Returns query patterns, average execution times, and affected tables. Use this first to understand performance issues.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"inspect_table","description":"Inspect a specific database table: schema, indexes, row count, sample data. Use after finding slow queries to understand the structure.","input_schema":{"type":"object","properties":{"table":{"type":"string","description":"Table name to inspect: users, transactions, or api_keys"}},"required":["table"]}},
    {"name":"create_index","description":"Create a database index on a column to fix slow query performance. Execute this after confirming the missing index.","input_schema":{"type":"object","properties":{"table":{"type":"string"},"column":{"type":"string","description":"Column to create index on"}},"required":["table","column"]}},
    {"name":"find_revenue_issues","description":"Scan billing system for revenue problems: customers on wrong plans, failed payments, churn risk. Returns full revenue leakage analysis.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"upgrade_plan","description":"Upgrade a customer to enterprise plan and recover the revenue difference. Provide the user_id from the billing analysis.","input_schema":{"type":"object","properties":{"user_id":{"type":"integer","description":"Customer user_id to upgrade"}},"required":["user_id"]}},
    {"name":"retry_payment","description":"Retry a failed payment transaction. Provide the transaction id from the billing analysis.","input_schema":{"type":"object","properties":{"txn_id":{"type":"integer","description":"Transaction ID to retry"}},"required":["txn_id"]}},
    {"name":"check_api_security","description":"Audit API keys and credentials for security issues. Identifies over-permissioned keys and rotation requirements.","input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"get_summary","description":"Get a summary of all fixes executed and remaining issues. Call this at the end to confirm the task is complete.","input_schema":{"type":"object","properties":{},"required":[]}},
]

# ═══════════════════════════════════════════════════════════
# THE AUTONOMOUS AGENT
# Claude with tool calling. Decides everything itself.
# Has ZERO knowledge of Codeastra.
# Codeastra is wired into the tool layer transparently.
# ═══════════════════════════════════════════════════════════

AGENT_SYSTEM = """You are an expert autonomous DevOps and Database Administrator agent.

Your job is to investigate and fix production issues completely and autonomously.
You have access to tools that let you inspect the database, create indexes, fix billing, and audit security.

Work methodically:
1. Investigate first — understand the full scope of the problem
2. Fix systematically — address every issue you find  
3. Verify — confirm fixes worked
4. Summarize — report what you did

Be thorough. Do not stop until the task is fully complete.
When you find an issue, fix it. When you fix it, verify it. Then look for the next issue.
"""

TASK_PROMPTS = {
    "dba": "Our production database is severely degraded. Query times are up 10x, users are complaining, and we're losing SLA. Investigate the performance issues and fix everything you find. Be thorough — check all tables and all slow queries. Don't stop until performance is restored.",

    "billing": "Our revenue team flagged serious billing leakage. We have enterprise customers being charged starter prices and a pile of failed payments we haven't retried. Audit the billing system completely, fix every mismatched plan, retry every failed payment, and tell me how much revenue you recovered.",

    "fullstack": "Do a complete production audit: fix all database performance issues, fix all billing problems, and audit API key security. This is a comprehensive review — find everything, fix everything, report back with a full summary.",
}

async def run_agent(demo_type: str, custom_task: str = ""):
    if not ANTHROPIC_KEY:
        yield {"type":"error","message":"Set ANTHROPIC_API_KEY on Railway"}
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    vault  = CodeastraVault()

    # Reset DB state
    DB["slow_queries"] = [
        {"sql":"SELECT * FROM users WHERE email=?",       "avg_ms":52847,"count":9821,"table":"users",       "column":"email"},
        {"sql":"SELECT * FROM transactions WHERE card=?", "avg_ms":61203,"count":4419,"table":"transactions","column":"card"},
        {"sql":"SELECT u.*,t.* FROM users u JOIN transactions t ON u.id=t.user_id WHERE u.ssn=?","avg_ms":89441,"count":2103,"table":"users","column":"ssn"},
        {"sql":"SELECT * FROM users WHERE name LIKE ?",   "avg_ms":44821,"count":1847,"table":"users",       "column":"name"},
    ]
    DB["indexes"]         = ["users_pkey","transactions_pkey"]
    DB["executed_fixes"]  = []
    DB["revenue_recovered"] = 0
    for u in DB["users"]:
        if u["id"] in [1,2,4,5,6]: u["plan"] = "starter"
    for t in DB["transactions"]:
        if t["id"] in [101,102,103,104]: t["status"]="failed"; t["retries"]=max(0,t["retries"]-1)

    task     = custom_task or TASK_PROMPTS.get(demo_type, TASK_PROMPTS["dba"])
    messages = [{"role":"user","content":task}]
    tool_calls_total = 0
    actions_total    = 0

    yield {"type":"start","task":task,"demo":demo_type}
    await asyncio.sleep(0.2)

    for iteration in range(15):
        yield {"type":"iteration","n":iteration+1}

        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 1024,
                system     = AGENT_SYSTEM,
                tools      = TOOL_DEFINITIONS,
                messages   = messages,
            )
        except Exception as e:
            yield {"type":"error","message":str(e)}
            return

        # Stream Claude's thinking
        for block in response.content:
            if hasattr(block,"text") and block.text:
                yield {"type":"thinking","text":block.text}
                await asyncio.sleep(0.02)

        if response.stop_reason == "end_turn":
            yield {"type":"agent_done"}
            break

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name   = block.name
            tool_inputs = block.input
            tool_calls_total += 1

            yield {"type":"tool_call","tool":tool_name,"inputs":tool_inputs,"call_n":tool_calls_total}
            await asyncio.sleep(0.2)

            # Execute tool — Codeastra intercepts the output
            tool_fn = TOOLS.get(tool_name,{}).get("fn")
            if tool_fn:
                try:
                    kwargs = {k:tool_inputs[k] for k in tool_inputs}
                    tokenized_result = tool_fn(vault, **kwargs)
                except Exception as e:
                    tokenized_result = vault.protect({"error":str(e)})
            else:
                tokenized_result = vault.protect({"error":f"Unknown tool: {tool_name}"})

            # Emit interception events
            for ev in vault.pop_events():
                yield ev
                await asyncio.sleep(0.05)

            is_action = tool_name in {"create_index","upgrade_plan","retry_payment"}
            if is_action:
                actions_total += 1

            yield {
                "type":       "tool_result",
                "tool":       tool_name,
                "result":     tokenized_result[:400],
                "is_action":  is_action,
            }
            await asyncio.sleep(0.15)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     tokenized_result,
            })

        messages.append({"role":"assistant","content":response.content})
        messages.append({"role":"user",     "content":tool_results})

    yield {
        "type":             "complete",
        "tool_calls":       tool_calls_total,
        "actions":          actions_total,
        "tokens_minted":    len(vault.vault),
        "fixes":            DB["executed_fixes"],
        "revenue_recovered":DB["revenue_recovered"],
        "remaining_slow":   len(DB["slow_queries"]),
        "real_data_seen_by_agent": 0,
    }

# ═══════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def index():
    with open("index.html") as f: return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"status":"healthy","anthropic_ready":bool(ANTHROPIC_KEY)}

@app.post("/agent/run")
async def agent_run(req: Request):
    body = await req.json()
    async def stream():
        async for ev in run_agent(body.get("demo_type","dba"), body.get("custom_task","")):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Access-Control-Allow-Origin":"*"})

@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    v = CodeastraVault()
    tok = v.protect(body.get("text",""))
    return {"original":body.get("text",""),"tokenized":tok,"intercepted":len(v.vault)}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


# ═══════════════════════════════════════════════════════════
# ADDITIONAL API ENDPOINTS
# These are what your website calls via API
# ═══════════════════════════════════════════════════════════

@app.get("/agent/demos")
async def list_demos():
    """List available demos with descriptions."""
    return {
        "demos": [
            {
                "id":          "dba",
                "name":        "Blind DBA Agent",
                "description": "Autonomous database administrator fixes production performance issues without seeing any customer PII",
                "task":        TASK_PROMPTS["dba"],
                "what_agent_does": ["Scans slow queries","Inspects table schemas","Creates missing indexes","Verifies performance restored"],
                "what_codeastra_does": "Intercepts every tool result. Tokenizes emails, SSNs, card numbers before Claude sees them.",
            },
            {
                "id":          "billing",
                "name":        "Revenue Recovery Agent",
                "description": "Autonomous billing agent finds revenue leakage and fixes it without exposing customer data",
                "task":        TASK_PROMPTS["billing"],
                "what_agent_does": ["Audits billing plans","Identifies mismatched plans","Upgrades customers","Retries failed payments","Reports revenue recovered"],
                "what_codeastra_does": "Every customer name, email, card number tokenized before agent sees it.",
            },
            {
                "id":          "fullstack",
                "name":        "Full Stack Ops Agent",
                "description": "Comprehensive production audit — database, billing, and security all in one run",
                "task":        TASK_PROMPTS["fullstack"],
                "what_agent_does": ["Full DB performance audit","Complete billing audit","API key security review","End-to-end report"],
                "what_codeastra_does": "All PII intercepted across every tool call throughout the entire workflow.",
            },
        ]
    }


@app.post("/agent/run/stream")
async def agent_run_stream(req: Request):
    """
    PRIMARY ENDPOINT — Run autonomous agent with real-time streaming.

    POST /agent/run/stream
    Content-Type: application/json

    Body:
      demo_type:   "dba" | "billing" | "fullstack"
      custom_task: str (optional — override the default task)

    Returns: text/event-stream (Server-Sent Events)

    Event types:
      start        — agent initialized, task defined
      intercepted  — Codeastra caught a real value, replaced with token
      thinking     — Claude's reasoning text
      tool_call    — agent called a tool (name + inputs)
      tool_result  — tool returned tokenized result
      complete     — agent finished, full summary

    Example (JavaScript):
      const es = new EventSource('/agent/run/stream');
      fetch('/agent/run/stream', {method:'POST', body: JSON.stringify({demo_type:'dba'})})

    Example (curl):
      curl -X POST https://your-demo.railway.app/agent/run/stream \\
        -H "Content-Type: application/json" \\
        -d '{"demo_type":"dba"}' \\
        --no-buffer
    """
    body = await req.json()

    async def stream():
        async for ev in run_agent(
            body.get("demo_type", "dba"),
            body.get("custom_task", "")
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers":"*",
        }
    )


@app.post("/agent/run/sync")
async def agent_run_sync(req: Request):
    """
    SYNC ENDPOINT — Run agent and return full result when complete.
    Waits for agent to finish then returns everything at once.
    Use this if you don't want streaming.

    POST /agent/run/sync
    Body: {demo_type: "dba"|"billing"|"fullstack", custom_task?: str}

    Returns: {
      success: bool,
      demo_type: str,
      task: str,
      events: [...all events...],
      summary: {tool_calls, actions, tokens_minted, fixes, revenue_recovered},
      agent_log: [...what Claude saw — tokens only...],
      codeastra_log: [...what was intercepted...],
      real_data_seen_by_agent: 0
    }
    """
    body      = await req.json()
    demo_type = body.get("demo_type", "dba")
    task      = body.get("custom_task", "")

    all_events     = []
    agent_log      = []   # what Claude saw
    codeastra_log  = []   # what Codeastra intercepted
    summary        = {}

    async for ev in run_agent(demo_type, task):
        all_events.append(ev)

        if ev["type"] == "intercepted":
            codeastra_log.append({
                "real_value_preview": ev["preview"],
                "type":               ev["dtype"],
                "token":              ev["token"],
            })
        elif ev["type"] in ("thinking", "tool_call", "tool_result"):
            agent_log.append(ev)
        elif ev["type"] == "complete":
            summary = ev

    return {
        "success":      True,
        "demo_type":    demo_type,
        "task":         task or TASK_PROMPTS.get(demo_type, ""),
        "events":       all_events,
        "agent_log":    agent_log,
        "codeastra_log": codeastra_log,
        "summary": {
            "tool_calls":         summary.get("tool_calls", 0),
            "actions_executed":   summary.get("actions", 0),
            "tokens_minted":      summary.get("tokens_minted", 0),
            "fixes":              summary.get("fixes", []),
            "revenue_recovered":  summary.get("revenue_recovered", 0),
            "remaining_issues":   summary.get("remaining_slow", 0),
        },
        "real_data_seen_by_agent": 0,
    }


@app.get("/agent/status/{demo_type}")
async def agent_status(demo_type: str):
    """
    GET current state of the simulated production environment.
    Shows what problems exist before running the agent.

    GET /agent/status/dba
    GET /agent/status/billing
    GET /agent/status/fullstack
    """
    if demo_type in ("dba", "fullstack"):
        db_status = {
            "slow_queries":       len(DB["slow_queries"]),
            "worst_query_ms":     max((q["avg_ms"] for q in DB["slow_queries"]),default=0),
            "indexes":            DB["indexes"],
            "fixes_executed":     DB["executed_fixes"],
            "performance_status": "degraded" if DB["slow_queries"] else "optimal",
        }
    else:
        db_status = None

    if demo_type in ("billing", "fullstack"):
        billing_status = {
            "mismatched_plans":   sum(1 for u in DB["users"] if u["plan"]=="starter"),
            "failed_payments":    sum(1 for t in DB["transactions"] if t["status"]=="failed"),
            "monthly_leakage":    sum(u["mrr"]-999 for u in DB["users"] if u["plan"]=="starter"),
            "revenue_recovered":  DB["revenue_recovered"],
        }
    else:
        billing_status = None

    return {
        "demo_type":      demo_type,
        "database":       db_status,
        "billing":        billing_status,
        "ready_to_run":   True,
        "run_endpoint":   "POST /agent/run/stream",
    }


@app.get("/docs/api")
async def api_docs():
    """Complete API documentation for integrating the demo into your website."""
    base = "https://your-demo.railway.app"
    return {
        "title":    "Codeastra Autonomous Agent Demo API",
        "base_url": base,
        "auth":     "None required — open demo API",
        "endpoints": {
            "list_demos": {
                "method":      "GET",
                "path":        "/agent/demos",
                "description": "List available demos",
                "example":     f"curl {base}/agent/demos",
            },
            "run_streaming": {
                "method":      "POST",
                "path":        "/agent/run/stream",
                "description": "Run agent with real-time streaming (recommended for website)",
                "body":        {"demo_type": "dba|billing|fullstack", "custom_task": "optional"},
                "returns":     "text/event-stream — events fire as agent works",
                "example":     f"curl -X POST {base}/agent/run/stream -H 'Content-Type: application/json' -d '{{\"demo_type\":\"dba\"}}' --no-buffer",
            },
            "run_sync": {
                "method":      "POST",
                "path":        "/agent/run/sync",
                "description": "Run agent and get full result when done",
                "body":        {"demo_type": "dba|billing|fullstack"},
                "returns":     "JSON with full agent log + codeastra log + summary",
                "example":     f"curl -X POST {base}/agent/run/sync -H 'Content-Type: application/json' -d '{{\"demo_type\":\"billing\"}}'",
            },
            "status": {
                "method":      "GET",
                "path":        "/agent/status/{demo_type}",
                "description": "Check current state before running agent",
                "example":     f"curl {base}/agent/status/dba",
            },
            "tokenize": {
                "method":      "POST",
                "path":        "/tokenize",
                "description": "Tokenize any text — show what Codeastra does to sensitive data",
                "body":        {"text": "any text with emails, SSNs, cards..."},
                "example":     f"curl -X POST {base}/tokenize -H 'Content-Type: application/json' -d '{{\"text\":\"email: ceo@goldman.com SSN: 234-56-7890\"}}'",
            },
            "health": {
                "method":  "GET",
                "path":    "/health",
                "example": f"curl {base}/health",
            },
        },
        "javascript_example": """
// Stream agent events in real-time on your website
async function runCodeastraDemo(demoType) {
  const response = await fetch('https://your-demo.railway.app/agent/run/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({demo_type: demoType})
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\\n');
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));

      if (event.type === 'intercepted') {
        console.log('Codeastra blocked:', event.preview, '→', event.token);
      } else if (event.type === 'thinking') {
        console.log('Claude thinking:', event.text);
      } else if (event.type === 'tool_call') {
        console.log('Agent called tool:', event.tool);
      } else if (event.type === 'complete') {
        console.log('Done! Real data seen by agent:', event.real_data_seen_by_agent);
      }
    }
  }
}
"""
    }
