"""Live check (needs vLLM): does the served model make valid search_docs tool calls?

    python -m tests.toolcall_check [base_url]
"""
import json, sys, urllib.request

from agent.tools import TOOLS

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1"
MODEL = "local-model"


PROMPTS = [
    "Who approved the Q3 vendor contract with Northwind?",
    "When was the database migration ticket closed?",
    "What did the team decide about the office move in the March planning meeting?",
    "Which engineer owns the billing service?",
    "Find the email where finance flagged the travel budget overrun.",
    "What was the root cause of the outage on the 14th?",
    "Who replaced Dana as project lead for Atlas?",
    "How many laptops were ordered for the new hires?",
    "What deadline was agreed for the security audit?",
    "Did anyone object to the pricing change? Who?",
    "Which customer complained about late invoices most often?",
    "What is the current status of ticket OPS-2291?",
    "Summarize what was said about hiring freezes in the leadership notes.",
    "Which vendor did legal reject, and why?",
    "Who was on call when the payment queue backed up?",
]


def ask(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 1024, "tools": TOOLS,
                       "tool_choice": "auto",
                       "messages": [{"role": "system", "content": "Answer using the search_docs tool. Do not guess."},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]


def valid(msg):
    """Return (ok, reason). Valid = at least one search_docs call whose args are JSON with a non-empty query."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, "no tool call: " + (msg.get("content") or "")[:80].replace("\n", " ")
    for c in calls:
        fn = c.get("function", {})
        if fn.get("name") != "search_docs":
            return False, f"wrong tool {fn.get('name')!r}"
        try:
            args = json.loads(fn.get("arguments") or "")
        except json.JSONDecodeError:
            return False, f"bad JSON args {fn.get('arguments')!r}"
        if not isinstance(args.get("query"), str) or not args["query"].strip():
            return False, f"missing query {args!r}"
    return True, args["query"]


# self-check of the grader before spending model calls
assert valid({"tool_calls": [{"function": {"name": "search_docs", "arguments": '{"query": "x"}'}}]})[0]
assert not valid({"tool_calls": [{"function": {"name": "search_docs", "arguments": "{bad"}}]})[0]
assert not valid({"content": "I think..."})[0]

ok = 0
for p in PROMPTS:
    good, why = valid(ask(p))
    ok += good
    print(("PASS " if good else "FAIL ") + p[:50].ljust(52) + why)
print(f"\n{ok}/{len(PROMPTS)} valid tool calls")
