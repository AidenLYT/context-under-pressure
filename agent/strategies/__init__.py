"""Each strategy is prepare_context(history, budget) -> messages (system prompt excluded).
budget is the whole prompt the model sees: system prompt, tool schema and chat template included."""
import json
import urllib.request

from agent.loop import CONFIG
from agent.tools import TOOLS


def count_tokens(messages, cfg=CONFIG):
    """Exact prompt tokens from vLLM's /tokenize for the request run_task would send with these messages."""
    body = {"model": cfg["model"], "messages": [{"role": "system", "content": cfg["system_prompt"]}] + messages,
            "tools": TOOLS, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": cfg["enable_thinking"]}}
    url = cfg["base_url"].removesuffix("/v1") + "/tokenize"
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["count"]


def get(name):
    from agent.strategies import s0_naive, s1_clearing
    return {"s0": s0_naive.prepare_context, "s1": s1_clearing.prepare_context}[name]
