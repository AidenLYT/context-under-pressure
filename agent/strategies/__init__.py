"""Each strategy is prepare_context(history, budget) -> messages (system prompt excluded)."""
import json


def estimate_tokens(messages):
    # shortcut: ~4 chars/token estimate over the JSON; the real prompt_tokens is logged per call,
    # switch to vLLM's /tokenize if budgets need to be exact.
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4


def get(name):
    from agent.strategies import s0_naive
    return {"s0": s0_naive.prepare_context}[name]
