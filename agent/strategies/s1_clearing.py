"""S1 tool-result clearing: when over budget, replace the oldest tool results with a stub, keeping every
call (so the model still sees what it searched for) and the newest turn's results. If the stubbed history
still does not fit, fall back to S0 and drop the oldest turns."""
from agent.strategies import count_tokens, s0_naive

STUB = "[Old tool result cleared to save context. Search again if you need it.]"


def prepare_context(history, budget):
    if count_tokens(history) <= budget:
        return history
    out = list(history)  # replace entries, never edit the caller's history
    newest = max((i for i, m in enumerate(out) if m["role"] == "assistant"), default=len(out))
    for i in range(newest):
        if out[i]["role"] == "tool":
            out[i] = {**out[i], "content": STUB}
            if count_tokens(out) <= budget:
                return out
    return s0_naive.prepare_context(out, budget)
