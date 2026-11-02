"""S0 naive baseline: keep the full history, drop the oldest turns when over budget."""
from agent.strategies import estimate_tokens


def prepare_context(history, budget):
    """history[0] is the task; later entries are turns (an assistant message plus its tool results).
    Drops whole turns from the front so no tool result is left without its call."""
    task, rest = history[:1], history[1:]
    starts = [i for i, m in enumerate(rest) if m["role"] == "assistant"]
    for s in starts:
        if estimate_tokens(task + rest[s:]) <= budget:
            return task + rest[s:]
    return task + rest[starts[-1]:] if starts else task  # newest turn alone is still too big: send it anyway
