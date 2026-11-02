"""The agent loop: model call -> tool calls -> repeat, with context managed by a strategy."""
import time

from openai import OpenAI

from agent.tools import TOOLS, call_tool

SYSTEM_PROMPT = (
    "You answer questions about a company archive. Use the search_docs tool to find evidence; "
    "do not guess. Later documents can override earlier ones. Questions often need several searches: "
    "when a document gives you a person's name, search for that name to find the next fact, and do not "
    "answer 'Unknown' while there is a name you have not searched for. When you are done, end your reply "
    "with a final line of the form 'ANSWER: <answer>', giving only the name or city."
)

# Fixed settings, logged with every run (rigor checklist).
CONFIG = {
    "base_url": "http://localhost:8001/v1",
    "model": "local-model",  # Qwen/Qwen3-8B-AWQ in the vllm container
    "temperature": 0.7,  # Qwen3 non-thinking recommendation; trials are real repeats
    "top_p": 0.8,
    "max_output_tokens": 1024,
    "enable_thinking": False,  # Qwen3 thinking off: GPU time is the binding constraint
    "max_turns": 30,
    "system_prompt": SYSTEM_PROMPT,
}


def client(cfg=CONFIG):
    return OpenAI(base_url=cfg["base_url"], api_key="unused")


def run_task(llm, question, docs, prepare_context, budget, cfg=CONFIG):
    """Run one task. Returns the final answer (or None), the full transcript and usage numbers."""
    history = [{"role": "user", "content": question}]
    calls, start = [], time.time()
    from agent.strategies import count_tokens  # here, not at top: strategies import CONFIG from this module
    prompt_budget = budget - cfg["max_output_tokens"]  # the prompt plus the reply must fit the budget
    for _ in range(cfg["max_turns"]):
        context = prepare_context(history, prompt_budget)
        counted = count_tokens(context, cfg)
        messages = [{"role": "system", "content": cfg["system_prompt"]}] + context
        resp = llm.chat.completions.create(
            model=cfg["model"], messages=messages, tools=TOOLS, temperature=cfg["temperature"], top_p=cfg["top_p"],
            max_tokens=cfg["max_output_tokens"],
            extra_body={"chat_template_kwargs": {"enable_thinking": cfg["enable_thinking"]}})
        msg = resp.choices[0].message
        calls.append({"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                      "counted_tokens": counted, "sent_messages": len(messages),
                      "history_messages": len(history)})  # sent < history + 1 (system) means the strategy trimmed
        turn = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            turn["tool_calls"] = [{"id": tc.id, "type": "function",
                                   "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                  for tc in msg.tool_calls]
        history.append(turn)
        if not msg.tool_calls:
            return _result(msg.content, "answered", history, calls, start)
        for tc in msg.tool_calls:
            history.append({"role": "tool", "tool_call_id": tc.id,
                            "content": call_tool(tc.function.name, tc.function.arguments, docs)})
    return _result(None, "turn_limit", history, calls, start)


def _result(answer, stop, history, calls, start):
    return {"answer": answer, "stop": stop, "turns": len(calls), "seconds": round(time.time() - start, 2),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "max_prompt_tokens": max((c["prompt_tokens"] for c in calls), default=0),
            "calls": calls, "transcript": history}
