# Context Under Pressure

An eval suite for context strategies in long-horizon agents.

**Question:** as agent tasks outgrow the context window, which context strategies keep working, what do they cost, and how do they fail?

Anthropic's post "Effective context engineering for AI agents" recommends just-in-time retrieval, compaction, structured note-taking and sub-agents for long tasks. This project turns those recommendations into testable claims. It runs each strategy on three task families under increasing context pressure and reports success rate, token cost and a labelled failure taxonomy. The headline result is a set of degradation curves: success rate against task length, one line per strategy.

## Research questions

1. **Degradation:** How fast does task success fall as required context grows, for each strategy?
2. **Cost:** What does each strategy cost in tokens, turns and latency for the success it buys?
3. **Failure modes:** How do strategies fail differently? For example, does compaction drop key facts while note-taking drifts off goal?
4. **Task fit:** Does the best strategy depend on the task type (retrieval-heavy vs. memory-heavy)?
5. **Context rot:** Holding information fixed, does adding irrelevant context measurably hurt performance?

## Hypotheses

Written down before any experiments were run.

| # | Hypothesis | How it could be wrong |
|---|---|---|
| H1 | The naive baseline degrades sharply once required context passes ~50% of the budget | Degradation is gradual, or starts much earlier |
| H2 | Note-taking beats compaction on memory-heavy tasks | Compaction summaries keep the key state just as well |
| H3 | Sub-agents win on retrieval-heavy tasks but cost 2–4x more tokens | The token overhead outweighs the gains |
| H4 | Tool-result clearing gives most of compaction's benefit at a fraction of the cost | Old tool outputs are needed later more often than expected |
| H5 | Adding irrelevant documents lowers accuracy even when all needed facts fit | Modern models ignore distractors well at these sizes |

Thresholds and multipliers are starting guesses, to be adjusted after a small pilot run.

## Setup

**Model:** `Qwen/Qwen3-8B-AWQ` served locally by vLLM's OpenAI-compatible API (thinking off, temperature 0.7, top_p 0.8; all settings in `agent/loop.py` and logged with every run).

```sh
docker run -d --name vllm --gpus all -p 8001:8000 vllm/vllm-openai:latest \
  Qwen/Qwen3-8B-AWQ --served-model-name local-model --max-model-len 32768 \
  --gpu-memory-utilization 0.88 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3

python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; bin/ on Linux/macOS
```

## Run

```sh
python -m unittest                  # offline checks: generator reachability, grader, S0 trimming
python -m tests.toolcall_check      # live: does the model make valid tool calls?
python -m runner.run_grid --split dev --strategy s0 --budgets 8192 16384   # tune prompts on dev only
python -m runner.run_grid --split test --strategy s0 --budgets 8192 16384
python -m analysis.summarize results/<file>.jsonl                          # accuracy by budget x pad, trimming
```

Each run writes `results/<timestamp>_A_<strategy>.jsonl`: a config line, then one line per task with the full transcript, answer, grade and token counts.

## Layout

```
tasks/archive_gen.py        family A: org archive; answer = lead city + manager's city (needs the early fact); dial = doc length (pad sentences)
agent/loop.py               model + tool loop
agent/tools.py              search_docs (IDF keyword search)
agent/strategies/           one prepare_context(history, budget) per strategy; s0_naive.py so far
runner/run_grid.py          split x strategy x budget x trial
graders/programmatic.py     exact match on planted answers
tests/
```

## Plan

| Strategy | What it does |
|---|---|
| S0 Naive baseline | Keeps full history; drops the oldest turns when the budget is full |
| S1 Tool-result clearing | Replaces tool outputs older than N turns with a short stub |
| S2 Compaction | Summarizes the history at 70% of budget, then continues from the summary |
| S3 Note-taking | Read/write tools for a persistent `notes.md` outside the context |
| S4 Sub-agents | An orchestrator delegates exploration to sub-agents returning ≤ 500-token summaries |
| S5 Pre-loaded context | All candidate documents in the prompt up front, where they fit |

Budgets: 8k and 16k tokens. Task families: A (synthetic org archive, built first), B (codebase investigation), C (stateful simulation).

**Status:** S0 on family A v1 (pure lookup chains, hops 1-6) showed no budget effect on test: chains are memoryless, so dropped turns were never needed. v2 asks for two cities, the first found early, with 3 hops fixed and doc length as the dial (Qwen3-8B loses its way on longer chains even with full context). Dev: 16k flat at ~75% across doc lengths; 8k collapses once the task outgrows the budget, with S0 re-running the same search after losing its context.
