"""Accuracy by budget x hops from a run_grid JSONL, with how often the strategy trimmed history.

    python -m analysis.summarize results/<file>.jsonl
"""
import collections
import json
import sys


def trimmed(run):
    return any(c["sent_messages"] < c["history_messages"] + 1 for c in run["calls"])  # +1 = system prompt


def summarize(runs):
    """{(budget, hops): {"n", "correct", "trimmed", "trimmed_correct", "max_prompt"}}"""
    cells = collections.defaultdict(lambda: {"n": 0, "correct": 0, "trimmed": 0, "trimmed_correct": 0, "max_prompt": 0})
    for r in runs:
        c = cells[(r["budget"], r["task"]["hops"])]
        t = trimmed(r)
        c["n"] += 1
        c["correct"] += r["correct"]
        c["trimmed"] += t
        c["trimmed_correct"] += t and r["correct"]
        c["max_prompt"] = max(c["max_prompt"], r["max_prompt_tokens"])
    return dict(cells)


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    runs = [r for r in rows if r["type"] == "run"]
    mismatched = sum(c["counted_tokens"] != c["prompt_tokens"] for r in runs for c in r["calls"])
    print(f"{len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.1f} min, token-count mismatches: {mismatched}")
    print("budget hops  accuracy        trimmed(correct)  max_prompt")
    for (budget, hops), c in sorted(summarize(runs).items()):
        print(f"{budget:>6} {hops:>4}  {c['correct']:>3}/{c['n']:<3} {c['correct'] / c['n']:>5.0%}"
              f"    {c['trimmed']:>3} ({c['trimmed_correct']})       {c['max_prompt']:>6}")
    for r in runs:
        if not r["correct"]:
            print(f"FAIL b={r['budget']} t={r['trial']} {r['task']['id']} hops={r['task']['hops']} "
                  f"gold={r['task']['answer']!r} got={r['extracted']!r}")


if __name__ == "__main__":
    main(sys.argv[1])
