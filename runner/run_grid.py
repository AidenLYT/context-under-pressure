"""Run strategy x budget x trial over a family A archive and save every transcript.

    python -m runner.run_grid --split dev --strategy s0 --budgets 8192 16384 --trials 1
"""
import argparse
import datetime
import json
import pathlib

from agent import strategies
from agent.loop import CONFIG, client, run_task
from graders.programmatic import extract_answer, grade
from tasks.archive_gen import SPLITS, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="s0")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--split", choices=SPLITS, required=True, help="dev for prompt tuning; test for reported numbers")
    args = ap.parse_args()

    archives = [generate(seed=seed) for seed in SPLITS[args.split]]
    tasks = [(a, q) for a in archives for q in a["questions"]]
    prepare = strategies.get(args.strategy)
    llm = client()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path("results") / f"{stamp}_A_{args.strategy}.jsonl"
    out.parent.mkdir(exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "config", **CONFIG, **vars(args), "seeds": SPLITS[args.split]}) + "\n")
        for budget in args.budgets:
            passed = total = 0
            for trial in range(args.trials):
                for archive, q in tasks:
                    r = run_task(llm, q["question"], archive["docs"], prepare, budget)
                    ok = grade(r["answer"], q["answer"])
                    passed, total = passed + ok, total + 1
                    f.write(json.dumps({"type": "run", "family": "A", "strategy": args.strategy, "budget": budget,
                                        "trial": trial, "task": q, "extracted": extract_answer(r["answer"]),
                                        "correct": ok, **r}) + "\n")
                    print(f"{'PASS' if ok else 'FAIL'} b={budget} t={trial} {q['id']} hops={q['hops']} "
                          f"gold={q['answer']!r} got={extract_answer(r['answer'])!r} "
                          f"turns={r['turns']} max_prompt={r['max_prompt_tokens']} stop={r['stop']}")
            print(f"== {args.strategy} budget {budget}: {passed}/{total} correct")
    print(f"transcripts: {out}")


if __name__ == "__main__":
    main()
