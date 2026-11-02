"""Run strategy x budget x trial over a family A archive and save every transcript.

    python -m runner.run_grid --strategy s0 --budgets 8192 16384 --trials 1
"""
import argparse
import datetime
import json
import pathlib

from agent import strategies
from agent.loop import CONFIG, client, run_task
from graders.programmatic import extract_answer, grade
from tasks.archive_gen import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="s0")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    archive = generate(n_docs=args.n_docs, n_questions=args.n_questions, seed=args.seed)
    prepare = strategies.get(args.strategy)
    llm = client()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path("results") / f"{stamp}_A_{args.strategy}.jsonl"
    out.parent.mkdir(exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "config", **CONFIG, **vars(args)}) + "\n")
        for budget in args.budgets:
            passed = total = 0
            for trial in range(args.trials):
                for q in archive["questions"]:
                    r = run_task(llm, q["question"], archive["docs"], prepare, budget)
                    ok = grade(r["answer"], q["answer"])
                    passed, total = passed + ok, total + 1
                    f.write(json.dumps({"type": "run", "family": "A", "strategy": args.strategy, "budget": budget,
                                        "trial": trial, "task": q, "extracted": extract_answer(r["answer"]),
                                        "correct": ok, **r}) + "\n")
                    print(f"{'PASS' if ok else 'FAIL'} b={budget} t={trial} {q['id']} hops={q['hops']} "
                          f"gold={q['answer']!r} got={extract_answer(r['answer'])!r} "
                          f"turns={r['turns']} max_prompt={r['max_prompt_tokens']}")
            print(f"== {args.strategy} budget {budget}: {passed}/{total} correct")
    print(f"transcripts: {out}")


if __name__ == "__main__":
    main()
