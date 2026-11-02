"""Offline checks (no model needed): python -m unittest"""
import json
import re
import unittest
from unittest import mock

from agent.strategies import s0_naive
from agent.tools import call_tool, search_docs
from graders.programmatic import extract_answer, grade
from tasks import archive_gen
from tasks.archive_gen import SPLITS, generate


def estimate(messages):  # offline stand-in for the vLLM token counter
    return sum(len(json.dumps(m)) for m in messages) // 4


class Generator(unittest.TestCase):
    def test_reachable_and_sized(self):
        for seed in range(20):
            a = generate(n_docs=50, n_questions=10, seed=seed)  # asserts every answer is in its support docs
            self.assertEqual(len(a["docs"]), 50)
            self.assertEqual(len(a["questions"]), 10)

    def test_deterministic(self):
        self.assertEqual(generate(seed=3), generate(seed=3))

    def test_too_few_docs(self):
        with self.assertRaises(ValueError):
            generate(n_docs=5, n_questions=10)

    def test_chain_walkable_by_search(self):
        """Hop i's support doc is in the top 5 when searching hop i-1's answer: the dial is real."""
        for seed in range(10):
            a = generate(seed=seed)
            self.assertEqual({q["hops"] for q in a["questions"]}, {1, 2, 3, 4, 5, 6})
            for q in a["questions"]:
                self.assertEqual(len(q["support"]), q["hops"])
                self.assertEqual(len(q["chain"]), q["hops"])
                self.assertEqual(q["chain"][-1], q["answer"])
                proj = q["question"].rstrip("?").split()[-1]
                queries = [f"lead {proj}"] + q["chain"][:-1]
                for query, doc in zip(queries, q["support"]):
                    self.assertIn(doc, search_docs(a["docs"], query), (seed, q["id"], query))

    def test_padding_is_generic(self):
        """Padding must not plant facts or steal search hits: no names, cities, projects or relation words."""
        banned = set(archive_gen.FIRST + archive_gen.LAST + archive_gen.CITIES + archive_gen.PROJECTS)
        banned |= {"lead", "leads", "office", "based", "reports", "manager", "project", "city"}
        for sentence in archive_gen.PADDING:
            words = set(re.findall(r"[A-Za-z]+", sentence))
            self.assertFalse(words & banned, sentence)

    def test_docs_are_long(self):
        a = generate(seed=0)
        self.assertTrue(all(len(d["text"]) > 1000 for d in a["docs"]))

    def test_splits_disjoint_20_80(self):
        self.assertFalse(set(SPLITS["dev"]) & set(SPLITS["test"]))
        self.assertEqual(len(SPLITS["dev"]) * 4, len(SPLITS["test"]))


class Grader(unittest.TestCase):
    def test_cases(self):
        self.assertTrue(grade("I searched.\nANSWER: Ava Abara", "Ava Abara"))
        self.assertTrue(grade("answer: **the Lisbon office**", "Lisbon"))
        self.assertFalse(grade("ANSWER: Ben Brandt", "Ava Abara"))
        self.assertFalse(grade("Ava Abara", "Ava Abara"))  # no marker = not answered
        self.assertFalse(grade(None, "Ava Abara"))
        self.assertEqual(extract_answer("ANSWER:"), "")


class Tools(unittest.TestCase):
    def test_bad_calls_return_errors(self):
        self.assertTrue(call_tool("search_docs", "{bad", []).startswith("Error"))
        self.assertTrue(call_tool("rm_rf", "{}", []).startswith("Error"))


class NaiveStrategy(unittest.TestCase):
    def history(self, turns):
        h = [{"role": "user", "content": "q"}]
        for i in range(turns):
            h.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
            h.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400})
        return h

    def setUp(self):
        patcher = mock.patch.object(s0_naive, "count_tokens", estimate)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fits_and_keeps_pairs(self):
        h = self.history(20)
        out = s0_naive.prepare_context(h, 600)
        self.assertLessEqual(estimate(out), 600)
        self.assertEqual(out[0], h[0])  # the task is never dropped
        self.assertEqual(out[1]["role"], "assistant")  # no orphaned tool result
        self.assertEqual(out[-1], h[-1])  # newest turn kept

    def test_no_trim_when_it_fits(self):
        h = self.history(2)
        self.assertEqual(s0_naive.prepare_context(h, 10_000), h)


class Summary(unittest.TestCase):
    def test_cells_and_trim_detection(self):
        from analysis.summarize import summarize

        def run(correct, sent, hist):
            return {"budget": 8192, "task": {"hops": 2}, "correct": correct, "max_prompt_tokens": 100,
                    "calls": [{"sent_messages": sent, "history_messages": hist}]}
        cell = summarize([run(True, 3, 2), run(False, 2, 4)])[(8192, 2)]  # untrimmed pass, trimmed fail
        self.assertEqual((cell["n"], cell["correct"], cell["trimmed"], cell["trimmed_correct"]), (2, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
