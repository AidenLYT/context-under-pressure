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
            a = generate(n_docs=150, n_questions=10, seed=seed)  # asserts every answer is in its support docs
            self.assertEqual(len(a["docs"]), 150)
            self.assertEqual(len(a["questions"]), 10)

    def test_deterministic(self):
        self.assertEqual(generate(seed=3), generate(seed=3))

    def test_too_few_docs(self):
        with self.assertRaises(ValueError):
            generate(n_docs=5, n_questions=10)

    def test_chain_walkable_by_search(self):
        """Each step's docs are in the top 5 for that step's query, and the query is only learnable from
        the previous step: hops = searches needed, so the dial is real."""
        for seed in range(10):
            a = generate(seed=seed, hops=(3, 4, 5, 6, 7, 8))
            self.assertEqual({q["hops"] for q in a["questions"]}, {3, 4, 5, 6, 7, 8})
            for q in a["questions"]:
                self.assertEqual(len(q["steps"]), q["hops"])
                for step in q["steps"]:
                    found = search_docs(a["docs"], step["query"])
                    for doc in step["docs"]:
                        self.assertIn(doc, found, (seed, q["id"], step["query"]))

    def test_answer_needs_the_early_fact(self):
        """The first city comes from step 2 (the lead's search); the second from the last step."""
        a = generate(seed=0)
        by_id = {d["id"]: d["text"] for d in a["docs"]}
        for q in a["questions"]:
            first, second = q["answer"].split("; ")
            self.assertTrue(any(f"based in the {first} office" in by_id[d] for d in q["steps"][1]["docs"]))
            self.assertTrue(any(f"based in the {second} office" in by_id[d] for d in q["steps"][-1]["docs"]))

    def test_padding_is_generic(self):
        """Padding must not plant facts or steal search hits: no names, cities, projects or relation words."""
        banned = set(archive_gen.FIRST + archive_gen.LAST + archive_gen.CITIES + archive_gen.PROJECTS)
        banned |= {"lead", "leads", "office", "based", "reports", "manager", "project", "city"}
        for sentence in archive_gen.PADDING:
            words = set(re.findall(r"[A-Za-z]+", sentence))
            self.assertFalse(words & banned, sentence)

    def test_pad_level_changes_only_doc_length(self):
        """The dial is doc length: same questions and answers at every level, longer docs."""
        short, long = generate(seed=4, pad_sentences=10), generate(seed=4, pad_sentences=80)
        strip = lambda qs: [(q["question"], q["answer"], q["steps"]) for q in qs]
        self.assertEqual(strip(short["questions"]), strip(long["questions"]))
        self.assertEqual({q["pad"] for q in long["questions"]}, {80})
        self.assertGreater(min(len(d["text"]) for d in long["docs"]), 4 * max(len(d["text"]) for d in short["docs"]))
        self.assertEqual({q["hops"] for q in long["questions"]}, {3})  # default: fixed short chain
        for q in long["questions"]:
            for step in q["steps"]:
                for doc in step["docs"]:
                    self.assertIn(doc, search_docs(long["docs"], step["query"]))

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

    def test_two_city_answers(self):
        self.assertTrue(grade("ANSWER: Lisbon; Osaka", "Lisbon; Osaka"))
        self.assertTrue(grade("ANSWER: the Lisbon office and the Osaka office", "Lisbon; Osaka"))
        self.assertFalse(grade("ANSWER: Osaka; Lisbon", "Lisbon; Osaka"))  # order matters
        self.assertFalse(grade("ANSWER: Lisbon", "Lisbon; Osaka"))


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
            return {"budget": 8192, "task": {"pad": 20}, "correct": correct, "max_prompt_tokens": 100,
                    "calls": [{"sent_messages": sent, "history_messages": hist}]}
        cell = summarize([run(True, 3, 2), run(False, 2, 4)])[(8192, 20)]  # untrimmed pass, trimmed fail
        self.assertEqual((cell["n"], cell["correct"], cell["trimmed"], cell["trimmed_correct"]), (2, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
