"""Offline checks (no model needed): python -m unittest"""
import unittest

from agent.strategies import estimate_tokens, s0_naive
from agent.tools import call_tool, search_docs
from graders.programmatic import extract_answer, grade
from tasks.archive_gen import generate


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

    def test_answer_findable_by_search(self):
        a = generate(seed=1)
        for q in a["questions"]:
            proj = q["question"].rstrip("?").split()[-1]
            self.assertIn(q["support"][0], search_docs(a["docs"], f"lead {proj}", k=10))


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

    def test_fits_and_keeps_pairs(self):
        h = self.history(20)
        out = s0_naive.prepare_context(h, 600)
        self.assertLessEqual(estimate_tokens(out), 600)
        self.assertEqual(out[0], h[0])  # the task is never dropped
        self.assertEqual(out[1]["role"], "assistant")  # no orphaned tool result
        self.assertEqual(out[-1], h[-1])  # newest turn kept

    def test_no_trim_when_it_fits(self):
        h = self.history(2)
        self.assertEqual(s0_naive.prepare_context(h, 10_000), h)


if __name__ == "__main__":
    unittest.main()
