"""Exact-match grading for tasks with planted answers (families A and C)."""
import re


def extract_answer(text):
    """The text after the last 'ANSWER:' marker, or None if the agent never gave one."""
    parts = re.split(r"ANSWER:", text or "", flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    lines = parts[-1].strip().splitlines()
    return lines[0].strip() if lines else ""


def normalize(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(the|office)\b", " ", s)
    return " ".join(s.split())


def grade(final_text, gold):
    ans = extract_answer(final_text)
    return ans is not None and normalize(ans) == normalize(gold)
