"""Tools the agent can call. Every strategy gets the same ones (strategies may add their own)."""
import json
import math
import re

TOOLS = [{"type": "function", "function": {
    "name": "search_docs",
    "description": "Keyword search over the company archive (emails, tickets, meeting notes). "
                   "Returns up to 5 matching documents with their id, kind and date.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "Search terms"}},
                   "required": ["query"]}}}]

_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "who", "which", "what", "is", "was", "and", "project"}


def _terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def search_docs(docs, query, k=5):
    """Rank by summed IDF of matched query terms, so rare words (names, projects) beat common ones.
    Docs containing the exact query phrase rank first, so "Olga Mendes" beats "Olga Abara ... Ben Mendes"."""
    q = set(_terms(query))
    phrase = " ".join(query.lower().split())
    doc_terms = [set(_terms(d["text"])) for d in docs]
    idf = {t: math.log(len(docs) / n) for t in q if (n := sum(t in dt for dt in doc_terms))}
    scored = []
    for d, dt in zip(docs, doc_terms):
        score = sum(idf.get(t, 0) for t in q & dt)
        if score > 0 and phrase in d["text"].lower():
            score += 1000
        if score > 0:
            scored.append((-score, d["id"], d))
    hits = [d for _, _, d in sorted(scored)[:k]]
    if not hits:
        return "No documents matched."
    return "\n".join(f"[{d['id']} | {d['kind']} | {d['date']}] {d['text']}" for d in hits)


def call_tool(name, arguments, docs):
    """Run one tool call from the model. Bad calls return an error string the model can see."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "Error: arguments were not valid JSON."
    if name == "search_docs" and isinstance(args.get("query"), str):
        return search_docs(docs, args["query"])
    return f"Error: unknown tool or missing arguments: {name}({arguments})"
