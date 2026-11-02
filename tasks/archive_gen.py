"""Family A: a synthetic org archive with planted facts, contradictions and distractors.

Difficulty dial: n_docs (more distractor documents) and hops (1 = who leads a project,
2 = which city that lead is based in).
"""
import datetime
import random

FIRST = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonah",
         "Kira", "Liam", "Maya", "Nikhil", "Olga", "Pavel", "Quinn", "Rosa", "Sami", "Tara"]
LAST = ["Abara", "Brandt", "Castillo", "Dubois", "Eriksen", "Fong", "Gallo", "Haddad", "Ivanova",
        "Jensen", "Kowalski", "Lindqvist", "Mendes", "Nakamura", "Okafor", "Petrov", "Quist",
        "Reyes", "Sato", "Thorne"]
PROJECTS = ["Atlas", "Beacon", "Cobalt", "Delta", "Ember", "Falcon", "Granite", "Harbor", "Iris",
            "Juniper", "Kestrel", "Lumen", "Mosaic", "Nimbus", "Orchid", "Pioneer", "Quartz",
            "Raven", "Summit", "Tundra", "Umbra", "Vertex", "Willow", "Zephyr"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Krakow", "Toronto", "Melbourne", "Austin",
          "Dublin", "Seoul"]
KINDS = ["email", "ticket", "meeting notes"]


def generate(n_docs=50, n_questions=10, hops=(1, 2), seed=0, p_correction=0.3):
    """Return {"docs": [{"id", "kind", "date", "text"}], "questions": [{"id", "question", "answer", "hops", "support"}]}."""
    rng = random.Random(seed)
    people = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], len(FIRST) * len(LAST))
    city_of = {p: rng.choice(CITIES) for p in people}
    asked = PROJECTS[:]
    rng.shuffle(asked)
    if n_questions > len(asked) - 4:
        raise ValueError(f"at most {len(asked) - 4} questions (need spare projects for decoys)")
    decoys, asked = asked[n_questions:], asked[:n_questions]

    facts, questions, used_people = [], [], set()  # facts: (key, day, text)

    def day():
        return rng.randint(1, 300)

    for qi, proj in enumerate(asked):
        owner = people[len(used_people)]
        used_people.add(owner)
        support = []
        if rng.random() < p_correction:
            old = people[len(used_people)]
            used_people.add(old)
            d_old = rng.randint(1, 150)
            facts.append((f"q{qi}-old", d_old, f"Agreed that {old} will lead project {proj}."))
            facts.append((f"q{qi}-owner", rng.randint(d_old + 1, 300),
                          f"Update: {owner} has taken over project {proj} from {old}, effective immediately."))
            facts.append((f"q{qi}-oldcity", day(), f"{old} is based in the {city_of[old]} office."))
        else:
            facts.append((f"q{qi}-owner", day(), f"Agreed that {owner} will lead project {proj}."))
        support.append(f"q{qi}-owner")
        h = hops[qi % len(hops)]
        if h == 1:
            q, a = f"Who currently leads project {proj}?", owner
        else:
            facts.append((f"q{qi}-city", day(), f"{owner} is based in the {city_of[owner]} office."))
            support.append(f"q{qi}-city")
            q, a = f"In which city is the office of the person who currently leads project {proj}?", city_of[owner]
        questions.append({"id": f"A-{seed}-{qi}", "question": q, "answer": a, "hops": h, "support": support})

    if n_docs < len(facts):
        raise ValueError(f"n_docs={n_docs} is below the {len(facts)} fact documents needed")
    others = [p for p in people if p not in used_people]
    for i in range(n_docs - len(facts)):
        p, p2 = rng.sample(others[:60], 2)
        filler = rng.choice([
            f"Agreed that {p} will lead project {rng.choice(decoys)}.",  # decoy project lead
            f"{p} is based in the {city_of[p]} office.",
            f"{p} visited the {rng.choice(CITIES)} office for the quarterly review.",
            f"Ticket OPS-{1000 + i}: {p} reported slow builds on project {rng.choice(asked)}.",
            f"{p} and {p2} demoed project {rng.choice(asked)} to the leadership team.",
            f"Reminder from {p}: expense reports for project {rng.choice(asked + decoys)} are due Friday.",
        ])
        facts.append((f"filler-{i}", day(), filler))

    rng.shuffle(facts)
    docs, key_to_id = [], {}
    for n, (key, d, text) in enumerate(facts):
        doc_id = f"doc-{n:04d}"
        key_to_id[key] = doc_id
        date = (datetime.date(2026, 1, 1) + datetime.timedelta(days=d)).isoformat()
        docs.append({"id": doc_id, "kind": rng.choice(KINDS), "date": date, "text": text})
    for q in questions:
        q["support"] = [key_to_id[k] for k in q["support"]]
    _check_reachable(docs, questions)
    return {"docs": docs, "questions": questions}


def _check_reachable(docs, questions):
    """Rigor check: every planted answer appears in its supporting documents."""
    by_id = {d["id"]: d for d in docs}
    for q in questions:
        assert any(q["answer"] in by_id[s]["text"] for s in q["support"]), q
