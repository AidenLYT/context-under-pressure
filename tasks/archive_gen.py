"""Family A: a synthetic org archive with planted facts, contradictions and distractors.

Every question asks for two cities: the project lead's, and that of a manager hops-2 levels above
them. The lead's city is found at the second search but only needed in the final answer, so a strategy
that drops old turns loses it (a pure chain lookup would not test that: each hop needs only the last).

Difficulty dial: pad_sentences, the routine text around each fact (~15 tokens per sentence), which sets
the tokens per search and so the context a task needs. Questions and answers are identical across pad
levels for a seed, so only context length changes. hops stays at 3 by default: Qwen3-8B without thinking
loses its way on longer chains even with full context, which would confound the budget effect.
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
SPLITS = {"dev": [0, 1], "test": [2, 3, 4, 5, 6, 7, 8, 9]}  # by archive seed; tune prompts on dev only
PAD_LEVELS = [20, 40, 50, 60]  # ~290 to ~830 tokens per doc; 3-search task ~3.4k to ~9.5k

# Routine filler around each fact. No names, cities, projects or relation words (tests enforce this).
PADDING = [
    "Thanks everyone for joining on short notice this morning.",
    "Please make sure your timesheets are submitted before the end of the week.",
    "The printer on the third floor is out of toner again, facilities have been notified.",
    "We will revisit the hiring plan once the budget numbers are final.",
    "Action items from last time were mostly closed, two are still open.",
    "The quarterly survey closes on Friday, so please fill it in if you have not yet.",
    "Remember that the parking garage will be closed for maintenance next Tuesday.",
    "Coffee and pastries will be available in the kitchen after the all-hands.",
    "The security team asks everyone to complete the phishing awareness training.",
    "Our sprint retrospective raised the usual points about meeting overload.",
    "Laptop refresh requests should go through the internal portal, not email.",
    "The new expense tool goes live next month and training sessions will be offered.",
    "Several people asked about the holiday schedule, which will be shared soon.",
    "Please keep customer data out of shared chat channels.",
    "The release checklist has been updated with two new review steps.",
    "We are still waiting on the vendor for revised pricing on the support contract.",
    "Interns start in two weeks and will need desks, badges and accounts.",
    "The wiki migration is about halfway done and some links may break for a while.",
    "Remember to book meeting rooms in advance during the busy season.",
    "The on-call rotation for next month has been posted on the team calendar.",
    "There was a short network outage overnight but service has been restored.",
    "Please update your emergency contact details in the employee portal.",
    "Finance reminds us that invoices older than ninety days need a second approval.",
    "The design review was moved to Thursday afternoon because of a conflict.",
    "We agreed to keep status updates short and post them in writing.",
    "The fire drill is scheduled for Wednesday at eleven, please do not use the elevators.",
    "Feedback on the onboarding guide has been positive overall.",
    "The test environment will be rebuilt over the weekend, expect some downtime.",
    "If you have not yet returned your old badge, please drop it at reception.",
    "Benefits enrollment opens next week and closes at the end of the month.",
    "A few teams reported slow video calls, IT is looking into the bandwidth.",
    "Let us aim to close out the remaining review comments by Monday.",
    "The style guide now covers naming conventions for shared folders.",
    "Please archive old documents rather than deleting them.",
    "Customer support volume was higher than usual after the last update.",
    "We will try a no-meeting Friday for the next month and see how it goes.",
    "Training budget requests for next year are due by the end of the quarter.",
    "The kitchen fridge will be cleared out on Friday afternoon.",
    "Remember to lock your screen when you step away from your desk.",
    "The team lunch has been rescheduled because of the public holiday.",
]


def generate(n_docs=150, n_questions=12, hops=(3,), seed=0, p_correction=0.3, pad_sentences=21):
    """Return {"docs": [{"id", "kind", "date", "text"}],
    "questions": [{"id", "question", "answer", "hops", "pad", "steps": [{"query", "docs"}]}]}."""
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

    def new_person():  # chain people go in used_people so filler never states facts about them
        p = people[len(used_people)]
        used_people.add(p)
        return p

    for qi, proj in enumerate(asked):
        owner = new_person()
        if rng.random() < p_correction:
            old = new_person()
            d_old = rng.randint(1, 150)
            facts.append((f"q{qi}-old", d_old, f"Agreed that {old} will lead project {proj}."))
            facts.append((f"q{qi}-owner", rng.randint(d_old + 1, 300),
                          f"Update: {owner} has taken over project {proj} from {old}, effective immediately."))
            facts.append((f"q{qi}-oldcity", day(), f"{old} is based in the {city_of[old]} office."))
        else:
            facts.append((f"q{qi}-owner", day(), f"Agreed that {owner} will lead project {proj}."))
        h = hops[qi % len(hops)]
        if h < 3:
            raise ValueError("hops must be >= 3: project, lead, then at least one manager")
        # Search steps: project -> lead (city + first manager) -> each manager -> top manager's city.
        facts.append((f"q{qi}-leadcity", day(), f"{owner} is based in the {city_of[owner]} office."))
        steps = [(f"project {proj}", [f"q{qi}-owner"]), (owner, [f"q{qi}-leadcity"])]
        chain = [owner]
        for m in range(h - 2):
            boss = new_person()
            facts.append((f"q{qi}-mgr{m}", day(), f"{chain[-1]} reports to {boss}."))
            steps[-1][1].append(f"q{qi}-mgr{m}")
            steps.append((boss, []))
            chain.append(boss)
        facts.append((f"q{qi}-topcity", day(), f"{chain[-1]} is based in the {city_of[chain[-1]]} office."))
        steps[-1][1].append(f"q{qi}-topcity")
        lead = f"the person who currently leads project {proj}"
        q = (f"In which city is the office of {lead}, and in which city is the office of "
             f"the {'manager of the ' * (h - 3)}manager of that person? Answer as '<first city>; <second city>'.")
        questions.append({"id": f"A-{seed}-{qi}-p{pad_sentences}", "question": q, "hops": h, "pad": pad_sentences,
                          "answer": f"{city_of[owner]}; {city_of[chain[-1]]}",
                          "steps": [{"query": query, "docs": keys} for query, keys in steps]})

    if n_docs < len(facts):
        raise ValueError(f"n_docs={n_docs} is below the {len(facts)} fact documents needed")
    others = [p for p in people if p not in used_people]
    mentions = dict.fromkeys(asked, 0)

    def distractor_project():
        # At most 2 filler mentions per asked project, so "project X" (the model's usual first query)
        # still returns the owner/old-lead docs in its top 5.
        proj = rng.choice(asked)
        if mentions[proj] >= 2:
            return rng.choice(decoys)
        mentions[proj] += 1
        return proj

    for i in range(n_docs - len(facts)):
        p, p2 = rng.sample(others[:60], 2)
        filler = rng.choice([
            f"Agreed that {p} will lead project {rng.choice(decoys)}.",  # decoy project lead
            f"{p} is based in the {city_of[p]} office.",
            f"{p} visited the {rng.choice(CITIES)} office for the quarterly review.",
            f"Ticket OPS-{1000 + i}: {p} reported slow builds on project {distractor_project()}.",
            f"{p} and {p2} demoed project {distractor_project()} to the leadership team.",
            f"Reminder from {p}: expense reports for project {distractor_project()} are due Friday.",
        ])
        facts.append((f"filler-{i}", day(), filler))

    rng.shuffle(facts)
    docs, key_to_id = [], {}
    for n, (key, d, text) in enumerate(facts):
        doc_id = f"doc-{n:04d}"
        key_to_id[key] = doc_id
        date = (datetime.date(2026, 1, 1) + datetime.timedelta(days=d)).isoformat()
        pad = rng.choices(PADDING, k=pad_sentences)  # with replacement: levels exceed the pool size
        pad.insert(rng.randint(0, pad_sentences), text)
        docs.append({"id": doc_id, "kind": rng.choice(KINDS), "date": date, "text": " ".join(pad)})
    for q in questions:
        for step in q["steps"]:
            step["docs"] = [key_to_id[k] for k in step["docs"]]
    _check_reachable(docs, questions)
    return {"docs": docs, "questions": questions}


def _check_reachable(docs, questions):
    """Rigor check: both planted cities appear in the step documents."""
    text = {d["id"]: d["text"] for d in docs}
    for q in questions:
        support = " ".join(text[i] for step in q["steps"] for i in step["docs"])
        assert all(f"based in the {c} office" in support for c in q["answer"].split("; ")), q
