"""Family A: a synthetic org archive with planted facts, contradictions and distractors.

Difficulty dial: hops, the length of the lookup chain. 1 = who leads a project; 2 = that lead's city;
3 = the city of the lead's manager; each further hop climbs one more manager (up to 6). Each hop is reachable only
by searching the previous hop's answer, and every document is ~300 tokens of routine text with one
fact buried in it, so each search adds ~1.5k tokens and long chains outgrow the budget.
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


def generate(n_docs=80, n_questions=12, hops=(1, 2, 3, 4, 5, 6), seed=0, p_correction=0.3, pad_sentences=21):
    """Return {"docs": [{"id", "kind", "date", "text"}],
    "questions": [{"id", "question", "answer", "hops", "chain", "support"}]}; chain[i] is hop i's answer."""
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
        support = []
        if rng.random() < p_correction:
            old = new_person()
            d_old = rng.randint(1, 150)
            facts.append((f"q{qi}-old", d_old, f"Agreed that {old} will lead project {proj}."))
            facts.append((f"q{qi}-owner", rng.randint(d_old + 1, 300),
                          f"Update: {owner} has taken over project {proj} from {old}, effective immediately."))
            facts.append((f"q{qi}-oldcity", day(), f"{old} is based in the {city_of[old]} office."))
        else:
            facts.append((f"q{qi}-owner", day(), f"Agreed that {owner} will lead project {proj}."))
        support.append(f"q{qi}-owner")
        h = hops[qi % len(hops)]
        chain = [owner]
        for m in range(h - 2):  # hops 3+ climb the management chain
            boss = new_person()
            facts.append((f"q{qi}-mgr{m}", day(), f"{chain[-1]} reports to {boss}."))
            support.append(f"q{qi}-mgr{m}")
            chain.append(boss)
        if h >= 2:
            facts.append((f"q{qi}-city", day(), f"{chain[-1]} is based in the {city_of[chain[-1]]} office."))
            support.append(f"q{qi}-city")
            chain.append(city_of[chain[-1]])
        whose = "the " + "manager of the " * (h - 2) + "person"
        q = (f"Who currently leads project {proj}?" if h == 1 else
             f"In which city is the office of {whose} who currently leads project {proj}?")
        questions.append({"id": f"A-{seed}-{qi}", "question": q, "answer": chain[-1], "hops": h,
                          "chain": chain, "support": support})

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
        pad = rng.sample(PADDING, pad_sentences)
        pad.insert(rng.randint(0, pad_sentences), text)
        docs.append({"id": doc_id, "kind": rng.choice(KINDS), "date": date, "text": " ".join(pad)})
    for q in questions:
        q["support"] = [key_to_id[k] for k in q["support"]]
    _check_reachable(docs, questions)
    return {"docs": docs, "questions": questions}


def _check_reachable(docs, questions):
    """Rigor check: every planted answer appears in its supporting documents."""
    by_id = {d["id"]: d for d in docs}
    for q in questions:
        assert any(q["answer"] in by_id[s]["text"] for s in q["support"]), q
