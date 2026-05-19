#!/usr/bin/env python3
"""
seed_journal.py — 65-day fake journal for Innerbloom

Character (consistent across the run):
- ~28, AI engineer on a deliberate career break to build skills & ship
  a personal AI app (this app, more or less).
- Returned from ~5 years in London to home in India. Misses UK life.
- Sober ~9 months and proud of the clarity it bought.
- Good friend circle (Kabir, Anjali, Sahir) + one bad friend (Vikram)
  who uses him; learning to set limits there.
- Highly empathetic / compassionate → also gets drained by it.
- Tired of building new friendships from scratch; trust takes time.
- Playful side: learning inline-skate tricks, loves house & pop music,
  cooks for himself.
- Choosing isolation/grind right now, but feels the cost.
- Disciplined: clean diet, mock interviews, daily commits, deliberate
  practice on communication.
- Applying for jobs, getting some rejections, some momentum.

Each row is (days_ago, hour, emotion, intensity, [tags], [themes],
"title", "text"). Some days have two entries; some days are skipped.
"""

import json
import os
import uuid
from datetime import datetime, timedelta

DATA_DIR = os.environ.get(
    "INNERBLOOM_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
os.makedirs(DATA_DIR, exist_ok=True)
JOURNAL_FILE = os.path.join(DATA_DIR, "journal.json")

TODAY = datetime.now().replace(hour=21, minute=30, second=0, microsecond=0)

RAW = [
    # =============================================================
    # ~9 weeks back — the decision to take a career break
    # =============================================================
    (64, 22, "anxious", 7, ["work","career","decision","uk"], ["career break","uncertainty"],
     "Sent the resignation",
     "Hands actually shook hitting send. The manager called within an hour, kind but trying to talk me out of it. I held my line. Said I needed time to build the thing I've been talking about for two years. Now I'm at the desk and the future is a blank page and I can't tell if that's freedom or terror."),
    (63, 21, "reflective", 6, ["uk","london","leaving"], ["uk nostalgia","ending"],
     "Last week in the flat",
     "Started packing the kitchen. Found the kettle from the first month here, still works. Five years in this country and I'm leaving with two suitcases. The light through the window in the morning — that specific soft grey — is what I'll miss most."),
    (62, 22, "sad", 6, ["uk","goodbye","friends"], ["uk nostalgia"],
     "Last pint at the local",
     "Not drinking but everyone else was. Sat with my lime soda and watched the people who became my life here for five years. Hard to explain that I love them and I'm also relieved to be leaving. Both things, at the same time, all night."),

    # =============================================================
    # 8 weeks ago — arrival, regrouping
    # =============================================================
    (61, 9, "tired", 6, ["sleep","jetlag","home","family"], ["transition"],
     "Three hours of sleep",
     "Landed at 1am. Couldn't fall asleep till 4. Body still on London time. Mom kept opening the door to check if I was okay. I love her but I'm 28."),
    (60, 23, "hopeful", 6, ["sober","milestone","detox"], ["sobriety"],
     "Eight months sober",
     "Eight months today. Didn't post it anywhere. Didn't tell anyone. Just sitting with the number quietly. The Friday-night version of me wouldn't recognize this one and that's the whole point."),
    (58, 21, "anxious", 6, ["jobs","applications","ai","career"], ["job hunt"],
     "First five applications out",
     "Sent five today. ML engineer roles, mostly remote. Polished the CV three times before I could press submit on any of them. The waiting starts now. The waiting is the hardest part."),
    (57, 20, "grateful", 7, ["friends","kabir","circle"], ["belonging"],
     "Kabir came over",
     "Kabir drove an hour to come hang. We didn't even do much, watched bad TV, ate dosa. He's been my friend since we were eleven. There's a kind of friendship where you don't have to perform and tonight I felt it again."),
    (56, 22, "lonely", 6, ["uk","missing","friends"], ["uk nostalgia","distance"],
     "Group chat scrolled past me",
     "The London group chat has been quiet for me but loud for them. Someone's birthday, someone's promotion. I'm thirteen replies behind and I don't have the energy to catch up. Feeling forgotten and knowing it's partly my own doing doesn't help."),

    # =============================================================
    # 7 weeks ago — starting the AI app
    # =============================================================
    (55, 10, "excited", 8, ["ai","coding","building","focus"], ["building","creative momentum"],
     "First commit",
     "Pushed the first commit on the journal app. It's nothing yet — a FastAPI skeleton and a hello-world endpoint. But it's mine. I have wanted to build something like this for years. Today I actually started."),
    (54, 22, "tired", 6, ["work","coding","late","grinding"], ["grinding"],
     "1am again",
     "Couldn't stop. Got the entry parser working and the Ollama hookup roughed in. Lost track of time. Eyes hurt. Going to wake up wrecked tomorrow but right now I don't regret it."),
    (53, 19, "hopeful", 7, ["health","fitness","diet"], ["self-improvement"],
     "Day one of clean eating",
     "Cooked all three meals. No processed anything. Sober, no junk, building something — I'm becoming a person I didn't think I had the discipline to be. Strange feeling. Suspicious of it, even."),
    (52, 19, "frustrated", 7, ["friends","vikram","trust","bad-friend"], ["used by friends"],
     "Vikram needed money",
     "Vikram texted again, asking to borrow money. Third time this year. He always pays back late, sometimes not at all. I said yes anyway. I'm tired of being the friend who can be asked. I'm also tired of being the friend who finally says no."),
    (51, 21, "calm", 6, ["walk","health","evening"], ["simple pleasures"],
     "Walked at sunset",
     "An hour, no music, no podcast. Watched two dogs play. Sat on a bench for ten minutes doing nothing. Came home and felt the kind of quiet you can't manufacture."),
    (50, 22, "reflective", 5, ["sober","detox","health","milestone"], ["sober identity"],
     "Eight and a half",
     "Realized I've stopped counting days. Used to count, now it's just life. The detox itself was hell for the first month. The clarity now — I never want to forget how this feels at 28 because the version of me at 24 would have laughed."),

    # =============================================================
    # 6 weeks ago — first rejection, skates arrive
    # =============================================================
    (49, 22, "anxious", 7, ["jobs","rejection","ai"], ["rejection"],
     "First rejection",
     "Auto-reply at 9pm. 'After careful consideration.' No interview. I'd been quietly optimistic about that one. The crash is sharper than I thought it would be. Going to bed without writing more about it because I know where rumination ends."),
    (48, 19, "calm", 6, ["skate","new","learning","play"], ["beginner","play"],
     "Got the skates",
     "Inline skates arrived. First time strapping them on since I was twelve. Ten minutes in the apartment, terrified to go outside. Eventually went out, did one cautious loop of the park. Knees survived. Feel like a kid again, in a good way."),
    (47, 21, "content", 6, ["music","house","evening","sober"], ["enjoyment","sober pleasures"],
     "House mix and a clean kitchen",
     "Put on a deep-house mix and cooked dal. Three hours just disappeared. The whole point of going sober wasn't to suffer. It was to enjoy nights like this without needing to be wrecked."),
    (46, 22, "frustrated", 7, ["communication","jobs","self"], ["working on self"],
     "I ramble when I'm nervous",
     "Did a mock interview with Sahir on a call. He said gently I should answer shorter, not start every reply with 'so basically.' Heard. Wrote it down. The thing I'm most insecure about is exactly what I have to drill."),
    (45, 23, "tired", 6, ["work","coding","late"], ["pace"],
     "Up too late again",
     "Closing the laptop at 1:30. Telling myself this is the season. Knowing the season has a body cost. Tomorrow I'll start before sunset instead of finishing after midnight."),
    (44, 22, "angry", 7, ["friends","trust","vikram","bad-friend"], ["trust fatigue","used by friends"],
     "Vikram cancelled, again",
     "He bailed on coming to help me move boxes. Texted at 6pm when I was supposed to pick him up at 6:30. 'Stuck at work.' Sahir mentioned later that Vikram was at a concert tonight. I keep extending grace I don't have anymore."),
    (43, 20, "lonely", 7, ["friends","new","trust","tired"], ["trust fatigue"],
     "Someone from the AI meetup wants coffee",
     "Got a message from a person I met at the meetup. Wants to grab coffee. I should be excited and instead I'm exhausted. Building a new friendship from scratch costs more energy than I have right now. Trust takes years and I don't have any to spare. Maybe I'll say yes anyway."),

    # =============================================================
    # 5 weeks ago — skate progress, overwhelm
    # =============================================================
    (42, 11, "proud", 7, ["skate","learning","play","progress"], ["beginner progress","play"],
     "Cross-overs",
     "Skated for an hour. Tried cross-overs going left. Fell once, hard, on the elbow pad. Got back up. Eventually got two in a row clean. Eleven-year-old me would be hysterical. 28-year-old me is hysterical."),
    (41, 22, "overwhelmed", 8, ["work","coding","jobs","ai"], ["overcommitting","pace"],
     "Too many tabs",
     "Three job applications, the contradictions engine debug, dinner I never made. Closed the laptop at midnight feeling like nothing got finished. Have to start writing things down again. The grind without a system becomes flailing."),
    (40, 20, "lonely", 7, ["friends","missing","fun","fomo"], ["isolation chosen"],
     "Skipping Sahir's brother's wedding",
     "Said no. Three days I can't afford right now. I know what I'm choosing. I know it costs. The cost is real and it's still the right call. Writing it down so I remember I chose it on purpose, not by drift."),
    (39, 22, "tired", 6, ["work","late","grinding"], ["pace"],
     "Seasons, not balance",
     "Read something about how 'balance' is a myth for builders. You have seasons. This is a season. Tells me I'm allowed to grind right now. The next season I get to soften."),
    (38, 21, "happy", 7, ["skate","music","play","park"], ["play"],
     "First ten-minute skate flow",
     "Headphones in, Folamour set, an empty stretch of pavement near the lake. For ten unbroken minutes I forgot every applications spreadsheet and every interview prompt. Just moved. Came home grinning at the wall like an idiot."),
    (37, 21, "hopeful", 7, ["jobs","interview","ai"], ["job hunt"],
     "Got a screening call",
     "Recruiter from a startup in Bangalore. 25-minute screening tomorrow. Practiced my elevator pitch in the bathroom mirror. Felt absurd and necessary."),

    # =============================================================
    # 4 weeks ago — interview rambling, communication work
    # =============================================================
    (36, 19, "anxious", 7, ["jobs","interview","communication"], ["performance"],
     "Did the screening",
     "It went… okay. I rambled on the 'tell me about yourself' question. Could feel myself losing the thread. Recruiter was kind. Wants to move me to the technical round. The communication piece is what I have to fix and it's also the hardest thing for me."),
    (35, 22, "frustrated", 6, ["communication","self","practice"], ["working on self"],
     "Mirror practice",
     "Practiced answering 'tell me about yourself' to the bathroom mirror tonight. Felt stupid. Did it eight more times. The eighth one was less rambly. Eight is what improvement looks like, not one perfect take."),
    (34, 22, "content", 6, ["music","pop","evening","sober"], ["enjoyment"],
     "Charli set on repeat",
     "Played that Charli XCX album three times through tonight while debugging. The right kind of loud. Eight months ago this would have been a club night. Tonight it's clean code and a bowl of noodles. Wouldn't trade."),
    (32, 20, "happy", 8, ["skate","play","music","house"], ["play"],
     "Skating to deep house",
     "Headphones in, deep tropical-house set, lake path. For thirty minutes I forgot every applications spreadsheet and every interview question. This is what nights are for now."),
    (31, 23, "lonely", 7, ["friends","missing","uk","london"], ["isolation","uk nostalgia"],
     "Missing the London Sundays",
     "It was the long pub lunches I miss. The way Sunday could stretch on, six of us, hours, nothing to do. India Sundays haven't found their shape yet. The friends I have here are scattered. Working on it. Slow."),
    (30, 22, "reflective", 6, ["journal","self","values","ai"], ["values clarity"],
     "What am I actually building",
     "Realized tonight the journal app isn't really about the app. It's the practice of building something every day. The discipline. The deliverable. Whether it ships or not, the version of me who can do this for 60 days is who I'm trying to become."),

    # =============================================================
    # 3 weeks ago — heavy isolation, Vikram blow-up coming
    # =============================================================
    (29, 21, "angry", 7, ["friends","vikram","bad-friend","used","boundaries"], ["used by friends"],
     "Vikram again, at 11pm",
     "Vikram showed up at 11pm wanting to crash, no notice. I let him in because what was I going to do at 11pm. He stayed up watching reels, ate everything in the fridge, left at 2am without doing the dishes. I'm not angry at him. I'm angry at me for not having a script yet."),
    (28, 9, "tired", 7, ["sleep","grinding","work","coding"], ["pace"],
     "Slept five hours",
     "Coding till 3am chasing an embeddings bug. Found it at 2:47am — off-by-one in the cosine sim. Felt the rush, then immediately the cost. This is the part of grinding that's actually expensive and I keep paying it."),
    (27, 21, "grateful", 7, ["friends","anjali","circle"], ["belonging"],
     "Anjali called",
     "Anjali called from Berlin. Hadn't spoken in two months. Forty minutes about nothing in particular. Felt seen in a way I didn't know I was missing. The good friends just are. They don't need maintenance the same way."),
    (26, 22, "proud", 8, ["ai","building","ship","coding"], ["seeing things through"],
     "Got the chat agent streaming",
     "Got the agent endpoint streaming tool calls in real-time. Watching the steps appear one by one in the browser is genuinely satisfying. Six weeks ago I didn't know how to build this. I do now."),
    (25, 20, "calm", 6, ["walk","health","evening"], ["simple pleasures"],
     "Walked for an hour",
     "No music, no podcast, no agenda. Watched two dogs play on the lawn at the school. Sat on a bench for ten minutes doing nothing. Came home and felt the kind of quiet you can't manufacture."),
    (24, 22, "frustrated", 6, ["jobs","rejection","communication"], ["rejection"],
     "Second rejection email",
     "Made it to the take-home round. They picked someone else. The feedback was 'strong candidate, more experienced match.' I think that's code for someone four years older. Tired."),
    (23, 21, "lonely", 7, ["friends","isolation","missing"], ["isolation"],
     "Three days no real conversation",
     "Realized I haven't had a real conversation in three days. Not counting recruiter calls. Three days of just me and the LLM. The work is good. Something else is starving and I have to admit it."),

    # =============================================================
    # 2 weeks ago — house music nights, sober pleasures
    # =============================================================
    (22, 21, "content", 6, ["music","house","pop","evening","sober"], ["enjoyment"],
     "Caribou and a candle",
     "Spent two hours just listening to Caribou. Sober, dinner I cooked, candle lit because I had one and it felt nice. The version of me that needed five drinks to get to this kind of softness — I don't know him anymore."),
    (21, 22, "overwhelmed", 8, ["ai","work","jobs","grinding"], ["overcommitting","pace"],
     "Drowning a little",
     "Two interview prep sessions tomorrow, the contradictions engine still flaky, mom's hinting about Diwali plans. Closed all the tabs and just stared at the wall for ten minutes. Counted backwards from 100. It worked, sort of."),
    (19, 23, "reflective", 6, ["sober","detox","milestone"], ["sober identity"],
     "Three weeks from a year",
     "Three weeks from a year sober. Started thinking about whether to mark it somehow. Or just let it be a Tuesday. The Tuesday-version of marking it is the one I lean toward."),
    (18, 21, "anxious", 6, ["jobs","interview","prep"], ["performance"],
     "Tech screen tomorrow",
     "Did three mock STAR answers with Anjali on a call. She caught me opening every answer with 'so basically' again. We laughed. Then I drilled it. Going to bed early tonight for once."),
    (17, 19, "happy", 8, ["skate","play","park","progress"], ["play","progress"],
     "Did a transition!",
     "Skated for an hour and a half. Tried switching from forward to backward — fell twice. Got it on the third try. Cleanly. Squealed out loud like an actual child. There was a couple on a bench and they laughed with me, not at me."),
    (16, 22, "anxious", 7, ["communication","jobs","interview"], ["performance"],
     "Technical interview tomorrow",
     "Reviewed transformers, distributed systems, the project I built. Did mock STAR answers. The technical part isn't what scares me. It's the 'tell me about a time' questions — the ones where I have to talk about myself and not sound like I'm reading off a cue card."),
    (15, 23, "reflective", 7, ["communication","self","jobs","interview"], ["working on self"],
     "Did the interview",
     "Three hours. I didn't ramble this time. Asked clarifying questions before answering. Said 'I don't know, here's how I'd think about it' twice and meant both. That's a different person than two months ago."),

    # =============================================================
    # Last 2 weeks — Vikram resolution, friend circle warmth
    # =============================================================
    (14, 22, "angry", 7, ["friends","vikram","bad-friend","trust","boundaries"], ["used by friends","trust fatigue"],
     "Found out Vikram lied",
     "Vikram told me he 'couldn't' come help me move last month. Today Sahir mentioned Vikram was at a concert that night with two other guys. So he just didn't want to. I keep extending grace I don't have. I think I'm done. The 'done' doesn't have a satisfying movie scene. It's a small quiet thing."),
    (13, 20, "calm", 7, ["walk","music","house","evening"], ["enjoyment"],
     "Evening walk, Mall Grab set",
     "Watched the sun go down with a Mall Grab set in my ears. Forty minutes. Came home and made aloo paratha. The little life. People underestimate the little life."),
    (12, 21, "grateful", 7, ["friends","kabir","circle"], ["belonging"],
     "Kabir, no agenda",
     "Kabir dropped by. Brought a book he thought I'd like. Stayed for tea. Didn't try to fix anything, didn't ask about jobs. The friends who don't ask about progress are the most healing kind."),
    (11, 22, "frustrated", 6, ["ai","work","bug","coding"], ["work pattern"],
     "Memory leak hunt",
     "Spent four hours hunting a memory leak in the embedding cache. Was a stupid one — a circular reference in the LRU eviction. The bug is fixed. The four hours aren't."),
    (10, 20, "hopeful", 7, ["jobs","interview","feedback"], ["job hunt"],
     "Decision this week",
     "Got an email saying decision this week from the Bangalore startup. Don't want to bank on it. Banking on it is how disappointment compounds. But I'd be lying if I didn't say I'm hopeful."),
    (9, 22, "proud", 8, ["ai","building","ship","refactor"], ["seeing things through"],
     "Refactor done",
     "Clean rewrite of the memory module — IDF weighting, MMR diversification, capped backfill. Three weeks ago this would have taken me five days. Today it took six hours and the tests still pass. This is what improvement actually looks like — the same thing taking less time."),
    (8, 21, "content", 6, ["sober","detox","reflection"], ["sober identity"],
     "What sobriety bought me",
     "Realized tonight what sobriety actually paid for. Not virtue. Time. I have so many more hours per week. So much more energy. So much less recovering. I just didn't see what I was spending."),

    # =============================================================
    # Last week — offers, decisions, joy
    # =============================================================
    (7, 19, "happy", 8, ["skate","play","tricks","park","progress"], ["play","progress"],
     "Backwards skating, a full lap",
     "Could skate backwards for an entire lap without falling today. A month ago I couldn't go more than three meters. Tried a one-foot glide on the way home, in front of an audience of one bored dog. Stayed up. Felt invincible for thirty seconds."),
    (6, 22, "lonely", 6, ["friends","missing","fomo","fun"], ["isolation chosen"],
     "Sahir's birthday I missed",
     "Pictures from Sahir's birthday went up on the chat. Everyone there. Beach. Looked beautiful. I chose not to go because I had a technical interview the morning after. Both things are true: it was the right call, and I'm sad I wasn't there. Holding both."),
    (5, 21, "anxious", 7, ["jobs","decision","ai","offer"], ["uncertainty","values clarity"],
     "Two offers in the funnel",
     "Bangalore startup making an offer Monday. Berlin remote startup wants a third-round. Both feel real. Suddenly the abstract 'job hunt' has corners and shapes and I have to decide which one I am. Anxious in a different way — not because nothing's happening, because something is."),
    (4, 23, "reflective", 6, ["self","communication","year"], ["working on self"],
     "What changed in 9 weeks",
     "Listed what's actually shifted since I came back: ramble less, hold my breath less, eat real food, lift my eyes from the screen. The journal entries from week one are unreadable to me now — anxious, scattered, lying to myself in subtle ways. The current me sees that. That's the gift."),
    (3, 20, "grateful", 8, ["friends","circle","kabir","sahir","trust"], ["belonging"],
     "Dinner with the actual circle",
     "Five of us at Kabir's. No agenda. The kind of dinner where you can put your feet on the chair and someone tells a stupid joke at exactly the right time. Vikram wasn't there and nobody noticed. The good circle is real. The one rotten apple doesn't define the orchard."),
    (2, 21, "happy", 8, ["skate","music","house","play","sober"], ["play","enjoyment"],
     "Best skate evening yet",
     "Skated for an hour and a half to a Jamie xx set. Light was perfect. Did a clean stop-and-pivot in front of a kid who said 'whoa.' Came home and just sat with the feeling. This is what sober-late-twenties was supposed to be. I didn't know it could be like this."),
    (1, 22, "hopeful", 8, ["jobs","career","decision","offer","ai"], ["values clarity"],
     "Offer letter arrived",
     "The Bangalore startup sent the formal offer. Read it three times. Salary fine. Stack good. Team I clicked with. I'll sleep on it but my body already knows the answer. Nine weeks ago I couldn't have imagined this paragraph. Funny how quickly impossible becomes obvious."),
    (0, 21, "content", 7, ["reflection","self","milestone","journal"], ["progress"],
     "65 days of writing",
     "Sixty-five entries in this journal. Going to take an offer tomorrow. Sober almost a year. Skate decently now. Built a real piece of software. Cooked real food most nights. Lonely sometimes. Missed UK most days. But I'm here, present in my own life, and not buying my way out of it with a drink. That's the thing."),
]


import re as _re_seed

# Names the character mentions across the seeded entries. Extracting these
# by capitalised-word heuristic so we don't need an LLM at seed time.
_KNOWN_NAMES = {"Kabir", "Vikram", "Sahir", "Anjali", "Mom", "Amma",
                "Rajiv", "Priya"}


def _extract_people_from_text(text: str) -> list:
    """Cheap name-detection for seed data: capitalised words that match
    our known-names list. Order-preserved, deduped."""
    found = []
    seen = set()
    for word in _re_seed.findall(r"[A-Z][a-zA-Z]+", text or ""):
        if word in _KNOWN_NAMES and word not in seen:
            seen.add(word)
            found.append(word)
    return found


def _entry_from_row(row):
    days_ago, hour, emotion, intensity, tags, themes, title, text = row
    ts = (TODAY - timedelta(days=days_ago)).replace(hour=hour, minute=30)
    summary = text.split(".")[0][:140] + "."
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "text": text,
        "summary": summary,
        "emotion": emotion,
        "intensity": intensity,
        "tags": tags,
        "themes": themes,
        "people": _extract_people_from_text(text),
        "user_mood": None,
        "word_count": len(text.split()),
        "timestamp": ts.isoformat(),
    }


def seed_journal():
    entries = [_entry_from_row(r) for r in RAW]
    entries.sort(key=lambda e: e["timestamp"])
    with open(JOURNAL_FILE, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Seeded {len(entries)} entries over {RAW[0][0] + 1} days")
    print(f"Range: {entries[0]['timestamp'][:10]} -> {entries[-1]['timestamp'][:10]}")
    print(f"File:  {JOURNAL_FILE}")


if __name__ == "__main__":
    seed_journal()
