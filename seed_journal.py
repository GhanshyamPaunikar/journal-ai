#!/usr/bin/env python3
"""
seed_journal.py — 40-day fake journal for Innerbloom

Writes directly to data/journal.json with analysis pre-filled (valid emotions,
intensities, tags, themes) so we don't have to wait on llama3.2:3b for 40
entries. Embeddings are left empty — the backend lazy-backfills them on the
first chat request, up to BACKFILL_BUDGET_PER_CALL per request.

Designed to give every panel something interesting:
  - mood line: visible dips and recoveries
  - heatmap: most days filled, some gaps
  - emotions: a mix that's neither all-positive nor all-negative
  - contradictions: stated "I will run" / "I will set boundaries" then doesn't
  - triggers: 'work', 'sleep', 'mom' show up with consistent mood deltas
  - wellbeing: mid-window slump then recovery
  - narrative: clear arcs (career stretch, fitness restart, family ties)
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

# Anchor: "today" in the app context. We space entries backward from here so
# the heatmap/streak/charts all light up around the present.
TODAY = datetime.now().replace(hour=21, minute=30, second=0, microsecond=0)

# Each row: (days_ago, hour, emotion, intensity, tags, themes, title, text)
# 50-day window, ~53 entries (a few days have 2, a few days are skipped so
# streaks and the heatmap look honest rather than ruler-flat).
RAW = [
    # ----- Week 7 ago (49–43 days ago) — "before I was journaling honestly" -----
    (49, 22, "frustrated",   6, ["work","sleep","habits"], ["intention drift"],
     "Said I'd start tomorrow",
     "Told myself last week I'd start journaling, running, sleeping by 11, all of it. Here I am at 11:45 typing this on my phone, eating chips. Tomorrow for real."),
    (48, 21, "anxious",      7, ["work","mom","family"], ["pleasing others"],
     "Mom's call again",
     "Mom called to ask about Diwali plans. The conversation slid into 'when are you settling down' inside three minutes. I went monosyllabic. Hung up feeling small and then mad at myself for feeling small."),
    (47, 9,  "tired",        6, ["sleep","work"], ["pace"],
     "Caffeine shaky",
     "Three coffees by 11am and still couldn't focus. The migration spec is half-written and the deadline is Friday. Going to grind through tonight even though I know that never works."),
    (46, 22, "overwhelmed",  8, ["work","deadlines","sleep"], ["overcommitting"],
     "Friday came",
     "Shipped the migration spec at 8:30pm. It's not good. It's done. Difference matters, I keep telling myself."),
    (44, 20, "lonely",       6, ["friendship","loneliness","weekend"], ["distance"],
     "Saturday alone",
     "Everyone has plans I'm not part of. Told myself I like Saturdays solo. The truth is I'd like to be invited and then say no."),
    (43, 21, "reflective",   5, ["reading","reflection"], ["self-awareness"],
     "Bought a notebook",
     "Walked into the bookstore for nothing specific, walked out with a notebook. Thinking about actually using it. We'll see if past-me's pattern wins."),

    # ----- Week 6 ago (42–37 days ago) — narrative arc starts -----
    (42, 22, "anxious",      7, ["work","sleep","anxiety"], ["rumination"],
     "Sunday dread, again",
     "Same Sunday-night tightness as every Sunday this year. Week ahead has the design review on Wednesday and I haven't started prep. Brain won't stop replaying tomorrow's standup."),
    (41, 9,  "tired",        6, ["sleep","running","habits"], ["intention drift"],
     "No run again",
     "Set the alarm. Hit snooze. Same as last week. The shoes by the door are now a quiet little accusation."),
    (40, 22, "frustrated",   6, ["work","feedback","coding"], ["ego and learning"],
     "Got called out in review",
     "My PR got pushed back. Reviewer was right — I knew the issues were there and shipped anyway. The annoying part is I'd told myself I wouldn't ship lazy code this quarter."),
    (39, 9,  "tired",        7, ["work","deadlines","sleep"], ["overwhelm"],
     "Long Monday",
     "Slack didn't stop pinging until 9pm. I closed the laptop and just stared at the ceiling for a while. Skipped dinner. Ate cereal at 10:30. Tomorrow's standup is going to be brutal."),
    (38, 22, "anxious",       8, ["sleep","anxiety","work"], ["performance anxiety"],
     "Couldn't sleep",
     "Lay awake until 2am running through the standup in my head. Imagined every way it could go wrong. Got out of bed and made tea, scrolled Twitter for half an hour. Promised myself I'd start sleeping by 11."),
    (37, 21, "neutral",       5, ["work","achievement"], ["productivity vs meaning"],
     "Shipped the thing",
     "Standup went fine. The migration draft was decent. Rajiv said 'good start' which from him is glowing praise. So why do I feel hollow? Closed my laptop at 7 and just sat at my desk."),
    (36, 20, "frustrated",    6, ["meditation","quitting"], ["low frustration tolerance"],
     "Three minutes of meditation",
     "Downloaded a meditation app. Tried the 10-minute beginner session. Made it through three minutes before checking the timer. Closed the app. Felt vaguely embarrassed. I keep saying I'll build a meditation habit."),
    (35, 21, "angry",         7, ["friendship","jealousy"], ["misplaced anger"],
     "Snapped at Kabir",
     "Kabir texted asking if I wanted to play badminton tomorrow. I said something passive-aggressive about not having time. He just said 'okay'. He didn't deserve that — I know exactly why I did it. He got promoted last week and I didn't."),
    (34, 22, "anxious",       7, ["work","anxiety","overthinking"], ["rumination","self-criticism"],
     "The meeting that drained me",
     "Two-hour call that could have been an email. Said something I instantly regretted — not mean, just clumsy — and I've been replaying it all evening. I told myself I would stop ruminating in the evenings."),
    (33, 9,  "hopeful",       6, ["health","running","habits"], ["new beginnings"],
     "New month-ish, fresh start",
     "Said today I'd start running every morning. Just 3k to begin. I keep saying this. We'll see if I make it past day three this time."),
    (33, 21, "calm",          6, ["reading","rest"], ["rest without guilt"],
     "A really good book evening",
     "Did nothing useful tonight and it was perfect. Read for two hours, made popcorn, didn't open my laptop. Book is about a man who walks across India — something about it feels grounding."),
    (32, 8,  "tired",         5, ["sleep","running"], ["intention drift"],
     "Skipped the run",
     "Alarm went off at 6:30 and I turned it off and slept until 8. Told myself I'd go in the evening. I won't."),
    (31, 22, "anxious",       8, ["work","deadlines","sleep"], ["Sunday dread"],
     "Sunday anxiety creeping in",
     "Week ahead looks brutal. Three deadlines, a presentation I haven't started, and I promised myself I'd finish the backend feature by Wednesday. Stomach already tight."),
    (30, 21, "proud",         8, ["work","achievement","confidence"], ["owning success"],
     "Smashed the presentation",
     "Nervous going in but it went really well. The CTO asked a follow-up I could answer confidently. Priya said 'that was impressive' after and I didn't know what to do with the compliment."),
    (29, 7,  "happy",         7, ["health","running","habits"], ["momentum"],
     "Good run, clearer head",
     "Went for a run for the first time in two weeks. Lungs hated me for the first kilometer. By the third I was actually enjoying it. There's something about running that quiets the noise."),
    (28, 22, "frustrated",    6, ["work","feedback","coding"], ["ego and learning"],
     "Hard feedback",
     "Got feedback on my PR that stung. Not because it was harsh — it was fair — but because I knew the issues were there and submitted anyway. I'd told myself I wouldn't ship lazy code."),
    (27, 21, "reflective",    5, ["journaling","reflection"], ["self-awareness"],
     "Two weeks in — noticing patterns",
     "Two weeks of journaling now and I'm noticing patterns. Anxiety spikes on certain days. Energy dips when I skip my morning run. Small things but they add up."),
    (26, 9,  "tired",         6, ["sleep","work"], ["pace"],
     "Tired in a heavy way",
     "Slept seven hours but it didn't feel like enough. Coffee didn't help. Stared at the IDE for thirty minutes before writing anything useful."),
    (26, 22, "anxious",       7, ["work","mom","family"], ["pleasing others"],
     "Mom's call",
     "Mom called and the conversation slid into 'when are you settling down' within ten minutes. I went quiet. I love her but those calls leave me wired and small."),
    (25, 21, "lonely",        6, ["friendship","loneliness"], ["distance"],
     "Friday alone again",
     "Everyone seems to have plans. I told myself I like Friday nights to myself. I do — but tonight I'd like the option, you know? Just to feel like someone wanted me there."),
    (24, 11, "content",       6, ["family","home"], ["belonging"],
     "Called Amma, felt grounded",
     "Long call with Amma. She updated me on everyone — who got married, who is expecting, the neighbour's new car. I didn't contribute much but I didn't need to. Felt held."),
    (23, 22, "overwhelmed",   8, ["work","deadlines","sleep"], ["pace","burnout"],
     "The burnout conversation",
     "My tech lead asked at lunch if I was okay. I said fine. He asked again and I told him — I've been running at 110% for three months and I'm not sure I can keep the pace. He listened. Said to take Friday off. I probably won't."),
    (22, 21, "anxious",       7, ["work","sleep"], ["self-criticism"],
     "Didn't take Friday off",
     "Worked Friday after all. Told myself I'd 'wrap up just one thing'. Five hours later. So much for boundaries."),
    (21, 8,  "happy",         8, ["health","running"], ["progress"],
     "6k for the first time",
     "Did a longer run this morning, 6k, which is the furthest I've gone. Took an embarrassing photo of my watch screen because nobody else is going to be proud for me, so."),
    (20, 21, "proud",         9, ["work","achievement","launch"], ["seeing things through"],
     "Launched the feature",
     "Backend feature I've been building for three weeks went live. No bugs in the first few hours, which felt like winning a lottery. Tech lead mentioned it in the team Slack."),
    (19, 22, "tired",         6, ["work","running"], ["earned exhaustion"],
     "Tired but in a good way",
     "Genuinely tired today but not the anxious-depleted kind — the kind where you worked hard and it shows. Going to sleep early for once."),
    (18, 20, "reflective",    7, ["reflection","values","life"], ["life design"],
     "Thinking about what I actually want",
     "Quiet evening, ended up thinking about the bigger picture. Five years from now — what does a good life look like? I don't want to optimize purely for career. Don't fully know what else yet."),
    (17, 21, "reflective",    6, ["self-awareness","patterns"], ["unconscious patterns"],
     "The uncomfortable thing I noticed",
     "I apologize a lot. Not for things that are my fault — for taking up space. 'Sorry to bother you.' 'Sorry, quick question.' Going to try to notice this week and stop a few times."),
    (16, 22, "sad",           5, ["mental-health","sadness"], ["unexplained moods"],
     "Rough day, no reason",
     "Some days are grey. Nothing went wrong. Work was normal. But a low hum of sadness all day that I couldn't pin to anything. I keep wanting to explain it and not being able to."),
    (15, 9,  "anxious",       7, ["work","sleep","deadlines"], ["overcommitting"],
     "Said yes when I should've said no",
     "Manager asked if I could pick up a small piece of work for someone else's project. I said yes. There is no small piece of work. I knew that and said yes anyway."),
    (14, 22, "frustrated",    7, ["sleep","running"], ["intention drift"],
     "Three runs this week — promised seven",
     "I told myself, very loudly, that I'd run every morning this week. Ran Monday, Wednesday, Saturday. Better than nothing. Also: not what I said."),
    (13, 20, "content",       6, ["reading","rest"], ["simple pleasures"],
     "Quiet weekend",
     "No plans. No agenda. Read for hours, made a real breakfast, walked to the market. Got nothing 'done'. Feel better than I have in weeks."),
    (12, 21, "anxious",       6, ["work","mom","family"], ["pleasing others"],
     "Another loaded call",
     "Mom called. The 'when are you coming to visit' question. I am visiting in three weeks. She doesn't actually mean now — she means 'are you still mine'. I think."),
    (11, 22, "tired",         6, ["work","sleep"], ["pace"],
     "Wednesday energy is gone",
     "Hit a wall at 4pm. Couldn't think. Tried to push through and just produced bad code. Closed the laptop at 6. Probably should have closed it at 5."),
    (10, 21, "hopeful",       7, ["learning","tech","growth"], ["self-directed learning"],
     "Started the system design course",
     "Took two hours after work to start a system design course. Not because work asked — because I've been feeling behind on the concepts and I want to understand distributed systems properly. Genuinely enjoyed it."),
    (9,  22, "lonely",        6, ["friendship","loneliness"], ["distance"],
     "Should've gone to the party",
     "Said no to a thing tonight because I was tired. Now I'm scrolling Instagram seeing everyone there. The tired wasn't wrong. The no maybe was."),
    (8,  21, "grateful",      7, ["friendship","gratitude"], ["repair"],
     "Apologized to Kabir",
     "Texted Kabir and said sorry for being a dick a few weeks ago. He sent back a meme. That's how he says it's fine. I'm lucky to have him."),
    (7,  9,  "anxious",       7, ["work","sleep","deadlines"], ["Sunday dread","Sunday dread"],
     "Sunday again",
     "It's the same Sunday-night tightness. Week ahead is heavy. Tried to read but kept checking work Slack. Stop checking work Slack on Sundays — I literally wrote that down two weeks ago."),
    (6,  22, "overwhelmed",   7, ["work","mom"], ["pleasing others","pace"],
     "Said yes to too many things",
     "Two new projects landed today. I agreed to both. Then mom texted asking if I'd send money home this month even though I sent extra last month. Said yes to that too. I don't know how to say no."),
    (5,  22, "frustrated",    6, ["running","sleep"], ["intention drift"],
     "No run today either",
     "Three days in a row of skipping the morning run. The reason changes — too tired, too much work, headache — but the pattern doesn't. I notice it though. That's something."),
    (4,  21, "calm",          7, ["reading","rest","family"], ["belonging"],
     "Quiet evening with a book",
     "Read for an hour. Texted my sister. Made dal. Didn't do anything 'productive'. Felt human again."),
    (3,  9,  "proud",         8, ["work","achievement","feedback"], ["owning success"],
     "Good review",
     "Quarterly review went well. My manager called out the launch and the way I handled the cross-team thing. Said 'you've been carrying a lot — let's plan around that next quarter'. Almost cried, didn't, won't."),
    (2,  21, "hopeful",       7, ["values","reflection","growth"], ["values clarity"],
     "Maybe I'm getting somewhere",
     "Re-reading old entries tonight. The me from 35 days ago was anxious about everything. The me now is still anxious — but specifically. I know what I'm anxious about, which feels like progress."),
    (1,  22, "content",       7, ["reflection","journaling","growth"], ["self-awareness","progress"],
     "Forty days",
     "Forty days of mostly writing every day. I don't have a grand conclusion. I feel slightly more honest with myself than I was. The patterns I notice are uncomfortable but useful. That's enough for tonight."),
]


def _entry_from_row(row):
    days_ago, hour, emotion, intensity, tags, themes, title, text = row
    ts = (TODAY - timedelta(days=days_ago)).replace(hour=hour, minute=30)
    summary = text.split(".")[0][:120] + "."
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "text": text,
        "summary": summary,
        "emotion": emotion,
        "intensity": intensity,
        "tags": tags,
        "themes": themes,
        "user_mood": None,
        "word_count": len(text.split()),
        "timestamp": ts.isoformat(),
        # Embedding left null — backend lazy-backfills on first chat.
    }


def seed_journal():
    entries = [_entry_from_row(r) for r in RAW]
    # Persist oldest-first (consistent with what /save would produce over time).
    entries.sort(key=lambda e: e["timestamp"])
    with open(JOURNAL_FILE, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Seeded {len(entries)} entries")
    print(f"✓ Range: {entries[0]['timestamp'][:10]} → {entries[-1]['timestamp'][:10]}")
    print(f"✓ File:  {JOURNAL_FILE}")


if __name__ == "__main__":
    seed_journal()
