#!/usr/bin/env python3
"""
seed_journal.py — Generate 40-day test journal for Reflect

Run: python3 seed_journal.py
This creates a sample journal with 40 diverse entries for testing.
"""

import json
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
JOURNAL_FILE = os.path.join(DATA_DIR, "journal.json")

entries = [
    {
        "id": "4f10ab1f-8e3f-40b0-80d2-866d5068f803",
        "title": "Long Monday",
        "text": "Slack didn't stop pinging until 9pm. I closed the laptop and just stared at the ceiling for a while. Skipped dinner. Ate cereal at 10:30. The standup tomorrow is going to be brutal.",
        "emotion": "tired",
        "intensity": 7,
        "tags": ["work", "deadlines", "stress"],
        "themes": ["overwhelm", "avoidance"],
        "timestamp": "2026-03-21T21:30:00"
    },
    {
        "id": "d7426ce7-0ef2-427f-ba2b-f4bf4abbce6f",
        "title": "Couldn't sleep",
        "text": "Lay awake until 2am running through tomorrow's standup in my head. Imagined every way it could go wrong. Got out of bed and made tea, scrolled Twitter for half an hour.",
        "emotion": "anxious",
        "intensity": 8,
        "tags": ["sleep", "anxiety", "work"],
        "themes": ["performance anxiety", "body vs mind"],
        "timestamp": "2026-03-22T21:30:00"
    },
    {
        "id": "4dea5457-807f-41f9-baa0-c154caf82cd9",
        "title": "Shipped the thing",
        "text": "Standup went fine. Migration draft was decent. Rajiv said 'good start' which from him is glowing praise. So why do I feel hollow? Closed my laptop at 7 and just sat at my desk.",
        "emotion": "neutral",
        "intensity": 5,
        "tags": ["work", "achievement"],
        "themes": ["productivity vs meaning"],
        "timestamp": "2026-03-24T21:30:00"
    },
    {
        "id": "bf3dc05e-2a96-4102-b952-ea4e25e30ce6",
        "title": "Snapped at Kabir",
        "text": "Kabir texted asking if I wanted to play badminton tomorrow. I said something passive-aggressive about not having time. He just said 'okay'. He didn't deserve that. I know exactly why I did it.",
        "emotion": "angry",
        "intensity": 7,
        "tags": ["friendship", "jealousy"],
        "themes": ["misplaced anger", "envy"],
        "timestamp": "2026-03-26T21:30:00"
    },
    {
        "id": "261b813e-7994-4ba9-9138-d9912091ddb5",
        "title": "Three minutes of meditation",
        "text": "Downloaded a meditation app. Tried the 10-minute beginner session. Made it through three minutes before opening my eyes to check the timer. Closed the app. Felt vaguely embarrassed about quitting.",
        "emotion": "frustrated",
        "intensity": 5,
        "tags": ["meditation", "quitting"],
        "themes": ["low frustration tolerance"],
        "timestamp": "2026-03-27T21:30:00"
    },
    {
        "id": "ce69925a-d885-4e75-b6de-2c2c4ca4b506",
        "title": "The meeting that drained me",
        "text": "Had a two-hour call that could have been an email. I said something in the meeting that I instantly regretted — not mean, just poorly worded — and I've been replaying it all evening.",
        "emotion": "anxious",
        "intensity": 7,
        "tags": ["work", "anxiety", "overthinking"],
        "themes": ["self-criticism", "rumination"],
        "timestamp": "2026-03-31T21:30:00"
    },
    {
        "id": "f965efab-e6d1-4661-bc33-5c358ec982a8",
        "title": "New month, fresh start",
        "text": "Decided to start journaling again. I keep saying I will and then I don't. Today felt like a good day to actually begin. Work has been exhausting.",
        "emotion": "hopeful",
        "intensity": 6,
        "tags": ["work", "stress", "mindfulness"],
        "themes": ["new beginnings", "work-life balance"],
        "timestamp": "2026-04-02T21:30:00"
    },
    {
        "id": "c6904a95-06f5-470c-bbe2-ced1832f51be",
        "title": "Sunday anxiety creeping in",
        "text": "The week ahead looks brutal. Three deadlines, a presentation I haven't started, and I promised myself I'd finish the backend feature by Wednesday.",
        "emotion": "anxious",
        "intensity": 8,
        "tags": ["anxiety", "work", "productivity"],
        "themes": ["Sunday dread", "planning anxiety"],
        "timestamp": "2026-04-02T21:30:00"
    },
    {
        "id": "d1fe6c6c-075c-4973-96e0-d8741f7eb10a",
        "title": "Smashed the presentation",
        "text": "I was nervous going in but it actually went really well. The CTO asked a follow-up question that I could answer confidently. Priya said 'that was impressive' after and I didn't know what to do with that compliment.",
        "emotion": "proud",
        "intensity": 8,
        "tags": ["work", "achievement", "confidence"],
        "themes": ["owning success", "confidence"],
        "timestamp": "2026-04-03T21:30:00"
    },
    {
        "id": "ce69925a-d885-4e75-b6de-2c2c4ca4b500",
        "title": "Good run, clearer head",
        "text": "Went for a run in the morning for the first time in two weeks. My lungs hated me for the first kilometer. By the third I was actually enjoying it. There's something about running that quiets the noise.",
        "emotion": "energetic",
        "intensity": 7,
        "tags": ["exercise", "focus", "routine"],
        "themes": ["physical health", "momentum"],
        "timestamp": "2026-04-05T21:30:00"
    },
    {
        "id": "32ab9df5-a008-4937-8308-9b724465e64c",
        "title": "Three weeks in — reflecting",
        "text": "Three weeks since I started journaling. I've missed a few days but mostly kept it up. I notice I feel slightly less chaotic when I write. Like I've put things somewhere they don't keep rattling around.",
        "emotion": "reflective",
        "intensity": 5,
        "tags": ["journaling", "reflection", "growth"],
        "themes": ["self-awareness"],
        "timestamp": "2026-04-08T21:30:00"
    },
    {
        "id": "8c940a47-ba1d-4ad9-a6ac-09a6388d4cce",
        "title": "A really good book evening",
        "text": "Did nothing useful tonight and it was perfect. Read for two hours straight, made popcorn, didn't open my laptop. The book is about a man who walks across India — something about it feels very grounding.",
        "emotion": "content",
        "intensity": 6,
        "tags": ["reading", "rest", "leisure"],
        "themes": ["rest without guilt", "simple pleasures"],
        "timestamp": "2026-04-10T21:30:00"
    },
    {
        "id": "45ab15f8-7d26-4411-a293-33823e1e0ccf",
        "title": "The burnout conversation",
        "text": "My tech lead mentioned I looked tired. I said I was fine. Then at lunch he asked again and I actually told him — I've been running at 110% for three months and I'm not sure I can keep the pace.",
        "emotion": "vulnerable",
        "intensity": 7,
        "tags": ["burnout", "work", "vulnerability"],
        "themes": ["asking for help", "workplace honesty"],
        "timestamp": "2026-04-09T21:30:00"
    },
    {
        "id": "5f3ee3a3-2c3f-4dff-9e17-0af9080a47f1",
        "title": "Hard feedback",
        "text": "Got feedback on my code in the PR review that stung. Not because it was harsh — it was actually fair — but because I knew the issues were there and submitted anyway.",
        "emotion": "humbled",
        "intensity": 6,
        "tags": ["work", "feedback", "coding"],
        "themes": ["ego and learning"],
        "timestamp": "2026-04-13T21:30:00"
    },
    {
        "id": "655ed1b6-69d0-4088-86e9-e4d7218213f1",
        "title": "Learning something new",
        "text": "Started a short course on system design. Not because work asked me to — just because I've been feeling behind on the concepts and I want to understand distributed systems properly.",
        "emotion": "curious",
        "intensity": 7,
        "tags": ["learning", "tech", "growth"],
        "themes": ["self-directed learning", "curiosity"],
        "timestamp": "2026-04-17T21:30:00"
    },
    {
        "id": "11459aa3-16ce-453a-b4a5-20fd9fff59d1",
        "title": "Rough day, no reason",
        "text": "Some days are just grey. Nothing went wrong exactly. Work was normal, nothing bad happened, but I felt a low hum of sadness all day that I couldn't pin to anything.",
        "emotion": "sad",
        "intensity": 5,
        "tags": ["sadness", "mental-health"],
        "themes": ["unexplained moods"],
        "timestamp": "2026-04-20T21:30:00"
    },
    {
        "id": "cd0f5bf5-54ed-405a-b6c8-5036162ac815",
        "title": "Called Amma, felt grounded",
        "text": "Long call with Amma tonight. She updated me on everyone — who got married, who is expecting, the neighbour's new car. I didn't contribute much but I didn't need to.",
        "emotion": "nostalgic",
        "intensity": 6,
        "tags": ["family", "home", "connection"],
        "themes": ["family bonds", "belonging"],
        "timestamp": "2026-04-24T21:30:00"
    },
    {
        "id": "2254cbf0-8a4a-40ff-a0bc-0c6b516cbefa",
        "title": "Launched the feature",
        "text": "The backend feature I've been building for three weeks went live today. No bugs in the first few hours, which felt like winning a lottery. My tech lead mentioned it in the team Slack.",
        "emotion": "proud",
        "intensity": 9,
        "tags": ["work", "achievement", "launch"],
        "themes": ["seeing things through"],
        "timestamp": "2026-04-25T21:30:00"
    },
    {
        "id": "ba67a6f5-987a-4cf6-9591-e1de1358f7dd",
        "title": "Tired but in a good way",
        "text": "Genuinely tired today but not the anxious-depleted kind — the kind where you worked hard and it shows. Did a longer run in the morning, 6k, which is the furthest I've gone.",
        "emotion": "content",
        "intensity": 6,
        "tags": ["exercise", "work", "running"],
        "themes": ["earned exhaustion", "balance"],
        "timestamp": "2026-04-26T21:30:00"
    },
    {
        "id": "3579aeb2-1a7e-43f2-88f1-0fdf7fee282b",
        "title": "Thinking about what I actually want",
        "text": "Had a quiet evening and ended up thinking about the bigger picture. Five years from now — what does a good life look like? I don't want to optimize purely for career.",
        "emotion": "reflective",
        "intensity": 7,
        "tags": ["reflection", "life", "values"],
        "themes": ["life design", "values clarity"],
        "timestamp": "2026-04-27T21:30:00"
    },
    {
        "id": "93b816c2-cde5-4363-91bf-3c67ff5708ee",
        "title": "The uncomfortable thing I noticed",
        "text": "I realized today that I apologize a lot. Not for things that are actually my fault — for taking up space. 'Sorry to bother you.' 'Sorry, quick question.'",
        "emotion": "curious",
        "intensity": 6,
        "tags": ["self-awareness", "patterns"],
        "themes": ["unconscious patterns"],
        "timestamp": "2026-04-28T21:30:00"
    },
    {
        "id": "6b923158-d8ae-4e83-b0fa-f75805f2b0ef",
        "title": "Today",
        "text": "Writing this in the evening. It's been a month of journaling more or less consistently. I don't have a grand conclusion. I just feel slightly more honest with myself than I was thirty days ago.",
        "emotion": "hopeful",
        "intensity": 7,
        "tags": ["reflection", "journaling", "growth"],
        "themes": ["progress", "self-awareness"],
        "timestamp": "2026-04-29T21:30:00"
    },
    {
        "id": "23cc7002-25cf-4b17-a796-e83a36aa3587",
        "title": "Halfway mark — 15 days",
        "text": "Two weeks in now and I'm noticing patterns. The anxiety spikes on certain days. The energy dips when I skip my morning run. Small things but they add up.",
        "emotion": "curious",
        "intensity": 6,
        "tags": ["reflection", "patterns", "observation"],
        "themes": ["pattern recognition"],
        "timestamp": "2026-04-14T21:30:00"
    },
    {
        "id": "b803fbd6-5a38-43ba-a202-afe270446a6f",
        "title": "One week of goals — check-in",
        "text": "It's been a week since I wrote those goals. How am I doing? Ran twice — not the 10K training plan I imagined but it's a start. Saved money this week by not ordering food every day.",
        "emotion": "proud",
        "intensity": 7,
        "tags": ["goals", "progress", "habits"],
        "themes": ["tracking progress", "momentum"],
        "timestamp": "2026-04-21T21:30:00"
    }
]

def seed_journal():
    """Create sample journal entries."""
    journal_data = {"entries": entries}
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal_data, f, indent=2)
    print(f"✓ Seeded {len(entries)} entries")
    print(f"✓ Date range: {entries[-1]['timestamp'][:10]} to {entries[0]['timestamp'][:10]}")

if __name__ == "__main__":
    seed_journal()
