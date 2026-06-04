"""
Backtest for Innerbloom backend.

Runs end-to-end against FastAPI's TestClient with Ollama + Spotify mocked.
Every route is exercised; stats/retrieval are validated with known data.
"""

import os, sys, json, tempfile, shutil, time, threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Fresh data dir per run
TMP = tempfile.mkdtemp(prefix="innerbloom_test_")
os.environ["INNERBLOOM_DATA_DIR"] = TMP
os.environ["INNERBLOOM_MODEL"] = "test-model"

# Import after env is set
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as innerbloom_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(innerbloom_app.app)

# ---------- tiny helpers ----------

PASS = 0
FAIL = 0
FAILURES = []

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}  — {detail}")
        print(f"  ✘ {name}  — {detail}")

def section(title):
    print(f"\n== {title} ==")

# ---------- LLM mocks ----------

def fake_llama(prompt, system="", temperature=0.7, timeout=180):
    if "Return ONLY a compact JSON object" in prompt:
        return '{"emotion":"happy","intensity":7,"summary":"A good day at work.","tags":["work","life","focus"],"themes":["productivity","growth"]}'
    if "exactly 3 follow-up questions" in prompt:
        return "What felt most alive today?\nWhat would you change if you could?\nWhat are you avoiding naming?"
    if "ONE short journaling prompt" in prompt:
        return "What surprised you today?"
    if "weekly review" in prompt.lower():
        return "**Highlight:** A win.\n**What weighed:** Stress.\n**Carry forward:** Rest."
    if "monthly review" in prompt.lower() or "monthly" in prompt.lower():
        return "**Overall arc:** Growth. **What you learned:** Patience. **What drained you:** Overcommitting. **What nourished you:** Walks. **Intention:** Sleep earlier."
    if "14-day window" in prompt:
        return "Your music has been lighter when your journal brightens, suggesting you self-soothe with tempo."
    if "recent journal" in prompt.lower():
        return "**Emotional trend:** Upward. **Recurring themes:** Work, sleep. **Blind spot:** Rest. **Suggestion:** Block one walk a day."
    return "Mocked reply."

def fake_llama_stream(prompt, system="", temperature=0.7):
    for token in ["Hello", " from", " Innerbloom", ".", " This", " references", " [cite:", "aaaaaaaa", "]."]:
        yield token

# ---------- Spotify mocks ----------

FAKE_SPOTIFY = {
    "recently_played": {
        "items": [
            {"track": {"id": "t1", "name": "Song 1", "artists": [{"name": "A1"}],
                       "album": {"name": "Alb", "images": [{"url": "x"}]},
                       "duration_ms": 200000, "popularity": 50},
             "played_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"},
            {"track": {"id": "t2", "name": "Song 2", "artists": [{"name": "A2"}],
                       "album": {"name": "Alb2", "images": []},
                       "duration_ms": 180000, "popularity": 30},
             "played_at": datetime.utcnow().isoformat() + "Z"},
        ]
    },
    "audio_features": {"audio_features": [
        {"id": "t1", "valence": 0.7, "energy": 0.6, "danceability": 0.5, "tempo": 120.0},
        {"id": "t2", "valence": 0.3, "energy": 0.4, "danceability": 0.4, "tempo": 90.0},
    ]},
    "me": {"id": "u1", "display_name": "Tester", "email": "t@x", "images": [{"url": "pic"}]},
    "top_tracks": {"items": [
        {"id": "tt1", "name": "Top Song", "artists": [{"name": "Topper"}],
         "album": {"images": [{"url": "img"}]}, "popularity": 90}
    ]},
    "token": {"access_token": "acc", "refresh_token": "ref", "expires_in": 3600, "scope": "x"},
}

def fake_spotify_post(url, data=None, headers=None, timeout=None, **kw):
    m = MagicMock()
    if "accounts.spotify.com/api/token" in url:
        m.status_code = 200
        m.json.return_value = FAKE_SPOTIFY["token"]
        return m
    m.status_code = 404
    return m

def fake_spotify_get(url, headers=None, params=None, timeout=None, **kw):
    m = MagicMock()
    if "/me/player/recently-played" in url:
        m.status_code = 200
        m.json.return_value = FAKE_SPOTIFY["recently_played"]
    elif "/audio-features" in url:
        m.status_code = 200
        m.json.return_value = FAKE_SPOTIFY["audio_features"]
    elif url.endswith("/v1/me"):
        m.status_code = 200
        m.json.return_value = FAKE_SPOTIFY["me"]
    elif "/me/top/" in url:
        m.status_code = 200
        m.json.return_value = FAKE_SPOTIFY["top_tracks"]
    else:
        m.status_code = 404
        m.json.return_value = {}
        m.text = "not found"
    return m

def fake_health_get(url, timeout=None, **kw):
    m = MagicMock()
    if "api/tags" in url:
        m.status_code = 200
        m.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
        return m
    return fake_spotify_get(url, **kw)


# =========================================================================
# Unit tests (non-HTTP helpers)
# =========================================================================

def test_unit_helpers():
    section("Unit helpers")

    # tokenize
    toks = innerbloom_app.tokenize("I had a great day at WORK, learning python!")
    check("tokenize drops stopwords", "had" not in toks and "great" in toks,
          f"got {toks}")
    check("tokenize lowercases", "work" in toks, f"got {toks}")

    # extract_json — plain
    j = innerbloom_app.extract_json('{"a": 1}')
    check("extract_json plain", j == {"a": 1}, str(j))

    # extract_json — with prose wrapper
    j = innerbloom_app.extract_json('Sure! {"a": 2, "b": "c"} — hope that helps')
    check("extract_json extracts from prose", j == {"a": 2, "b": "c"}, str(j))

    # extract_json — trailing comma (common LLM slip)
    j = innerbloom_app.extract_json('{"a": 1, "b": 2,}')
    check("extract_json tolerant of trailing comma", j == {"a": 1, "b": 2}, str(j))

    # extract_json — single quotes
    j = innerbloom_app.extract_json("{'a': 1, 'b': 2}")
    check("extract_json tolerant of single quotes", j == {"a": 1, "b": 2}, str(j))

    # extract_json — malformed
    j = innerbloom_app.extract_json("totally not json at all")
    check("extract_json returns None on garbage", j is None)

    # score_entry + retrieve_relevant
    entries = [
        {"id": "a", "text": "I love cooking pasta with garlic", "tags": ["food"], "themes": [],
         "timestamp": datetime.now().isoformat()},
        {"id": "b", "text": "Work was brutal again today", "tags": ["work"], "themes": [],
         "timestamp": (datetime.now() - timedelta(days=30)).isoformat()},
        {"id": "c", "text": "Ate pasta and relaxed", "tags": ["food", "rest"], "themes": [],
         "timestamp": (datetime.now() - timedelta(days=1)).isoformat()},
    ]
    hits = innerbloom_app.retrieve_relevant(entries, "pasta", k=2)
    check("retrieve finds pasta entries", any(e["id"] == "a" for e in hits) and any(e["id"] == "c" for e in hits),
          f"got {[e['id'] for e in hits]}")
    check("retrieve includes recent entry as grounding", any(e["id"] == "a" for e in hits))

    # streak
    today = datetime.now().date()
    e = [{"timestamp": (datetime.now() - timedelta(days=i)).isoformat()} for i in [0, 1, 2, 5, 6]]
    s = innerbloom_app.compute_streak(e)
    check("streak current=3", s["current"] == 3, str(s))
    check("streak longest>=3", s["longest"] >= 3, str(s))

    # streak — empty
    s = innerbloom_app.compute_streak([])
    check("streak empty=0/0", s == {"current": 0, "longest": 0})

    # streak — yesterday still counts toward current streak
    e = [{"timestamp": (datetime.now() - timedelta(days=i)).isoformat()} for i in [1, 2, 3]]
    s = innerbloom_app.compute_streak(e)
    check("streak honours yesterday start", s["current"] == 3, str(s))

    # format_context returns citation-ready ids
    ctx = innerbloom_app.format_context([{"id": "abcd1234-xxxx", "text": "hi", "timestamp": "2024-01-01T00:00:00",
                                       "emotion": "calm", "tags": ["x"]}])
    check("format_context embeds short id", "id=abcd1234" in ctx, ctx[:80])

    # extract_citations
    rels = [{"id": "aaaaaaaa-1111", "title": "T", "timestamp": "2024-01-01T00:00:00", "emotion": "happy"}]
    got = innerbloom_app.extract_citations("Great thinking [cite:aaaaaaaa]", rels)
    check("extract_citations resolves short id", len(got) == 1 and got[0]["id"] == "aaaaaaaa-1111", str(got))

    # clean_reply strips tokens
    cleaned = innerbloom_app.clean_reply("Hi [cite:abc12345]. Nice.")
    check("clean_reply strips [cite:]", "[cite:" not in cleaned and "Hi" in cleaned, cleaned)


# =========================================================================
# Journal routes
# =========================================================================

def test_journal_routes():
    section("Journal routes")

    with patch.object(innerbloom_app, "call_llama", side_effect=fake_llama):
        r = client.post("/save", json={"text": "I had a great day at work today.", "mood": 7})
        check("POST /save 200", r.status_code == 200, r.text[:120])
        entry_id = r.json()["entry"]["id"]
        check("entry has analysis fields", all(k in r.json()["entry"]
              for k in ["emotion", "intensity", "summary", "tags", "themes", "word_count"]))

        # Empty entry -> 400
        r = client.post("/save", json={"text": "   "})
        check("POST /save 400 on empty", r.status_code == 400)

        # Add a second entry so we can filter
        client.post("/save", json={"text": "Another reflective evening after a long day."})

        r = client.get("/journal")
        check("GET /journal 200", r.status_code == 200)
        check("GET /journal returns entries", len(r.json()["entries"]) >= 2,
              f"got {r.json().get('total')}")

        r = client.get("/journal", params={"q": "work"})
        check("GET /journal q= filters", len(r.json()["entries"]) >= 1)

        r = client.get(f"/journal/{entry_id}")
        check("GET /journal/{id} 200", r.status_code == 200 and r.json()["id"] == entry_id)

        r = client.get("/journal/does-not-exist")
        check("GET /journal/{id} 404", r.status_code == 404)

        r = client.put(f"/journal/{entry_id}", json={"title": "Updated title"})
        check("PUT /journal title-only 200", r.status_code == 200
              and r.json()["entry"]["title"] == "Updated title")

        r = client.put(f"/journal/{entry_id}", json={"text": "Totally new text to trigger re-analysis."})
        check("PUT /journal text re-analysis 200", r.status_code == 200
              and r.json()["entry"]["summary"])

        r = client.put("/journal/nope", json={"text": "x"})
        check("PUT /journal 404", r.status_code == 404)

        r = client.delete(f"/journal/{entry_id}")
        check("DELETE /journal 200", r.status_code == 200)

        r = client.delete(f"/journal/{entry_id}")
        check("DELETE /journal 404 on second try", r.status_code == 404)


# =========================================================================
# Chat (incl. citations + streaming)
# =========================================================================

def test_chat_routes():
    section("Chat routes")
    # Seed an entry so retrieval has something
    with patch.object(innerbloom_app, "call_llama", side_effect=fake_llama):
        client.post("/save", json={"text": "Thinking about pasta and evenings at home."})

    # Force the fake reply to include a [cite:xxxx] matching a real entry id
    entries = innerbloom_app.load_file(innerbloom_app.JOURNAL_FILE)
    short_id = entries[-1]["id"][:8]
    def fake_llama_with_cite(prompt, system="", temperature=0.7, timeout=180):
        return f"A calm thought grounded in your last entry. [cite:{short_id}]"

    with patch.object(innerbloom_app, "call_llama", side_effect=fake_llama_with_cite):
        r = client.post("/chat", json={"message": "How do I feel about pasta?"})
        check("POST /chat 200", r.status_code == 200)
        body = r.json()
        check("chat reply is clean of [cite:]", "[cite:" not in body["reply"], body["reply"])
        check("chat returns resolved citations", isinstance(body["citations"], list)
              and len(body["citations"]) == 1
              and body["citations"][0]["id"].startswith(short_id), str(body["citations"]))

        r = client.post("/chat", json={"message": "  "})
        check("POST /chat 400 empty", r.status_code == 400)

        r = client.get("/chat")
        check("GET /chat returns history", r.status_code == 200 and len(r.json()["messages"]) >= 1)

    # Streaming — just assert we get bytes back that include manifest + reply
    with patch.object(innerbloom_app, "call_llama_stream", side_effect=fake_llama_stream):
        with client.stream("POST", "/chat/stream", json={"message": "Any other pasta thoughts?"}) as resp:
            chunks = b""
            for c in resp.iter_bytes():
                chunks += c
            check("POST /chat/stream streams bytes", len(chunks) > 0, f"len={len(chunks)}")
            check("POST /chat/stream prepends manifest", chunks.startswith(b"\x1e"),
                  chunks[:80].decode(errors="replace"))

    # Clear
    r = client.delete("/chat")
    check("DELETE /chat clears", r.status_code == 200
          and client.get("/chat").json()["messages"] == [])


# =========================================================================
# Insights, reflect, connections
# =========================================================================

def test_insights():
    section("Insights / reflect / connections / prompts / search / export")
    with patch.object(innerbloom_app, "call_llama", side_effect=fake_llama):
        # Seed a handful of entries
        for t in [
            "I walked today and felt calm after a busy morning.",
            "Stress at work, lots of meetings and shallow wins.",
            "Baked bread, enjoyed the quiet evening, gratitude.",
        ]:
            client.post("/save", json={"text": t})

        r = client.get("/stats")
        check("GET /stats 200", r.status_code == 200)
        s = r.json()
        check("stats has daily=30", len(s["daily"]) == 30)
        check("stats heatmap ~= 366", 300 < len(s["heatmap"]) <= 366)
        check("stats emotions non-empty", len(s["emotions"]) >= 1)

        r = client.get("/analyze")
        check("GET /analyze 200", r.status_code == 200 and r.json()["result"])

        r = client.get("/weekly-review")
        check("GET /weekly-review 200", r.status_code == 200)

        r = client.get("/monthly-review")
        check("GET /monthly-review 200", r.status_code == 200)

        # reflect on latest entry
        latest_id = innerbloom_app.load_file(innerbloom_app.JOURNAL_FILE)[-1]["id"]
        r = client.get(f"/reflect/{latest_id}")
        check("GET /reflect/{id} 200", r.status_code == 200
              and len(r.json()["questions"]) == 3, str(r.json()))

        r = client.get("/reflect/nope")
        check("GET /reflect 404", r.status_code == 404)

        r = client.get(f"/connections/{latest_id}")
        check("GET /connections/{id} 200", r.status_code == 200
              and isinstance(r.json()["related"], list))

        r = client.get("/prompt")
        check("GET /prompt 200", r.status_code == 200 and r.json()["prompt"])

        r = client.get("/search", params={"q": "bread"})
        check("GET /search finds bread", r.status_code == 200
              and len(r.json()["results"]) >= 1, str(r.json()))

        r = client.get("/export")
        check("GET /export 200 markdown", r.status_code == 200
              and "# Innerbloom" in r.text)


# =========================================================================
# Spotify (all network mocked)
# =========================================================================

def test_spotify():
    section("Spotify routes (mocked network)")

    with patch("app.requests.get", side_effect=fake_spotify_get), \
         patch("app.requests.post", side_effect=fake_spotify_post):

        # status when disconnected
        r = client.get("/spotify/status")
        check("GET /spotify/status disconnected", r.status_code == 200
              and r.json()["connected"] is False)

        # recent fails before connect
        r = client.get("/spotify/recent")
        check("GET /spotify/recent 401 before connect", r.status_code == 401)

        # set config
        r = client.post("/spotify/config", json={
            "client_id": "cid",
            "redirect_uri": "http://localhost:5500/",
        })
        check("POST /spotify/config 200", r.status_code == 200)

        # exchange code -> tokens saved
        r = client.post("/spotify/exchange", json={
            "code": "auth-code",
            "code_verifier": "v" * 43,
            "redirect_uri": "http://localhost:5500/",
        })
        check("POST /spotify/exchange 200", r.status_code == 200
              and r.json()["status"] == "connected")
        check("spotify user attached", r.json().get("user", {}).get("id") == "u1")

        r = client.get("/spotify/status")
        check("GET /spotify/status connected", r.json()["connected"] is True)

        # recent
        r = client.get("/spotify/recent")
        check("GET /spotify/recent 200", r.status_code == 200
              and len(r.json()["tracks"]) == 2)

        # top
        r = client.get("/spotify/top", params={"kind": "tracks"})
        check("GET /spotify/top 200", r.status_code == 200
              and len(r.json()["tracks"]) == 1)

        r = client.get("/spotify/top", params={"kind": "bogus"})
        check("GET /spotify/top 400 bad kind", r.status_code == 400)

    # mood + insight — uses the mocked getters plus the real LLM mock
    with patch("app.requests.get", side_effect=fake_spotify_get), \
         patch("app.requests.post", side_effect=fake_spotify_post), \
         patch.object(innerbloom_app, "call_llama", side_effect=fake_llama):
        r = client.get("/spotify/mood")
        check("GET /spotify/mood 200", r.status_code == 200)
        m = r.json()
        check("mood returns 30-point timeline", len(m["points"]) == 30, f"len={len(m.get('points', []))}")
        non_null = [p for p in m["points"] if p["valence"] is not None]
        check("mood has some valence datapoints", len(non_null) >= 1, f"non_null={len(non_null)}")
        check("avg_valence computed", m["avg_valence"] is not None)

        r = client.get("/spotify/insight")
        check("GET /spotify/insight 200", r.status_code == 200 and r.json()["insight"])

    # disconnect
    with patch("app.requests.get", side_effect=fake_spotify_get), \
         patch("app.requests.post", side_effect=fake_spotify_post):
        r = client.post("/spotify/disconnect")
        check("POST /spotify/disconnect 200", r.status_code == 200)
        r = client.get("/spotify/status")
        check("status disconnected after disconnect", r.json()["connected"] is False)


# =========================================================================
# Health
# =========================================================================

def test_health():
    section("Health")
    r = client.get("/")
    check("GET / 200", r.status_code == 200 and r.json()["status"] == "ok")

    with patch("app.requests.get", side_effect=fake_health_get):
        r = client.get("/health")
        check("GET /health up when ollama responds", r.status_code == 200
              and r.json()["ollama"] == "up")

    # simulate ollama down
    def raise_err(*a, **kw): raise Exception("refused")
    with patch("app.requests.get", side_effect=raise_err):
        r = client.get("/health")
        check("GET /health down when ollama refused", r.json()["ollama"] == "down")


# =========================================================================
# People (social CRM)
# =========================================================================

def test_people():
    section("People (social CRM)")
    fake_analysis = {
        "emotion": "happy", "intensity": 6, "summary": "Coffee with Bob.",
        "tags": ["coffee"], "themes": ["friendship"], "people": ["Bob"],
    }
    with patch.object(innerbloom_app, "analyze_entry", return_value=fake_analysis), \
         patch.object(innerbloom_app, "embed", return_value=None):
        r = client.post("/save", json={"text": "Coffee with Bob was lovely."})
        check("POST /save 200 (people)", r.status_code == 200, r.text[:120])

    r = client.get("/people")
    people = r.json()["people"]
    bob = next((p for p in people if p["key"] == "bob"), None)
    check("GET /people auto-created Bob", bob is not None, str([p["name"] for p in people]))
    check("Bob has a CRM record", bool(bob and bob["has_record"]))
    check("Bob mention counted", bool(bob and bob["mentions"] >= 1))

    r = client.put("/people/bob", json={"relationship": "friend", "cadence_days": 7, "birthday": "03-12"})
    check("PUT /people/{key} 200", r.status_code == 200 and r.json()["relationship"] == "friend")
    check("birthday_in_days computed", r.json().get("birthday_in_days") is not None)

    r = client.put("/people/bob", json={"relationship": "enemy"})
    check("PUT /people invalid relationship 400", r.status_code == 400)

    r = client.post("/people/bob/notes", json={"text": "Rough patch at work — check in next week."})
    check("POST /people/{key}/notes 200", r.status_code == 200)
    nid = r.json()["note"]["id"]

    r = client.get("/people/bob")
    check("GET /people/{key} 200 + note present", r.status_code == 200
          and any(n["id"] == nid for n in r.json()["notes"]))

    r = client.post("/people/bob/notes", json={"text": "   "})
    check("POST empty note 400", r.status_code == 400)

    r = client.delete(f"/people/bob/notes/{nid}")
    check("DELETE note 200", r.status_code == 200)
    r = client.delete(f"/people/bob/notes/{nid}")
    check("DELETE note 404 second time", r.status_code == 404)

    client.put("/people/bob", json={"hidden": True})
    vis = client.get("/people").json()["people"]
    allp = client.get("/people", params={"include_hidden": "true"}).json()["people"]
    check("hidden excluded by default", all(p["key"] != "bob" for p in vis))
    check("hidden shown with include_hidden", any(p["key"] == "bob" for p in allp))

    r = client.get("/people/zzzznobody")
    check("GET /people/{unknown} 404", r.status_code == 404)

    # --- catch-up clears an overdue flag ---
    # Seed an old mention so the check-in is overdue against a 7-day cadence.
    old_ts = (datetime.now() - timedelta(days=40)).isoformat()
    entries = innerbloom_app.load_file(innerbloom_app.JOURNAL_FILE)
    entries.append({
        "id": "carol-old", "title": "t", "text": "Lunch with Carol.",
        "summary": "Lunch with Carol.", "emotion": "happy", "intensity": 5,
        "tags": [], "themes": [], "people": ["Carol"],
        "word_count": 3, "timestamp": old_ts,
    })
    innerbloom_app.save_file(innerbloom_app.JOURNAL_FILE, entries)

    client.put("/people/carol", json={"cadence_days": 7})
    r = client.get("/people/carol").json()
    check("Carol overdue before catch-up", r["overdue"] is True, str(r.get("days_since_last_seen")))

    r = client.post("/people/carol/checkin")
    check("POST /people/{key}/checkin 200", r.status_code == 200)
    check("catch-up clears overdue", r.json()["overdue"] is False
          and r.json()["days_since_contact"] == 0, str(r.json()))


def test_insight_write_concurrency():
    section("Insight cache concurrency")
    # The four engines write insights.json concurrently from refresh-all.
    # _update_insight must serialise the read-modify-write so no key is lost.
    innerbloom_app.save_file(innerbloom_app.INSIGHTS_FILE, {})
    keys = [f"engine_{i}" for i in range(12)]

    def writer(k):
        # tiny stagger maximises interleave odds
        time.sleep(0.001)
        innerbloom_app._update_insight(k, {"ok": True, "k": k})

    threads = [threading.Thread(target=writer, args=(k,)) for k in keys]
    for t in threads: t.start()
    for t in threads: t.join()

    cache = innerbloom_app._load_insights()
    missing = [k for k in keys if k not in cache]
    check("concurrent _update_insight loses no keys", not missing,
          f"missing: {missing}")


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    t0 = time.time()
    try:
        test_unit_helpers()
        test_journal_routes()
        test_chat_routes()
        test_insights()
        test_spotify()
        test_people()
        test_insight_write_concurrency()
        test_health()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"PASS: {PASS}    FAIL: {FAIL}    time: {time.time()-t0:.2f}s")
    if FAIL:
        print("\nFailures:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("All green.")
