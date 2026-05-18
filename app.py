"""
Innerbloom — Advanced AI Journal Backend
----------------------------------------
FastAPI + Ollama (llama3.2:3b) + Spotify integration

Features:
- Structured entry analysis (emotion, intensity, tags, themes, summary)
- RAG chat ("talk to your journal") with citations to source entries
- Entry CRUD with edit / delete
- Full-text search
- Statistics (streak, word count, mood distribution, per-day volume)
- Adaptive writing prompts based on recent context
- Reflection questions per entry
- Weekly + monthly reviews, long-term analysis
- "Connections" — find related past entries
- Spotify OAuth (PKCE) + listening-mood correlation
- Markdown export
- Token streaming for chat

All data stored locally as JSON; nothing leaves the machine except Spotify
traffic when the user explicitly connects.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import requests
import json
import math
import os
import re
import uuid
import time
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.environ.get("INNERBLOOM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("INNERBLOOM_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("INNERBLOOM_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.environ.get("INNERBLOOM_OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_EMBED_URL = os.environ.get("INNERBLOOM_OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")

DATA_DIR = os.environ.get(
    "INNERBLOOM_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
os.makedirs(DATA_DIR, exist_ok=True)

JOURNAL_FILE = os.path.join(DATA_DIR, "journal.json")
CHAT_FILE = os.path.join(DATA_DIR, "chat.json")
SPOTIFY_FILE = os.path.join(DATA_DIR, "spotify.json")  # config + tokens
INSIGHTS_FILE = os.path.join(DATA_DIR, "insights.json")  # phase-3 insight engines

STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from into out up
down is are was were be been being am has have had do does did doing i me my
mine we us our ours you your yours he him his she her hers it its they them
their theirs this that these those as not no yes also very really just too
about over under again further here there when where why how all any both
each few more most other some such only own same than can will would should
could may might must shall need dare get got go going went come came make
made take took see saw look looked know knew think thought feel felt want
wanted tell told said say says like liked way ways thing things time today
yesterday tomorrow day days week weeks month months year years
""".split())

app = FastAPI(title="Innerbloom — AI Journal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Browsers ignore credentials when origin is "*". We never send cookies
    # to this API anyway — tokens (Spotify) are stored server-side — so
    # disabling credentials keeps the CORS contract correct.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class EntryInput(BaseModel):
    text: str
    title: Optional[str] = None
    mood: Optional[int] = None

class EntryUpdate(BaseModel):
    text: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = None

class ChatInput(BaseModel):
    message: str
    mode: Optional[str] = "journal"          # "journal" | "general"
    personality: Optional[str] = "companion"  # see PERSONALITIES below

class SpotifyConfig(BaseModel):
    client_id: str
    redirect_uri: str

class SpotifyExchange(BaseModel):
    code: str
    code_verifier: str
    redirect_uri: str


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_file(path, default=None):
    if default is None:
        default = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_file(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _safe_dt(s):
    """Parse an ISO timestamp safely. Returns None on bad input —
    callers use this as a 'do this entry have a usable date?' filter."""
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def call_llama(prompt: str, system: str = "", temperature: float = 0.7, timeout: int = 180) -> str:
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def call_llama_stream(prompt: str, system: str = "", temperature: float = 0.7):
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    chunk = obj.get("response", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        yield f"[LLM error: {e}]"


def embed(text: str) -> Optional[List[float]]:
    """Generate an embedding via Ollama. Returns None if the service is
    unreachable or the input is empty — callers should treat that as a
    signal to fall back to keyword retrieval."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text[:8000]},
            timeout=60,
        )
        vec = r.json().get("embedding")
        if isinstance(vec, list) and vec:
            return vec
    except Exception:
        pass
    return None


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    blob = match.group(0)
    try:
        return json.loads(blob)
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", blob)
        cleaned = cleaned.replace("'", '"')
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Entry analysis
# ---------------------------------------------------------------------------

EMOTIONS = [
    "happy", "sad", "anxious", "calm", "grateful", "frustrated",
    "excited", "reflective", "hopeful", "tired", "angry", "content",
    "lonely", "proud", "overwhelmed", "neutral",
]

def analyze_entry(text: str) -> Dict[str, Any]:
    prompt = f"""You are an assistant that classifies a journal entry.
Return ONLY a compact JSON object. No prose. No markdown fences.

Schema:
{{
  "emotion": "one of: {', '.join(EMOTIONS)}",
  "intensity": integer 1-10,
  "summary": "one concise sentence, <= 20 words",
  "tags": ["2-4 lowercase single-word topic tags"],
  "themes": ["1-3 short themes, 1-3 words each"]
}}

Entry:
\"\"\"{text}\"\"\"

JSON:"""
    raw = call_llama(prompt, temperature=0.3)
    data = extract_json(raw) or {}

    emotion = str(data.get("emotion", "neutral")).lower().strip()
    if emotion not in EMOTIONS:
        emotion = "neutral"
    try:
        intensity = int(data.get("intensity", 5))
        intensity = max(1, min(10, intensity))
    except Exception:
        intensity = 5
    summary = str(data.get("summary", "")).strip()[:280]
    tags = data.get("tags") or []
    themes = data.get("themes") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    if isinstance(themes, str):
        themes = [t.strip() for t in themes.split(",")]
    tags = [re.sub(r"[^a-z0-9\-]", "", str(t).lower())[:24] for t in tags if t][:5]
    tags = [t for t in tags if t]
    themes = [str(t).strip()[:40] for t in themes if t][:3]

    return {
        "emotion": emotion,
        "intensity": intensity,
        "summary": summary,
        "tags": tags,
        "themes": themes,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase tokens with stopwords removed. Keeps short emotional anchors
    like 'ex', 'dad', 'mom' — those are exactly what people search for."""
    raw = re.findall(r"[a-zA-Z][a-zA-Z'\-]*", (text or "").lower())
    return [w for w in raw if w not in STOPWORDS and len(w) >= 2]

def score_entry(entry: dict, query_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    haystack = " ".join([
        entry.get("text", ""),
        entry.get("summary", ""),
        " ".join(entry.get("tags", [])),
        " ".join(entry.get("themes", [])),
    ]).lower()
    score = 0.0
    for t in query_tokens:
        score += haystack.count(t)
    try:
        ts = datetime.fromisoformat(entry["timestamp"])
        age = (datetime.now() - ts).total_seconds() / 86400.0
        score += max(0.0, 1.0 - age / 30.0) * 0.5
    except Exception:
        pass
    return score

def retrieve_relevant(entries: List[dict], query: str, k: int = 5) -> List[dict]:
    if not entries:
        return []
    tokens = tokenize(query)
    scored = [(score_entry(e, tokens), e) for e in entries]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [e for s, e in scored if s > 0][:k]
    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:2]
    seen = {e["id"] for e in hits}
    for r in recent:
        if r["id"] not in seen:
            hits.append(r); seen.add(r["id"])
    return hits[:k + 2]


def format_context(entries: List[dict]) -> str:
    lines = []
    for e in entries:
        date_str = e.get("timestamp", "")[:10]
        snippet = e.get("text", "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        tags = ", ".join(e.get("tags", []))
        lines.append(
            f"[id={e.get('id','')[:8]} | {date_str} | emotion={e.get('emotion','?')} | tags={tags}]\n{snippet}"
        )
    return "\n\n".join(lines) if lines else "(no past entries)"


# ---------------------------------------------------------------------------
# Semantic memory — embeddings, similarity, unified retrieval
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# Memory tuning knobs. Pulled out so both retrievers can read them and so a
# future user-config endpoint can adjust without touching call sites.
RECENCY_HALF_LIFE_DAYS = 30.0   # cosine score is multiplied by 0.5^(age/HL)
RECENCY_FLOOR = 0.55            # never decay below this (old but relevant > recent fluff)
MMR_LAMBDA = 0.7                # 1.0 = pure relevance, 0.0 = pure diversity


def _age_days(item: dict) -> float:
    ts = _safe_dt(item.get("timestamp") or "")
    if not ts:
        return 0.0
    return max(0.0, (datetime.now() - ts).total_seconds() / 86400.0)


def _recency_weight(item: dict) -> float:
    """Soft exponential decay with a floor — recent things win ties, but a
    relevant 6-month-old entry still beats a vague yesterday entry."""
    age = _age_days(item)
    decay = 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)
    return max(RECENCY_FLOOR, decay)


def _mmr_select(scored: List[tuple], k: int,
                vector_field: str = "embedding") -> List[dict]:
    """Maximal Marginal Relevance — pick k items balancing relevance against
    diversity from items already picked. Prevents three near-duplicate days
    about the same topic dominating the top-k."""
    if not scored:
        return []
    pool = list(scored)
    picked: List[tuple] = []
    while pool and len(picked) < k:
        if not picked:
            best = max(pool, key=lambda x: x[0])
        else:
            def mmr(cand):
                rel = cand[0]
                v = cand[1].get(vector_field) or []
                if not v:
                    return MMR_LAMBDA * rel  # no embedding → no penalty
                max_sim = max(
                    (cosine_similarity(v, p[1].get(vector_field) or []) for p in picked),
                    default=0.0,
                )
                return MMR_LAMBDA * rel - (1.0 - MMR_LAMBDA) * max_sim
            best = max(pool, key=mmr)
        picked.append(best)
        pool.remove(best)
    return [it for _, it in picked]


def retrieve_similar(query: str, items: List[dict], k: int = 5,
                     embed_field: str = "embedding",
                     apply_recency: bool = True,
                     diversify: bool = True) -> List[dict]:
    """Embed the query, score by cosine × recency, return top k with MMR
    diversification. Items missing an embedding are skipped — use
    backfill_*_embeddings to populate them lazily."""
    qvec = embed(query)
    if not qvec:
        return []
    scored = []
    for it in items:
        vec = it.get(embed_field)
        if not vec:
            continue
        s = cosine_similarity(qvec, vec)
        if s <= 0:
            continue
        if apply_recency:
            s *= _recency_weight(it)
        scored.append((s, it))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    # Look at a small over-fetch window for MMR to have room to diversify.
    head = scored[: max(k * 3, k + 2)]
    if diversify and len(head) > k:
        return _mmr_select(head, k, vector_field=embed_field)
    return [it for _, it in head[:k]]


def _entry_embed_text(entry: dict) -> str:
    parts = [entry.get("text", ""), entry.get("summary", "")]
    return "\n".join(p for p in parts if p)


def _chat_embed_text(turn: dict) -> str:
    return f"{turn.get('user','')}\n{turn.get('ai','')}".strip()


# How many missing embeddings to fill in one chat request. Cold-start with
# 200 entries used to hang the first chat as we synchronously embedded all of
# them. We now top up at most this many per call — by the third or fourth
# message everything is filled in.
BACKFILL_BUDGET_PER_CALL = 12


def backfill_entry_embeddings(entries: List[dict],
                              budget: int = BACKFILL_BUDGET_PER_CALL) -> bool:
    """Fill missing entry embeddings in place, capped per call. Newest entries
    are filled first so what the user just wrote is searchable immediately."""
    changed = False
    filled = 0
    pending = sorted(
        [e for e in entries if not e.get("embedding")],
        key=lambda e: e.get("timestamp", ""),
        reverse=True,
    )
    for e in pending:
        if filled >= budget:
            break
        vec = embed(_entry_embed_text(e))
        if vec:
            e["embedding"] = vec
            changed = True
            filled += 1
    return changed


def backfill_chat_embeddings(history: List[dict],
                             budget: int = BACKFILL_BUDGET_PER_CALL) -> bool:
    changed = False
    filled = 0
    pending = sorted(
        [t for t in history if not t.get("embedding")],
        key=lambda t: t.get("timestamp", ""),
        reverse=True,
    )
    for t in pending:
        if filled >= budget:
            break
        vec = embed(_chat_embed_text(t))
        if vec:
            t["embedding"] = vec
            changed = True
            filled += 1
    return changed


def _surfaced_insights(query: str) -> List[dict]:
    """Pull the most relevant items from cached insight engines (contradictions,
    triggers, wellbeing, narrative). These are *short, structured* facts the
    chat router can inject so the model can reference what the app already
    knows about the user instead of re-deriving it on every turn."""
    cache = load_file(INSIGHTS_FILE, default={})
    if not isinstance(cache, dict) or not cache:
        return []

    out: List[dict] = []
    ql = (query or "").lower()

    # Contradictions — short "stated vs. behavior" facts.
    for it in (cache.get("contradictions", {}) or {}).get("items", [])[:6]:
        stated = (it.get("stated") or "").strip()
        behavior = (it.get("behavior") or "").strip()
        if stated and behavior:
            out.append({"kind": "contradiction",
                        "text": f"Contradiction — said: \"{stated[:120]}\"; did: \"{behavior[:120]}\"."})

    # Triggers — patterns with mood direction.
    for it in (cache.get("triggers", {}) or {}).get("items", [])[:6]:
        label = (it.get("label") or "").strip()
        outcome = (it.get("outcome") or "").strip()
        if label:
            tail = f" → {outcome}" if outcome else ""
            out.append({"kind": "trigger",
                        "text": f"Trigger ({it.get('direction','?')}): {label}{tail}"})

    # Wellbeing — only mention if the level is above 'ok'.
    wb = cache.get("wellbeing") or {}
    for key in ("burnout", "spiral"):
        sig = wb.get(key) or {}
        if sig.get("level") and sig["level"] != "ok" and sig.get("summary"):
            out.append({"kind": f"wellbeing.{key}",
                        "text": f"Wellbeing/{key} ({sig['level']}): {sig['summary'][:240]}"})

    # Narrative — identity lines + becoming statement.
    nar = cache.get("narrative") or {}
    for line in (nar.get("identity_lines") or [])[:3]:
        out.append({"kind": "identity", "text": f"Identity: {line[:160]}"})
    if nar.get("becoming"):
        out.append({"kind": "becoming",
                    "text": f"Direction: {nar['becoming'][:200]}"})

    # If query is small-talk-ish, keep at most 4 insights to avoid clutter.
    if len(ql.split()) <= 3:
        return out[:4]
    return out[:10]


def retrieve_memory(query: str) -> Dict[str, List[dict]]:
    """Unified semantic memory router.

    Returns relevant journal entries, past chat turns, and learned insights
    about the user. Falls back to keyword retrieval for entries when the
    embedding service is unavailable, and always includes the 1–2 most
    recent entries so the model has a sense of what's happening *now*.
    """
    entries = load_file(JOURNAL_FILE)
    history = load_file(CHAT_FILE)

    # Lazy-backfill: any item without an embedding gets one now, then we
    # persist once. This keeps old data working without a migration script.
    if backfill_entry_embeddings(entries):
        save_file(JOURNAL_FILE, entries)
    if backfill_chat_embeddings(history):
        save_file(CHAT_FILE, history)

    top_entries = retrieve_similar(query, entries, k=5)
    if not top_entries and entries:
        # Embedding service down or no embeddings yet — fall back to keyword.
        top_entries = retrieve_relevant(entries, query, k=5)

    # Always keep 1–2 most recent entries in scope as a fallback so the
    # model never feels blind to the present.
    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:2]
    seen = {e["id"] for e in top_entries}
    for r in recent:
        if r["id"] not in seen:
            top_entries.append(r)
            seen.add(r["id"])

    top_chat = retrieve_similar(query, history, k=3)

    # Pull cached findings from the insight engines (contradictions, triggers,
    # wellbeing, narrative) so chat replies can reference what the app
    # already knows about the user instead of re-deriving on every turn.
    insights = _surfaced_insights(query)

    return {"entries": top_entries[:7], "chat": top_chat, "insights": insights}


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def build_context(message: str, memory: Dict[str, List[dict]]) -> str:
    """Compose a structured prompt from the memory router output.

    Sections are emitted only when they have content. Total budget is held
    well under ~2000 tokens by truncating long entries and chat turns.
    """
    parts: List[str] = []

    entries = memory.get("entries", [])
    if entries:
        block = ["=== Relevant Journal Entries ==="]
        for e in entries:
            date_s = (e.get("timestamp") or "")[:10]
            tags = ", ".join(e.get("tags", []))
            block.append(
                f"[id={e.get('id','')[:8]} | {date_s} | emotion={e.get('emotion','?')} | tags={tags}]"
            )
            block.append(_truncate(e.get("text", ""), 400))
            block.append("")
        parts.append("\n".join(block).rstrip())

    chat_hits = memory.get("chat", [])
    if chat_hits:
        block = ["=== Relevant Past Conversations ==="]
        for t in chat_hits:
            date_s = (t.get("timestamp") or "")[:10]
            block.append(f"[{date_s}]")
            block.append(f"User: {_truncate(t.get('user',''), 240)}")
            block.append(f"Innerbloom: {_truncate(t.get('ai',''), 240)}")
            block.append("")
        parts.append("\n".join(block).rstrip())

    # Patterns section: surfaced insights about the user that the
    # contradictions / triggers / wellbeing / narrative engines have produced.
    pattern_lines: List[str] = []
    for ins in memory.get("insights", []):
        text = ins.get("text") if isinstance(ins, dict) else str(ins)
        if text:
            pattern_lines.append(f"- {text}")
    if pattern_lines:
        parts.append("=== Known Patterns About User ===\n" + "\n".join(pattern_lines))

    parts.append(f"=== Current Message ===\n{message}\n\nInnerbloom:")
    return "\n\n".join(parts)


def _strip_embedding(obj: dict) -> dict:
    """Return a shallow copy without the 'embedding' field — we never want
    to ship 768 floats per entry over the wire."""
    if not isinstance(obj, dict):
        return obj
    return {k: v for k, v in obj.items() if k != "embedding"}


# ---------------------------------------------------------------------------
# Journal routes
# ---------------------------------------------------------------------------

@app.post("/save")
def save_journal(data: EntryInput):
    text = (data.text or "").strip()
    if not text:
        raise HTTPException(400, "Entry is empty.")

    analysis = analyze_entry(text)

    entry = {
        "id": str(uuid.uuid4()),
        "title": (data.title or analysis["summary"] or text[:60]).strip(),
        "text": text,
        "summary": analysis["summary"],
        "emotion": analysis["emotion"],
        "intensity": analysis["intensity"],
        "tags": analysis["tags"],
        "themes": analysis["themes"],
        "user_mood": data.mood,
        "word_count": len(text.split()),
        "timestamp": datetime.now().isoformat(),
    }
    entry["embedding"] = embed(_entry_embed_text(entry))

    entries = load_file(JOURNAL_FILE)
    entries.append(entry)
    save_file(JOURNAL_FILE, entries)
    return {"status": "saved", "entry": _strip_embedding(entry)}


@app.get("/journal")
def get_journal(q: Optional[str] = None, tag: Optional[str] = None,
                emotion: Optional[str] = None, limit: int = 500):
    entries = load_file(JOURNAL_FILE)
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    if q:
        ql = q.lower()
        entries = [e for e in entries if ql in (e.get("text", "") + " " +
                                                e.get("summary", "") + " " +
                                                " ".join(e.get("tags", []))).lower()]
    if tag:
        tl = tag.lower()
        entries = [e for e in entries if tl in [t.lower() for t in e.get("tags", [])]]
    if emotion:
        el = emotion.lower()
        entries = [e for e in entries if (e.get("emotion", "") or "").lower() == el]

    return {
        "entries": [_strip_embedding(e) for e in entries[:limit]],
        "total": len(entries),
    }


@app.get("/journal/{entry_id}")
def get_entry(entry_id: str):
    entries = load_file(JOURNAL_FILE)
    for e in entries:
        if e["id"] == entry_id:
            return _strip_embedding(e)
    raise HTTPException(404, "Entry not found")


@app.put("/journal/{entry_id}")
def update_entry(entry_id: str, data: EntryUpdate):
    entries = load_file(JOURNAL_FILE)
    for i, e in enumerate(entries):
        if e["id"] == entry_id:
            if data.text is not None:
                e["text"] = data.text.strip()
                e["word_count"] = len(e["text"].split())
                analysis = analyze_entry(e["text"])
                e.update({
                    "summary": analysis["summary"],
                    "emotion": analysis["emotion"],
                    "intensity": analysis["intensity"],
                    "tags": analysis["tags"],
                    "themes": analysis["themes"],
                })
                # Text changed → embedding is stale; regenerate.
                e["embedding"] = embed(_entry_embed_text(e))
            if data.title is not None:
                e["title"] = data.title.strip()
            if data.tags is not None:
                e["tags"] = [str(t).lower().strip() for t in data.tags if t]
            e["updated_at"] = datetime.now().isoformat()
            entries[i] = e
            save_file(JOURNAL_FILE, entries)
            return {"status": "updated", "entry": _strip_embedding(e)}
    raise HTTPException(404, "Entry not found")


@app.delete("/journal/{entry_id}")
def delete_entry(entry_id: str):
    entries = load_file(JOURNAL_FILE)
    new = [e for e in entries if e["id"] != entry_id]
    if len(new) == len(entries):
        raise HTTPException(404, "Entry not found")
    save_file(JOURNAL_FILE, new)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Chat — personalities, modes, safety (RAG with citations)
# ---------------------------------------------------------------------------

PERSONALITIES = {
    "companion": (
        "You are Innerbloom — a thoughtful friend who has read every entry in the user's journal. "
        "ALWAYS speak TO the user directly using 'you' / 'your'. Never refer to them as 'they' or 'the user' or 'this person'. "
        "Talk like a friend would, not a chatbot. Use contractions. Be warm and curious, not formal. "
        "When you reference something they wrote, sound like you remember it ('that Sunday thing you wrote about'), "
        "not like you looked it up. Notice patterns out loud. Ask a follow-up question when it would help — "
        "not as a way to dodge the question they asked."
    ),
    "observer": (
        "You are Innerbloom in observer mode — a quiet, attentive reader of the user's journal. "
        "ALWAYS address the user directly with 'you' / 'your'. Never refer to them in third person. "
        "Speak with restraint. Notice patterns the way a careful friend does — 'three of the last four Sundays "
        "you wrote about dread' rather than 'you seem anxious'. Distinguish what you can see from what you're "
        "inferring. Never moralize. Never prescribe. Plain language. Short, exact sentences."
    ),
    "challenger": (
        "You are Innerbloom in challenger mode — affectionate, direct, and unafraid to name the gap between "
        "what the user says and what they actually do. "
        "ALWAYS speak TO the user using 'you' / 'your'. Never write about them as 'they' or 'this person'. "
        "Care is the foundation; bluntness is the tool. Never moralize, never lecture. "
        "When you push, push from inside their own words. Short sentences. No motivational language. No 'you should'."
    ),
}

# Map old personality keys to the new ones so existing chat history keeps working.
PERSONALITY_ALIASES = {
    "honest_coach": "challenger",
    "calm_therapist": "companion",
    "analytical_observer": "observer",
}


def _resolve_personality(key: Optional[str]) -> str:
    """Accept new keys, legacy keys, or default to 'companion'."""
    if key in PERSONALITIES:
        return key
    if key in PERSONALITY_ALIASES:
        return PERSONALITY_ALIASES[key]
    return "companion"

JOURNAL_MODE_RULES = (
    "You have the user's journal entries and past conversations in the context above. "
    "Answer the user's question directly — even when the entries don't cover it perfectly, do your best with what's there. "
    "Reference specific entries inline with [cite:id] using the short id shown in brackets. "
    "Do NOT default to asking the user for clarification — plain questions like 'what themes come up' or 'what was my happiest day' are NEVER ambiguous; just answer. "
    "Do NOT open with 'Based on your recent journal entries...' or any chatbot-style preamble. Talk like a person. "
    "Do NOT enumerate entry titles in parentheses; weave them into prose instead. "
    "If something genuinely isn't in the entries, say so in one short sentence and offer your best read anyway — never refuse, never bounce the question back."
)

GENERAL_MODE_RULES = (
    "You are in 'general' mode — the user is treating you as a normal assistant, not as their journal. "
    "Do NOT pull from journal entries unless the user explicitly asks. "
    "Help with whatever they bring: questions, ideas, planning, code, writing. "
    "Stay concise. Talk like a person. Ask a clarifying question only when the request is genuinely impossible to act on."
)


def build_system_prompt(mode: str, personality: str) -> str:
    p = PERSONALITIES[_resolve_personality(personality)]
    rules = GENERAL_MODE_RULES if mode == "general" else JOURNAL_MODE_RULES
    return (
        p
        + "\n\n" + rules
        + "\n\nLength: keep replies tight. 2–5 sentences unless the user explicitly asks for depth. "
          "Voice: natural, like a real person speaking. Contractions are fine. "
          "Never use phrases like 'I couldn't find', 'Based on your entries', 'It seems', 'It appears' — "
          "just say the thing."
    )


# Crisis detection ----------------------------------------------------------
#
# Two-stage: a fast keyword pass (no LLM) catches the most common signals,
# then for anything ambiguous we ask the model. We never block a reply — the
# user always gets an answer — we just attach a `safety` block so the UI can
# surface resources alongside it.

CRISIS_KEYWORDS = [
    r"\bsuicide\b", r"\bsuicidal\b", r"\bkill (?:my)?self\b", r"\bkms\b",
    r"\bend (?:it|my life|things)\b", r"\bdon'?t want to (?:be here|live|exist)\b",
    r"\bwant to die\b", r"\bbetter off (?:dead|without me)\b",
    r"\bself[- ]harm\b", r"\bcut(?:ting)? myself\b", r"\bhurt myself\b",
    r"\bno reason to live\b", r"\bcan'?t go on\b", r"\bgive up on life\b",
]
CRISIS_RESOURCES = [
    {"name": "iCall (India)", "detail": "+91 9152987821 · Mon–Sat 8am–10pm"},
    {"name": "AASRA (India)", "detail": "+91 9820466726 · 24/7"},
    {"name": "988 Suicide & Crisis Lifeline (US)", "detail": "Call or text 988 · 24/7"},
    {"name": "Samaritans (UK & ROI)", "detail": "116 123 · 24/7"},
    {"name": "Find a Helpline (worldwide)", "detail": "https://findahelpline.com"},
]


def crisis_check(text: str) -> Optional[dict]:
    """Return a safety block if the message likely indicates suicidal ideation
    or self-harm, otherwise None. Two-stage: regex first, then a tight LLM
    classifier for ambiguous cases like 'I just want it all to stop'."""
    if not text:
        return None
    low = text.lower()

    keyword_hit = any(re.search(p, low) for p in CRISIS_KEYWORDS)

    # Fast path — clear keyword match → high severity, skip LLM.
    if keyword_hit:
        return {
            "level": "high",
            "reason": "explicit self-harm or suicidal language",
            "resources": CRISIS_RESOURCES,
            "message": (
                "What you're carrying sounds heavy, and I'm glad you wrote it down. "
                "You don't have to figure this out alone — please reach out to one of the lines below. "
                "They're free, confidential, and trained for exactly this."
            ),
        }

    # Slow path — only run the classifier for messages with strong negative
    # affect words. Cheap heuristic: avoids LLM call on every "I feel sad".
    soft_signals = ("hopeless", "pointless", "no point", "worthless",
                    "can't do this", "cant do this", "exhausted", "numb",
                    "nothing matters", "want it to stop", "tired of")
    if not any(s in low for s in soft_signals):
        return None

    prompt = (
        "Classify the following message for crisis risk. Reply with ONE of:\n"
        "  NONE       — no concerning content\n"
        "  DISTRESS   — significant distress but no self-harm/suicidal ideation\n"
        "  CRISIS     — possible self-harm or suicidal ideation, even if implicit\n"
        "Reply with only the single word.\n\n"
        f"Message: \"\"\"{text[:500]}\"\"\""
    )
    verdict = call_llama(prompt, temperature=0.0, timeout=20).strip().upper()
    if "CRISIS" in verdict:
        return {
            "level": "high",
            "reason": "model flagged crisis-level distress",
            "resources": CRISIS_RESOURCES,
            "message": (
                "I want to check in — what you wrote sounds really heavy. "
                "If any part of you is thinking about not being here, please reach out to one of the lines below. "
                "You don't have to be sure it's 'serious enough' to call."
            ),
        }
    if "DISTRESS" in verdict:
        return {
            "level": "soft",
            "reason": "elevated distress",
            "resources": CRISIS_RESOURCES[:2],
            "message": (
                "Sounds like a hard moment. If it gets heavier, these lines are there — "
                "no need to be in 'crisis' to use them."
            ),
        }
    return None

def extract_citations(reply: str, relevant: List[dict]) -> List[dict]:
    """Parse [cite:XXXX] short-ids and resolve back to full entries."""
    short_ids = set(re.findall(r"\[cite:([a-z0-9\-]{4,})\]", reply))
    cites = []
    for e in relevant:
        sid = e["id"][:8]
        if sid in short_ids or e["id"] in short_ids:
            cites.append({
                "id": e["id"],
                "title": e.get("title") or e.get("summary") or "",
                "date": e.get("timestamp", "")[:10],
                "emotion": e.get("emotion", "neutral"),
            })
    return cites


def clean_reply(reply: str) -> str:
    """Strip [cite:xxx] tokens and any punctuation left dangling around them.

    The model often writes things like:
      - "that thing about Sunday ([cite:abc])"  →  "that thing about Sunday"
      - "X [cite:abc] and Y"                    →  "X and Y"
      - "(X [cite:abc], Y [cite:def])"          →  "(X, Y)"
    """
    out = reply or ""
    # 1. Strip cite tokens that sit alone inside their own parens/brackets.
    out = re.sub(r"\s*[\(\[]\s*\[cite:[a-z0-9\-]+\]\s*[\)\]]", "", out)
    # 2. Strip cite tokens that follow a comma inside a list ("X, [cite:abc]")
    out = re.sub(r",\s*\[cite:[a-z0-9\-]+\]", "", out)
    # 3. Strip remaining bare cite tokens.
    out = re.sub(r"\s*\[cite:[a-z0-9\-]+\]", "", out)
    # 4. Clean up the empty containers / leftover punctuation we may have made.
    out = re.sub(r"[\(\[]\s*[\)\]]", "", out)        # empty parens/brackets
    out = re.sub(r"\s+,", ",", out)                   # " ," → ","
    out = re.sub(r",\s*([.,;:!?])", r"\1", out)        # ", ." → "."
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)         # " ." → "."
    out = re.sub(r"\(\s*,\s*", "(", out)               # "( , X)" → "(X)"
    out = re.sub(r",\s*\)", ")", out)                  # "(X ,)" → "(X)"
    out = re.sub(r"[ \t]{2,}", " ", out)                # collapse double-spaces
    return out.strip()


def _chat_memory_for_mode(msg: str, mode: str) -> Dict[str, List[dict]]:
    """In journal mode: full semantic memory. In general mode: only the
    recent chat history is relevant — keep the assistant general-purpose."""
    if mode == "general":
        history = load_file(CHAT_FILE)
        if backfill_chat_embeddings(history):
            save_file(CHAT_FILE, history)
        return {
            "entries": [],
            "chat": retrieve_similar(msg, history, k=3),
            "insights": [],
        }
    return retrieve_memory(msg)


# ---------------------------------------------------------------------------
# Agent — structured planner + tool loop
#
# Two-stage architecture (more reliable on a 3B model than ReAct):
#   1) Planner LLM picks 0–3 tools to run, or asks a clarifying question
#   2) We execute the tools deterministically (Python, no LLM)
#   3) Drafter LLM writes the reply with the observations as context
#
# The endpoint streams JSONL events so the UI can show "agent working" steps.
# ---------------------------------------------------------------------------

# --- tools -----------------------------------------------------------------

def tool_search_entries(query: str, k: int = 5) -> List[dict]:
    entries = load_file(JOURNAL_FILE)
    if backfill_entry_embeddings(entries):
        save_file(JOURNAL_FILE, entries)
    hits = retrieve_similar(query, entries, k=k)
    if not hits:
        hits = retrieve_relevant(entries, query, k=k)
    return [
        {
            "id": e["id"][:8],
            "full_id": e["id"],
            "date": (e.get("timestamp") or "")[:10],
            "emotion": e.get("emotion", "neutral"),
            "title": e.get("title") or e.get("summary") or "",
            "summary": (e.get("summary") or e.get("text", ""))[:240],
            "tags": e.get("tags", []),
        }
        for e in hits
    ]


def tool_get_entry(ref: str) -> Optional[dict]:
    """Fetch one entry by id-prefix or by date (YYYY-MM-DD)."""
    if not ref:
        return None
    entries = load_file(JOURNAL_FILE)
    ref = ref.strip()
    for e in entries:
        if e["id"].startswith(ref) or e["id"][:8] == ref:
            return _strip_embedding(e)
    for e in entries:
        if (e.get("timestamp") or "").startswith(ref):
            return _strip_embedding(e)
    return None


def tool_list_themes(period_days: int = 30) -> dict:
    entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=max(1, period_days))
    period = [e for e in entries
              if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    tags, emotions, themes = Counter(), Counter(), Counter()
    for e in period:
        for t in e.get("tags", []): tags[t] += 1
        emotions[e.get("emotion", "neutral")] += 1
        for th in e.get("themes", []): themes[th] += 1
    return {
        "period_days": period_days,
        "entry_count": len(period),
        "top_tags": tags.most_common(10),
        "top_emotions": emotions.most_common(),
        "top_themes": themes.most_common(8),
    }


def tool_period_summary(period: str = "week") -> dict:
    days = 7 if period == "week" else 30
    entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=days)
    period_entries = [e for e in entries
                      if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    total = len(period_entries)
    intensities = [e.get("intensity") for e in period_entries
                   if isinstance(e.get("intensity"), (int, float))]
    avg_intensity = round(sum(intensities) / len(intensities), 2) if intensities else None
    return {
        "period": period,
        "days": days,
        "entry_count": total,
        "avg_intensity": avg_intensity,
        "summaries": [
            {
                "id": e["id"][:8],
                "date": (e.get("timestamp") or "")[:10],
                "emotion": e.get("emotion", "?"),
                "intensity": e.get("intensity"),
                "summary": e.get("summary") or (e.get("text", "")[:140]),
            }
            for e in sorted(period_entries, key=lambda x: x.get("timestamp", ""))[-15:]
        ],
    }


def tool_emotion_extremes(emotion: str = "positive", k: int = 5) -> List[dict]:
    """Return the top-k entries scored by (emotion polarity × intensity).

    `emotion` can be:
      - "positive" / "best" / "happiest" — finds the most positive entries
      - "negative" / "worst" / "hardest" / "saddest" — finds the most negative
      - one of the specific emotion names (e.g. "anxious", "proud") —
        returns entries with that exact emotion, ranked by intensity
    """
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return []
    label = (emotion or "").strip().lower()
    positive_aliases = {"positive", "best", "happiest", "happy_day", "high", "up", "good"}
    negative_aliases = {"negative", "worst", "hardest", "saddest", "low", "down", "bad"}

    def _score(e):
        try:
            inten = float(e.get("intensity") or 5)
        except Exception:
            inten = 5.0
        pol = EMOTION_POLARITY.get((e.get("emotion") or "neutral").lower(), 0.0)
        return pol * inten

    if label in positive_aliases:
        candidates = [(_score(e), e) for e in entries]
        candidates = [c for c in candidates if c[0] > 0]
        candidates.sort(key=lambda x: -x[0])
    elif label in negative_aliases:
        candidates = [(_score(e), e) for e in entries]
        candidates = [c for c in candidates if c[0] < 0]
        candidates.sort(key=lambda x: x[0])  # most negative first
    else:
        # Specific emotion name
        match = [e for e in entries if (e.get("emotion") or "").lower() == label]
        match.sort(key=lambda e: -float(e.get("intensity") or 0))
        candidates = [(_score(e), e) for e in match]

    out = []
    for s, e in candidates[: max(1, min(k, 20))]:
        out.append({
            "id": e["id"][:8],
            "full_id": e["id"],
            "date": (e.get("timestamp") or "")[:10],
            "emotion": e.get("emotion", "neutral"),
            "intensity": e.get("intensity"),
            "title": e.get("title") or e.get("summary") or "",
            "summary": (e.get("summary") or e.get("text", ""))[:220],
            "tags": e.get("tags", []),
            "score": round(s, 2),
        })
    return out


TOOLS = {
    "search_entries": tool_search_entries,
    "get_entry": tool_get_entry,
    "list_themes": tool_list_themes,
    "period_summary": tool_period_summary,
    "emotion_extremes": tool_emotion_extremes,
}


def run_tool(name: str, args: dict) -> Any:
    fn = TOOLS.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    try:
        if name == "search_entries":
            return fn(str(args.get("query", "")), int(args.get("k", 5) or 5))
        if name == "get_entry":
            return fn(str(args.get("ref", "")))
        if name == "list_themes":
            return fn(int(args.get("period_days", 30) or 30))
        if name == "period_summary":
            p = str(args.get("period", "week")).lower()
            if p not in ("week", "month"):
                p = "week"
            return fn(p)
        if name == "emotion_extremes":
            return fn(str(args.get("emotion", "positive")), int(args.get("k", 5) or 5))
    except Exception as e:
        return {"error": str(e)}
    return None


# --- planner ---------------------------------------------------------------

def plan_tools(message: str) -> dict:
    """Planner: returns {thinking, tools, ask_back}.

    `tools` is a list of {name, args}. `ask_back` is a clarifying question to
    return to the user instead of calling tools (used when the message is
    too vague to act on).
    """
    schema_hint = (
        '{\n'
        '  "thinking": "one short sentence about what you need to find out",\n'
        '  "tools": [\n'
        '    {"name": "search_entries", "args": {"query": "...", "k": 5}}\n'
        '  ],\n'
        '  "ask_back": null\n'
        '}'
    )
    system = (
        "You are an agent planner for a personal journal app. "
        "Output strict JSON only — no prose, no markdown fences."
    )
    prompt = f"""User message: \"\"\"{message}\"\"\"

Available tools:
- search_entries(query: str, k: int=5) — semantic + keyword search across the journal
- get_entry(ref: str) — fetch one entry by 8-char id or by date (YYYY-MM-DD)
- list_themes(period_days: int=30) — top tags, emotions, and themes across the period (USE for "what themes / topics / patterns do I write about")
- period_summary(period: "week"|"month") — entry counts + avg intensity + recent summaries
- emotion_extremes(emotion: str, k: int=5) — entries scored by (emotion polarity × intensity). USE for "happiest / saddest / hardest / best / worst day" or "when was I most X". For positive: emotion="positive"; for negative: emotion="negative"; or a specific feeling like "anxious".

Routing hints (follow these unless the question is clearly different):
- "themes / topics / patterns / what do I write about" → list_themes (period_days=60 unless they specify) PLUS search_entries with the most relevant keyword from their question.
- "happiest / hardest / saddest / best / worst day" → emotion_extremes (emotion="positive" for happiest/best, "negative" for hardest/saddest/worst).
- "when did I feel X / when was I X" → emotion_extremes with emotion=X PLUS search_entries.
- "what changed / vs last month / compared to a year ago" → period_summary.
- Any reference to a feeling/person/topic/time without one of the above patterns → search_entries.

Rules:
- Pick 1–3 tools whenever the question is about the journal. Small talk ("hi", "thanks") gets 0 tools.
- Set ask_back ONLY when the message is genuinely impossible to act on (e.g. a bare "yes" with no prior context). NEVER use ask_back for plain questions like "what themes come up" or "what's my happiest day" — those have a clear answer in the data.
- Never invent ids or dates.

Respond with this JSON shape exactly (no prose, no fences):
{schema_hint}
"""
    raw = call_llama(prompt, system=system, temperature=0.2, timeout=60)
    obj = extract_json(raw) or {}
    tools_out = []
    for t in (obj.get("tools") or [])[:3]:
        if isinstance(t, dict) and t.get("name") in TOOLS:
            args = t.get("args") if isinstance(t.get("args"), dict) else {}
            tools_out.append({"name": t["name"], "args": args})
    ask_back = obj.get("ask_back")
    ask_back = str(ask_back).strip() if ask_back else None
    return {
        "thinking": str(obj.get("thinking", "")).strip()[:240],
        "tools": tools_out,
        "ask_back": ask_back or None,
    }


# --- formatting helpers ----------------------------------------------------

def _args_label(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, str):
            s = v if len(v) <= 28 else v[:28] + "…"
            parts.append(f'{k}="{s}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _summarize_tool_result(name: str, result: Any) -> str:
    if result is None:
        return "no result"
    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"
    if name == "search_entries" and isinstance(result, list):
        return f"{len(result)} matching entries"
    if name == "get_entry" and isinstance(result, dict):
        return f"{(result.get('timestamp') or '')[:10]} · {result.get('emotion', '?')}"
    if name == "list_themes" and isinstance(result, dict):
        tags = result.get("top_tags", [])[:3]
        tag_s = ", ".join(t[0] for t in tags) or "—"
        return f"{result.get('entry_count', 0)} entries · top tags: {tag_s}"
    if name == "period_summary" and isinstance(result, dict):
        return (f"{result.get('entry_count', 0)} entries · "
                f"avg intensity {result.get('avg_intensity')}")
    if name == "emotion_extremes" and isinstance(result, list):
        if not result:
            return "no matching entries"
        top = result[0]
        return f"{len(result)} entries · top: {top.get('date','?')} ({top.get('emotion','?')}/{top.get('intensity','?')})"
    return "ok"


def build_agent_prompt(message: str, observations: List[dict]) -> str:
    """Assemble the drafter prompt from the user's message + tool observations."""
    lines: List[str] = []
    for o in observations:
        name = o["tool"]; args = o.get("args", {}); res = o.get("result")
        lines.append(f"--- {name}({_args_label(args)}) ---")
        if name == "search_entries" and isinstance(res, list):
            for e in res:
                lines.append(
                    f"[id={e.get('id','')} | {e.get('date','')} | "
                    f"emotion={e.get('emotion','?')} | tags={','.join(e.get('tags', []))}]"
                )
                lines.append(_truncate(e.get("summary") or e.get("title", ""), 240))
        elif name == "emotion_extremes" and isinstance(res, list):
            for e in res:
                lines.append(
                    f"[id={e.get('id','')} | {e.get('date','')} | "
                    f"emotion={e.get('emotion','?')}/{e.get('intensity','?')} | "
                    f"score={e.get('score')} | tags={','.join(e.get('tags', []))}]"
                )
                lines.append(_truncate(e.get("summary") or e.get("title", ""), 240))
        elif name == "get_entry" and isinstance(res, dict):
            lines.append(
                f"[id={(res.get('id') or '')[:8]} | "
                f"{(res.get('timestamp') or '')[:10]} | "
                f"emotion={res.get('emotion','?')}]"
            )
            lines.append(_truncate(res.get("text", ""), 600))
        elif isinstance(res, (dict, list)):
            try:
                lines.append(_truncate(json.dumps(res), 600))
            except Exception:
                lines.append(str(res)[:600])
        lines.append("")
    obs_block = "\n".join(lines).strip() or "(no tool results — answer from what you already know about the user from context)"

    return (
        "=== What I just looked up in their journal ===\n"
        f"{obs_block}\n\n"
        "=== Their question ===\n"
        f"{message}\n\n"
        "Now reply. Rules:\n"
        "1. ANSWER the question. Plain questions are never ambiguous — never ask them to clarify what they meant.\n"
        "2. Speak like a person talking to a friend, not a chatbot. No 'Based on your entries...', no 'It seems', no 'I couldn't find any specific mentions'.\n"
        "3. When you reference what they wrote, cite inline with [cite:id] using the short id from the brackets above. Weave it into prose, don't list entries in parentheses.\n"
        "4. If the lookups are thin, lead with your best read in 1-2 sentences, then mention what's missing. Never refuse.\n"
        "5. Keep it tight — 2-5 sentences unless they asked for depth.\n\n"
        "Innerbloom:"
    )


# --- agent endpoint --------------------------------------------------------

def _stream_text(text: str, chunk_size: int = 14):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


@app.post("/chat/agent")
def chat_agent(data: ChatInput):
    """Agent loop with live step events. Always journal-mode behavior —
    general mode keeps using /chat/stream."""
    msg = (data.message or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    personality = _resolve_personality(data.personality)
    system = build_system_prompt("journal", personality)
    safety = crisis_check(msg)

    def event(obj: dict) -> str:
        return json.dumps(obj) + "\n"

    def gen():
        # 1) Safety up front so the UI can render the resources card immediately.
        if safety:
            yield event({"type": "safety", "safety": safety})

        # 2) Planning step
        yield event({"type": "step", "id": "plan", "kind": "plan",
                     "label": "Reading the question", "status": "running"})
        plan = plan_tools(msg)
        yield event({"type": "step", "id": "plan", "kind": "plan",
                     "label": "Reading the question", "status": "done",
                     "detail": plan.get("thinking") or "—"})

        # 2a) Ask-back path: stream the question back as the reply, no drafter.
        if plan.get("ask_back"):
            ab = plan["ask_back"]
            yield event({"type": "manifest", "candidates": [],
                         "mode": "journal", "personality": personality})
            for ch in _stream_text(ab):
                yield event({"type": "token", "value": ch})
            turn = {
                "id": str(uuid.uuid4()), "user": msg, "ai": ab,
                "citations": [], "mode": "journal", "personality": personality,
                "safety": safety, "ask_back": True,
                "agent_steps": {"thinking": plan.get("thinking", ""), "tools_used": []},
                "timestamp": datetime.now().isoformat(),
            }
            turn["embedding"] = embed(_chat_embed_text(turn))
            history = load_file(CHAT_FILE); history.append(turn); save_file(CHAT_FILE, history)
            yield event({"type": "done", "citations": []})
            return

        # 3) Run tools deterministically; emit running/done events for each.
        observations: List[dict] = []
        relevant_refs: List[dict] = []  # for the citation manifest
        for i, t in enumerate(plan["tools"]):
            sid = f"tool-{i}"
            label = f"{t['name']}({_args_label(t.get('args', {}))})"
            yield event({"type": "step", "id": sid, "kind": "tool",
                         "label": label, "status": "running"})
            result = run_tool(t["name"], t.get("args", {}))
            yield event({"type": "step", "id": sid, "kind": "tool",
                         "label": label, "status": "done",
                         "detail": _summarize_tool_result(t["name"], result)})
            observations.append({"tool": t["name"], "args": t.get("args", {}), "result": result})

            # Collect entry references for the citation manifest
            if t["name"] in ("search_entries", "emotion_extremes") and isinstance(result, list):
                relevant_refs.extend(result)
            elif t["name"] == "get_entry" and isinstance(result, dict):
                relevant_refs.append({
                    "full_id": result.get("id"),
                    "id": (result.get("id") or "")[:8],
                    "date": (result.get("timestamp") or "")[:10],
                    "emotion": result.get("emotion", "neutral"),
                    "title": result.get("title") or result.get("summary") or "",
                })

        # 4) Always include 1–2 most-recent entries in citation candidates.
        all_entries = load_file(JOURNAL_FILE)
        recent = sorted(all_entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:2]
        seen, candidates = set(), []
        for r in relevant_refs:
            sid = r.get("full_id") or r.get("id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            candidates.append({
                "id": (sid or "")[:8],
                "full_id": sid,
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "emotion": r.get("emotion", "neutral"),
            })
        for e in recent:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            candidates.append({
                "id": e["id"][:8], "full_id": e["id"],
                "title": e.get("title") or e.get("summary") or "",
                "date": (e.get("timestamp") or "")[:10],
                "emotion": e.get("emotion", "neutral"),
            })
        yield event({"type": "manifest", "candidates": candidates,
                     "mode": "journal", "personality": personality})

        # 5) Drafter — streams tokens
        yield event({"type": "step", "id": "draft", "kind": "draft",
                     "label": "Drafting the reply", "status": "running"})
        prompt = build_agent_prompt(msg, observations)
        collected: List[str] = []
        for chunk in call_llama_stream(prompt, system=system, temperature=0.6):
            collected.append(chunk)
            yield event({"type": "token", "value": chunk})
        yield event({"type": "step", "id": "draft", "kind": "draft",
                     "label": "Drafting the reply", "status": "done"})

        raw = "".join(collected)

        # Resolve citations against actual entries (need full objects)
        full_by_id = {e["id"]: e for e in all_entries}
        relevant_full = [full_by_id[c["full_id"]] for c in candidates
                         if c.get("full_id") in full_by_id]
        citations = extract_citations(raw, relevant_full)
        reply = clean_reply(raw)

        # Persist the turn
        turn = {
            "id": str(uuid.uuid4()), "user": msg, "ai": reply,
            "citations": citations, "mode": "journal", "personality": personality,
            "safety": safety,
            "agent_steps": {
                "thinking": plan.get("thinking", ""),
                "tools_used": [{"name": o["tool"], "args": o.get("args", {})} for o in observations],
            },
            "timestamp": datetime.now().isoformat(),
        }
        turn["embedding"] = embed(_chat_embed_text(turn))
        history = load_file(CHAT_FILE); history.append(turn); save_file(CHAT_FILE, history)

        yield event({"type": "done", "citations": citations})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat")
def chat(data: ChatInput):
    msg = (data.message or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    mode = data.mode if data.mode in ("journal", "general") else "journal"
    personality = _resolve_personality(data.personality)
    system = build_system_prompt(mode, personality)

    safety = crisis_check(msg)
    memory = _chat_memory_for_mode(msg, mode)
    relevant = memory["entries"]
    prompt = build_context(msg, memory)

    raw = call_llama(prompt, system=system, temperature=0.6)
    citations = extract_citations(raw, relevant)
    reply = clean_reply(raw)

    turn = {
        "id": str(uuid.uuid4()),
        "user": msg,
        "ai": reply,
        "citations": citations,
        "mode": mode,
        "personality": personality,
        "safety": safety,
        "timestamp": datetime.now().isoformat(),
    }
    turn["embedding"] = embed(_chat_embed_text(turn))

    history = load_file(CHAT_FILE)
    history.append(turn)
    save_file(CHAT_FILE, history)
    return {
        "reply": reply,
        "citations": citations,
        "safety": safety,
        "turn": _strip_embedding(turn),
    }


@app.post("/chat/stream")
def chat_stream(data: ChatInput):
    msg = (data.message or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    mode = data.mode if data.mode in ("journal", "general") else "journal"
    personality = _resolve_personality(data.personality)
    system = build_system_prompt(mode, personality)

    safety = crisis_check(msg)
    memory = _chat_memory_for_mode(msg, mode)
    relevant = memory["entries"]
    prompt = build_context(msg, memory)

    def gen():
        collected = []
        # Emit a metadata frame first (manifest + safety) so the UI can show
        # citation pills and a crisis card before the reply finishes streaming.
        manifest = {
            "type": "manifest",
            "candidates": [
                {"id": e["id"][:8], "full_id": e["id"],
                 "title": e.get("title") or e.get("summary") or "",
                 "date": e.get("timestamp", "")[:10],
                 "emotion": e.get("emotion", "neutral")}
                for e in relevant
            ],
            "safety": safety,
            "mode": mode,
            "personality": personality,
        }
        yield "\x1e" + json.dumps(manifest) + "\n"  # record separator
        for chunk in call_llama_stream(prompt, system=system, temperature=0.6):
            collected.append(chunk)
            yield chunk
        raw = "".join(collected)
        citations = extract_citations(raw, relevant)
        reply = clean_reply(raw)
        turn = {
            "id": str(uuid.uuid4()),
            "user": msg,
            "ai": reply,
            "citations": citations,
            "mode": mode,
            "personality": personality,
            "safety": safety,
            "timestamp": datetime.now().isoformat(),
        }
        turn["embedding"] = embed(_chat_embed_text(turn))
        history = load_file(CHAT_FILE)
        history.append(turn)
        save_file(CHAT_FILE, history)

    return StreamingResponse(gen(), media_type="text/plain")


@app.get("/chat")
def get_chat():
    return {"messages": [_strip_embedding(m) for m in load_file(CHAT_FILE)]}


@app.delete("/chat")
def clear_chat():
    save_file(CHAT_FILE, [])
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _entry_date(e: dict) -> Optional[date]:
    try:
        return datetime.fromisoformat(e["timestamp"]).date()
    except Exception:
        return None

def compute_streak(entries: List[dict]) -> Dict[str, int]:
    days = sorted({d for d in (_entry_date(e) for e in entries) if d}, reverse=True)
    if not days:
        return {"current": 0, "longest": 0}

    today = date.today()
    current = 0
    if days[0] == today or days[0] == today - timedelta(days=1):
        cursor = days[0]
        day_set = set(days)
        while cursor in day_set:
            current += 1
            cursor -= timedelta(days=1)

    longest = 1
    run = 1
    for i in range(1, len(days)):
        if days[i - 1] - days[i] == timedelta(days=1):
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return {"current": current, "longest": longest}


@app.get("/stats")
def stats():
    entries = load_file(JOURNAL_FILE)
    total_entries = len(entries)
    total_words = sum(e.get("word_count", 0) for e in entries)

    emotion_counts = Counter(e.get("emotion", "neutral") for e in entries)
    tag_counts = Counter()
    for e in entries:
        for t in e.get("tags", []):
            tag_counts[t] += 1

    today = date.today()
    by_day = defaultdict(int)
    mood_by_day = defaultdict(list)
    for e in entries:
        d = _entry_date(e)
        if not d: continue
        delta = (today - d).days
        if 0 <= delta <= 29:
            by_day[d.isoformat()] += 1
            intensity = e.get("intensity")
            if isinstance(intensity, (int, float)):
                mood_by_day[d.isoformat()].append(intensity)

    daily = []
    for i in range(29, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        intensities = mood_by_day.get(d, [])
        daily.append({
            "date": d,
            "count": by_day.get(d, 0),
            "avg_intensity": round(sum(intensities) / len(intensities), 2) if intensities else None,
        })

    # Year heatmap (up to 365 days)
    heatmap = []
    heat_day = defaultdict(int)
    for e in entries:
        d = _entry_date(e)
        if not d: continue
        if (today - d).days <= 365:
            heat_day[d.isoformat()] += 1
    for i in range(365, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        heatmap.append({"date": d, "count": heat_day.get(d, 0)})

    streak = compute_streak(entries)

    return {
        "total_entries": total_entries,
        "total_words": total_words,
        "avg_words": round(total_words / total_entries, 1) if total_entries else 0,
        "streak_current": streak["current"],
        "streak_longest": streak["longest"],
        "emotions": emotion_counts.most_common(),
        "top_tags": tag_counts.most_common(20),
        "daily": daily,
        "heatmap": heatmap,
    }


@app.get("/analyze")
def analyze_all():
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return {"result": "No entries yet. Write a few and I'll surface patterns."}

    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:40]
    combined = "\n\n".join(
        f"[{e.get('timestamp','')[:10]}] {e.get('summary') or e.get('text','')[:300]}"
        for e in recent
    )

    prompt = f"""Analyze these journal entries as a sharp, caring personal coach.
Write in second person. Be specific — name actual themes, not vague abstractions.
Avoid generic advice. Every sentence should feel like it could only be written for this person.

Structure your response with these four sections (use **bold** headers):
**Emotional arc** — how the mood and tone has shifted over this period (2 sentences)
**What keeps coming up** — 2-3 specific recurring themes or tensions (2-3 sentences)
**The thing you're not naming** — a blind spot, avoidance pattern, or unspoken need you notice (2 sentences)
**One honest suggestion** — one concrete, actionable thing worth trying this week (1-2 sentences)

Entries:
{combined}
"""
    return {"result": call_llama(prompt, temperature=0.6)}


@app.get("/weekly-review")
def weekly_review():
    entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=7)
    week = [e for e in entries
            if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    if not week:
        return {"result": "No entries in the last 7 days."}

    combined = "\n\n".join(
        f"[{e['timestamp'][:10]}] ({e.get('emotion','?')}) {e.get('text','')[:400]}"
        for e in week
    )
    prompt = f"""Write a sharp, personal weekly review from these 7-day journal entries.
Second person. Warm but direct — no fluff, no filler.

Use these sections (bold headers):
**The week in one sentence** — capture the emotional texture honestly
**What gave you energy** — a specific moment, interaction, or choice that worked
**What drained you** — a pattern, situation, or habit that cost you
**One thing to carry forward** — a micro-commitment for next week, concrete and small

Entries:
{combined}
"""
    return {"result": call_llama(prompt, temperature=0.6)}


@app.get("/monthly-review")
def monthly_review():
    entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=30)
    month = [e for e in entries
             if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    if not month:
        return {"result": "No entries in the last 30 days."}

    summaries = "\n".join(
        f"- [{e['timestamp'][:10]}] ({e.get('emotion','?')}/10={e.get('intensity','?')}) {e.get('summary') or e.get('text','')[:160]}"
        for e in month
    )

    prompt = f"""Write a rich monthly review from these 30-day journal summaries.
Second person. Be a thoughtful coach — specific, honest, forward-looking.

Sections (bold headers, 2-3 sentences each):
**The arc of this month** — what was the underlying current beneath the surface events?
**What you actually learned** — name 1-2 real insights, not just observations
**What kept costing you** — patterns, habits, or circumstances that drained you
**What quietly nourished you** — what gave you life, even unexpectedly
**Your one intention for next month** — one focused commitment, written as a personal promise

Journal summaries:
{summaries}
"""
    return {"result": call_llama(prompt, temperature=0.6)}


@app.get("/reflect/{entry_id}")
def reflect(entry_id: str):
    entries = load_file(JOURNAL_FILE)
    entry = next((e for e in entries if e["id"] == entry_id), None)
    if not entry:
        raise HTTPException(404, "Entry not found")
    prompt = f"""Read this journal entry and write exactly 3 thoughtful, open-ended
follow-up questions the author could reflect on. No numbering, no preamble.
Return them on separate lines.

Entry:
\"\"\"{entry['text']}\"\"\"
"""
    raw = call_llama(prompt, temperature=0.7)
    questions = [q.strip("-•* ").strip() for q in raw.splitlines() if q.strip()][:3]
    return {"questions": questions}


class ReflectAnswers(BaseModel):
    answers: List[dict]


@app.post("/reflect/{entry_id}/answers")
def save_reflect_answers(entry_id: str, body: ReflectAnswers):
    entries = load_file(JOURNAL_FILE)
    for e in entries:
        if e["id"] == entry_id:
            e["reflections"] = body.answers
            save_file(JOURNAL_FILE, entries)
            return {"status": "saved"}
    raise HTTPException(404, "Entry not found")


@app.get("/connections/{entry_id}")
def connections(entry_id: str, k: int = 4):
    entries = load_file(JOURNAL_FILE)
    entry = next((e for e in entries if e["id"] == entry_id), None)
    if not entry:
        raise HTTPException(404, "Entry not found")
    others = [e for e in entries if e["id"] != entry_id]
    seed = " ".join([entry.get("text", ""), " ".join(entry.get("tags", [])),
                     " ".join(entry.get("themes", []))])
    tokens = tokenize(seed)
    scored = [(score_entry(e, tokens), e) for e in others]
    scored.sort(key=lambda x: x[0], reverse=True)
    related = [
        {
            "id": e["id"],
            "title": e.get("title") or e.get("summary") or "",
            "date": e.get("timestamp", "")[:10],
            "emotion": e.get("emotion", "neutral"),
            "score": round(s, 2),
        }
        for s, e in scored if s > 0
    ][:k]
    return {"related": related}


@app.get("/prompt")
def get_prompt():
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return {"prompt": "What's on your mind right now, no filter?"}

    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:5]
    context = "\n".join(
        f"- ({e.get('emotion','?')}) {e.get('summary') or e.get('text','')[:120]}"
        for e in recent
    )

    prompt = f"""You write sharp, personal journal prompts. Based on this person's recent emotional themes,
write ONE journaling prompt — one sentence, under 22 words.
The prompt should invite honest self-examination, not generic reflection.
Connect to a specific theme or tension you notice. No preamble. No quotes. Just the prompt.

Recent mood and topics:
{context}
"""
    out = call_llama(prompt, temperature=0.92).strip().strip('"').strip()
    out = out.splitlines()[0] if out else "What are you tolerating right now that you haven't admitted to yourself?"
    return {"prompt": out}


# ---------------------------------------------------------------------------
# Anniversary / time-shifted surfacing
# ---------------------------------------------------------------------------
# Pull a window of entries from ±N days around a target date, compare against
# a matching recent window, and (optionally) ask the LLM to write a "then vs
# now" reflection. Used by the Write view's surfacing card.

def _window_around(entries: List[dict], target: datetime,
                   spread_days: int = 6) -> List[dict]:
    lo = target - timedelta(days=spread_days)
    hi = target + timedelta(days=spread_days)
    out = []
    for e in entries:
        dt = _safe_dt(e.get("timestamp"))
        if dt and lo <= dt <= hi:
            out.append(e)
    out.sort(key=lambda e: e.get("timestamp", ""))
    return out


def _summarize_window(entries: List[dict]) -> dict:
    """Compact stats for one window — top emotion / tags / themes, avg
    intensity, representative excerpts."""
    if not entries:
        return {"count": 0}
    emotions = Counter()
    tags = Counter()
    themes = Counter()
    intensities = []
    for e in entries:
        emotions[(e.get("emotion") or "neutral").lower()] += 1
        for t in e.get("tags", []) or []:
            tags[str(t).lower()] += 1
        for t in e.get("themes", []) or []:
            themes[str(t).lower()] += 1
        if isinstance(e.get("intensity"), (int, float)):
            intensities.append(float(e["intensity"]))
    avg_intensity = round(sum(intensities) / len(intensities), 2) if intensities else None
    excerpts = [{
        "id": e["id"], "short": e["id"][:8],
        "date": (e.get("timestamp") or "")[:10],
        "emotion": e.get("emotion", "neutral"),
        "intensity": e.get("intensity"),
        "title": e.get("title") or e.get("summary") or "",
        "summary": (e.get("summary") or e.get("text", ""))[:200],
    } for e in entries[-3:]]
    return {
        "count": len(entries),
        "avg_intensity": avg_intensity,
        "top_emotion": emotions.most_common(1)[0] if emotions else None,
        "emotions": emotions.most_common(5),
        "tags": tags.most_common(5),
        "themes": themes.most_common(5),
        "excerpts": excerpts,
    }


@app.get("/anniversary")
def anniversary(days: int = 30, spread: int = 6, with_reflection: bool = True):
    """Then-vs-now comparison surfacing.

    - `days`: how far back to look (default 30 = one month ago)
    - `spread`: half-width of the comparison window in days (default 6)
    - `with_reflection`: include the LLM-written 'then vs now' card

    Returns side-by-side stats for both windows plus an LLM reflection
    grounded in cited entry IDs.
    """
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return {"available": False, "reason": "no entries yet"}

    now = datetime.now()
    target = now - timedelta(days=max(1, days))

    then_window = _window_around(entries, target, spread)
    now_window  = _window_around(entries, now,    spread)

    if not then_window:
        return {
            "available": False,
            "reason": f"no entries within {spread} days of {target.date().isoformat()}",
            "days": days, "spread": spread,
        }

    then_summary = _summarize_window(then_window)
    now_summary  = _summarize_window(now_window)

    reflection_text = ""
    reflection_citations: List[dict] = []
    if with_reflection and len(then_window) + len(now_window) >= 3:
        by_id = {e["id"]: e for e in entries}

        def _lines(window: List[dict]) -> str:
            out = []
            for e in window[-8:]:
                snippet = (e.get("summary") or e.get("text", ""))[:200].replace("\n", " ").strip()
                out.append(
                    f"[id={e['id'][:8]} | {(e.get('timestamp') or '')[:10]} | "
                    f"emotion={e.get('emotion','?')}/{e.get('intensity',5)}] {snippet}"
                )
            return "\n".join(out) if out else "(nothing in window)"

        period_label = "a month ago" if days == 30 else (
            "a year ago" if days == 365 else f"{days} days ago"
        )

        schema = (
            '{\n'
            '  "same":      "1 sentence — what is unchanged. Cite [id8].",\n'
            '  "shifted":   "1 sentence — what is actually different. Cite [id8].",\n'
            '  "avoiding":  "1 sentence — what is still being avoided. Cite [id8].",\n'
            '  "becoming":  "1 short sentence — direction they are moving in"\n'
            '}'
        )
        system = (
            "You are an honest pattern observer for a personal journal. "
            "Write a short, grounded then-vs-now reflection. Every claim that "
            "names a fact must cite an 8-char entry id from the brackets. "
            "Don't moralise; don't motivate. Output strict JSON only."
        )
        prompt = f"""Compare the user's life {period_label} versus right now.

THEN window (around {target.date().isoformat()}):
{_lines(then_window)}

NOW window (around {now.date().isoformat()}):
{_lines(now_window)}

Rules:
- Each of "same", "shifted", "avoiding" should reference at least one [id8] from the windows above.
- Be specific. Quote their framing where you can.
- "becoming" should describe direction, not destination, no citation needed.

Schema (JSON exactly, no fences):
{schema}
"""
        raw = call_llama(prompt, system=system, temperature=0.4, timeout=180)
        obj = extract_json(raw) or {}
        reflection_text = {
            "same":     str(obj.get("same", "")).strip()[:400],
            "shifted":  str(obj.get("shifted", "")).strip()[:400],
            "avoiding": str(obj.get("avoiding", "")).strip()[:400],
            "becoming": str(obj.get("becoming", "")).strip()[:400],
        }
        # Resolve every [id8] referenced in the reflection back to a citation pill
        cited = set()
        for s in reflection_text.values():
            for m in re.findall(r"\[([a-z0-9]{4,8})\]", s):
                cited.add(m)
        reflection_citations = _enrich_evidence(list(cited), by_id)

    return {
        "available": True,
        "days": days,
        "spread": spread,
        "then": {"date_center": target.date().isoformat(), **then_summary},
        "now":  {"date_center": now.date().isoformat(), **now_summary},
        "reflection": reflection_text or None,
        "reflection_citations": reflection_citations,
    }



@app.get("/search")
def search(q: str):
    entries = load_file(JOURNAL_FILE)
    tokens = tokenize(q)
    if not tokens:
        return {"results": []}
    scored = [(score_entry(e, tokens), e) for e in entries]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [e for s, e in scored if s > 0][:25]
    return {"results": results}


@app.get("/export", response_class=PlainTextResponse)
def export_markdown():
    entries = load_file(JOURNAL_FILE)
    entries.sort(key=lambda e: e.get("timestamp", ""))
    lines = ["# Innerbloom — Journal Export\n"]
    for e in entries:
        d = e.get("timestamp", "")[:10]
        title = e.get("title", "") or e.get("summary", "")
        lines.append(f"## {d} — {title}\n")
        meta = []
        if e.get("emotion"): meta.append(f"**Emotion:** {e['emotion']}")
        if e.get("intensity") is not None: meta.append(f"**Intensity:** {e['intensity']}/10")
        if e.get("tags"): meta.append(f"**Tags:** {', '.join(e['tags'])}")
        if meta: lines.append(" · ".join(meta) + "\n")
        lines.append((e.get("text", "") + "\n").rstrip() + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# (Grow / goals endpoints were removed — no UI consumed them. If goal-tracking
# returns, both the data file and the routes should come back together.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spotify integration (OAuth PKCE + audio-feature mood correlation)
# ---------------------------------------------------------------------------

SPOTIFY_SCOPES = (
    "user-read-recently-played user-top-read user-read-playback-position "
    "user-library-read user-read-email"
)

def load_spotify():
    return load_file(SPOTIFY_FILE, default={})

def save_spotify(data):
    save_file(SPOTIFY_FILE, data)

def spotify_authed() -> Optional[dict]:
    cfg = load_spotify()
    if not cfg.get("access_token"):
        return None
    # Refresh if expired
    if cfg.get("expires_at", 0) < time.time() + 30:
        cfg = refresh_spotify_token(cfg)
        if not cfg:
            return None
    return cfg

def refresh_spotify_token(cfg: dict) -> Optional[dict]:
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": cfg.get("refresh_token"),
                "client_id": cfg.get("client_id"),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        tok = r.json()
        cfg["access_token"] = tok["access_token"]
        cfg["expires_at"] = time.time() + tok.get("expires_in", 3600)
        if tok.get("refresh_token"):
            cfg["refresh_token"] = tok["refresh_token"]
        save_spotify(cfg)
        return cfg
    except Exception:
        return None

def spotify_get(cfg: dict, path: str, params: dict = None) -> dict:
    r = requests.get(
        f"https://api.spotify.com/v1{path}",
        headers={"Authorization": f"Bearer {cfg['access_token']}"},
        params=params or {},
        timeout=20,
    )
    if r.status_code == 401:
        cfg = refresh_spotify_token(cfg)
        if not cfg:
            raise HTTPException(401, "Spotify auth expired")
        r = requests.get(
            f"https://api.spotify.com/v1{path}",
            headers={"Authorization": f"Bearer {cfg['access_token']}"},
            params=params or {}, timeout=20,
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:200])
    return r.json()


@app.get("/spotify/status")
def spotify_status():
    cfg = load_spotify()
    connected = bool(cfg.get("access_token"))
    info = {
        "connected": connected,
        "has_client_id": bool(cfg.get("client_id")),
        "client_id": cfg.get("client_id", ""),
        "redirect_uri": cfg.get("redirect_uri", ""),
        "scopes": SPOTIFY_SCOPES,
    }
    if connected:
        info["user"] = cfg.get("user")
    return info


@app.post("/spotify/config")
def spotify_config(body: SpotifyConfig):
    cfg = load_spotify()
    cfg["client_id"] = body.client_id.strip()
    cfg["redirect_uri"] = body.redirect_uri.strip()
    save_spotify(cfg)
    return {"status": "saved", "scopes": SPOTIFY_SCOPES}


@app.post("/spotify/exchange")
def spotify_exchange(body: SpotifyExchange):
    cfg = load_spotify()
    if not cfg.get("client_id"):
        raise HTTPException(400, "Set client_id first via /spotify/config")
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "authorization_code",
                "code": body.code,
                "redirect_uri": body.redirect_uri,
                "client_id": cfg["client_id"],
                "code_verifier": body.code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        if r.status_code != 200:
            raise HTTPException(400, f"Spotify token exchange failed: {r.text[:200]}")
        tok = r.json()
        cfg.update({
            "access_token": tok["access_token"],
            "refresh_token": tok.get("refresh_token"),
            "expires_at": time.time() + tok.get("expires_in", 3600),
            "scope": tok.get("scope"),
        })
        # Fetch user profile for display
        try:
            me = spotify_get(cfg, "/me")
            cfg["user"] = {
                "id": me.get("id"),
                "display_name": me.get("display_name"),
                "email": me.get("email"),
                "image": (me.get("images") or [{}])[0].get("url"),
            }
        except Exception:
            pass
        save_spotify(cfg)
        return {"status": "connected", "user": cfg.get("user")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Exchange error: {e}")


@app.post("/spotify/disconnect")
def spotify_disconnect():
    cfg = load_spotify()
    for k in ["access_token", "refresh_token", "expires_at", "user", "scope"]:
        cfg.pop(k, None)
    save_spotify(cfg)
    return {"status": "disconnected"}


@app.get("/spotify/recent")
def spotify_recent(limit: int = 50):
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")
    data = spotify_get(cfg, "/me/player/recently-played", {"limit": min(limit, 50)})
    items = data.get("items", [])
    tracks = [{
        "id": it["track"]["id"],
        "name": it["track"]["name"],
        "artist": ", ".join(a["name"] for a in it["track"]["artists"]),
        "album": it["track"]["album"]["name"],
        "image": (it["track"]["album"].get("images") or [{}])[0].get("url"),
        "played_at": it.get("played_at"),
        "duration_ms": it["track"].get("duration_ms"),
        "popularity": it["track"].get("popularity"),
    } for it in items if it.get("track")]
    return {"tracks": tracks}


@app.get("/spotify/top")
def spotify_top(kind: str = "tracks", time_range: str = "short_term", limit: int = 20):
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")
    if kind not in ("tracks", "artists"):
        raise HTTPException(400, "kind must be tracks or artists")
    data = spotify_get(cfg, f"/me/top/{kind}", {"time_range": time_range, "limit": min(limit, 50)})
    items = data.get("items", [])
    if kind == "tracks":
        out = [{
            "id": t["id"],
            "name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "image": (t["album"].get("images") or [{}])[0].get("url"),
            "popularity": t.get("popularity"),
        } for t in items]
    else:
        out = [{
            "id": a["id"],
            "name": a["name"],
            "genres": a.get("genres", []),
            "image": (a.get("images") or [{}])[0].get("url"),
            "popularity": a.get("popularity"),
        } for a in items]
    return {kind: out}


@app.get("/spotify/listening-pattern")
def spotify_listening_pattern():
    """Listening counts by hour-of-day and day-of-week + top genres from top artists."""
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")

    # Recently played for time-of-day pattern
    recent = spotify_get(cfg, "/me/player/recently-played", {"limit": 50}).get("items", [])
    by_hour = [0] * 24
    by_dow = [0] * 7  # Mon=0
    for item in recent:
        ts = item.get("played_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            by_hour[dt.hour] += 1
            by_dow[dt.weekday()] += 1
        except Exception:
            continue

    # Top genres from top artists
    artists = spotify_get(cfg, "/me/top/artists", {"time_range": "medium_term", "limit": 50}).get("items", [])
    genre_counts = Counter()
    for a in artists:
        for g in a.get("genres", []):
            genre_counts[g] += 1

    # Total listening minutes (from recently played) — rough since each track played once
    total_minutes = round(sum((it.get("track", {}).get("duration_ms", 0) or 0) for it in recent) / 60000)

    return {
        "by_hour": by_hour,
        "by_dow": by_dow,
        "top_genres": genre_counts.most_common(8),
        "total_minutes_recent": total_minutes,
        "tracks_analysed": len(recent),
    }


# ---------------------------------------------------------------------------
# Genre → mood mapping.
#
# Spotify deprecated /audio-features for new apps in Nov 2024, so we can't ask
# them for valence/energy any more. But every artist still ships genre tags,
# and genres carry pretty consistent mood signatures. Each entry is
# (valence, energy) on [0,1]² — valence = positivity (sad→happy),
# energy = intensity (mellow→intense). Tuned by hand from public listener
# studies; not gospel, but defensible.
#
# Match strategy: substring match against the genre string. "lo-fi indie pop"
# would match the "lo-fi" and "indie" and "pop" entries; we average the hits.
# ---------------------------------------------------------------------------

GENRE_MOOD = {
    # high valence, high energy
    "pop":            (0.78, 0.72),
    "dance":          (0.80, 0.85),
    "edm":            (0.72, 0.88),
    "house":          (0.74, 0.82),
    "disco":          (0.85, 0.80),
    "funk":           (0.82, 0.75),
    "afrobeat":       (0.82, 0.78),
    "afrobeats":      (0.82, 0.78),
    "reggaeton":      (0.78, 0.80),
    "latin":          (0.78, 0.70),
    "k-pop":          (0.78, 0.78),
    "j-pop":          (0.72, 0.70),
    "punjabi":        (0.78, 0.74),
    "bollywood":      (0.72, 0.68),
    "filmi":          (0.70, 0.66),

    # high valence, moderate energy
    "indie pop":      (0.65, 0.55),
    "synth-pop":      (0.62, 0.62),
    "soft rock":      (0.55, 0.45),
    "soul":           (0.58, 0.55),
    "r&b":            (0.55, 0.55),
    "neo soul":       (0.55, 0.50),
    "jazz":           (0.50, 0.40),
    "bossa nova":     (0.65, 0.40),

    # moderate
    "rock":           (0.55, 0.72),
    "alternative":    (0.45, 0.65),
    "alt rock":       (0.45, 0.65),
    "alternative rock":(0.45, 0.65),
    "garage":         (0.50, 0.78),
    "blues":          (0.40, 0.55),
    "country":        (0.55, 0.55),
    "folk":           (0.50, 0.40),

    # low valence, high energy
    "metal":          (0.35, 0.90),
    "punk":           (0.40, 0.85),
    "hardcore":       (0.30, 0.92),
    "trap":           (0.45, 0.78),
    "drill":          (0.30, 0.82),
    "hip hop":        (0.50, 0.70),
    "hip-hop":        (0.50, 0.70),
    "rap":            (0.50, 0.72),
    "grunge":         (0.30, 0.72),

    # low valence, low energy
    "sad":            (0.20, 0.30),
    "lo-fi":          (0.45, 0.30),
    "lofi":           (0.45, 0.30),
    "chillhop":       (0.50, 0.35),
    "indie folk":     (0.45, 0.35),
    "singer-songwriter":(0.40, 0.40),
    "shoegaze":       (0.35, 0.55),
    "dream pop":      (0.50, 0.40),
    "ambient":        (0.50, 0.20),
    "classical":      (0.55, 0.35),
    "piano":          (0.55, 0.30),
    "neoclassical":   (0.50, 0.30),
    "post-rock":      (0.40, 0.60),
    "emo":            (0.30, 0.65),

    # textural / catchall
    "acoustic":       (0.55, 0.35),
    "chill":          (0.60, 0.35),
    "indie":          (0.55, 0.55),
    "instrumental":   (0.55, 0.35),
    "electronic":     (0.55, 0.65),
}


def _mood_from_genres(genres: List[str]) -> Optional[tuple]:
    """Average the (valence, energy) hits across a track's genres. Returns
    None if no genres match the table — caller decides what to do."""
    if not genres:
        return None
    hits = []
    for g in genres:
        gl = (g or "").lower()
        for key, vec in GENRE_MOOD.items():
            if key in gl:
                hits.append(vec)
    if not hits:
        return None
    v = sum(h[0] for h in hits) / len(hits)
    e = sum(h[1] for h in hits) / len(hits)
    return round(v, 3), round(e, 3)


def _mood_quadrant_label(valence: float, energy: float) -> str:
    """Plain-English label for any (v,e) point. Russell circumplex
    territory — happy/energetic top-right, sad/mellow bottom-left."""
    v_hi = valence >= 0.55
    e_hi = energy >= 0.55
    if v_hi and e_hi:    return "Happy & energetic"
    if v_hi and not e_hi: return "Calm & content"
    if not v_hi and e_hi: return "Tense & intense"
    return "Mellow & melancholic"


@app.get("/spotify/genre-mood")
def spotify_genre_mood():
    """
    Replacement for the deprecated /audio-features path.

    Maps each recently-played track to a (valence, energy) point via its
    artist genres, aggregates per day, returns:
      - a 30-day time series so the music chart still works
      - a per-track list so we can plot a 2-D mood scatter
      - per-day quadrant counts so we can show "what mood you were in"
      - a single weighted-average point so the user gets a one-glance read

    Returns 200 with `enriched_share` so the UI can warn if our genre table
    matched too few tracks (e.g. when artists have no genre tags at all).
    """
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")

    # 1) Pull recent plays (50 max from Spotify)
    recent = spotify_get(cfg, "/me/player/recently-played", {"limit": 50}).get("items", [])
    if not recent:
        return {"tracks": [], "points": [], "quadrants": {}, "enriched_share": 0.0}

    # 2) Collect unique artist IDs to fetch genres in bulk
    artist_ids: List[str] = []
    seen_artists = set()
    plays: List[dict] = []
    for it in recent:
        tr = it.get("track")
        if not tr:
            continue
        primary_artist = (tr.get("artists") or [{}])[0]
        aid = primary_artist.get("id")
        if aid and aid not in seen_artists:
            seen_artists.add(aid)
            artist_ids.append(aid)
        plays.append({
            "track_id": tr.get("id"),
            "name": tr.get("name") or "",
            "artist_id": aid,
            "artist": ", ".join(a["name"] for a in tr.get("artists", []) if a.get("name")),
            "image": (tr.get("album", {}).get("images") or [{}])[0].get("url"),
            "played_at": it.get("played_at"),
            "duration_ms": tr.get("duration_ms"),
            "popularity": tr.get("popularity"),
        })

    # 3) Bulk-fetch genres (Spotify lets us pull up to 50 artists at once)
    artist_genres: Dict[str, List[str]] = {}
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i + 50]
        try:
            data = spotify_get(cfg, "/artists", {"ids": ",".join(batch)})
        except HTTPException:
            continue
        for a in data.get("artists", []) or []:
            if a and a.get("id"):
                artist_genres[a["id"]] = a.get("genres") or []

    # 4) Score each track via its primary artist's genres
    tracks_out: List[dict] = []
    enriched = 0
    for p in plays:
        genres = artist_genres.get(p["artist_id"], [])
        mood = _mood_from_genres(genres)
        if mood:
            v, e = mood
            enriched += 1
        else:
            v = e = None
        tracks_out.append({
            "track_id": p["track_id"],
            "name": p["name"],
            "artist": p["artist"],
            "image": p["image"],
            "played_at": p["played_at"],
            "genres": genres[:4],
            "valence": v,
            "energy": e,
            "quadrant": _mood_quadrant_label(v, e) if (v is not None and e is not None) else None,
        })

    enriched_share = round(enriched / len(plays), 3) if plays else 0.0

    # 5) Per-day rollup (30-day window, oldest-first to match other charts)
    journal = load_file(JOURNAL_FILE)
    j_by_day: Dict[str, List[float]] = defaultdict(list)
    for e in journal:
        ts = _safe_dt(e.get("timestamp"))
        if not ts:
            continue
        if e.get("intensity") is not None:
            j_by_day[ts.date().isoformat()].append(e["intensity"])

    day_buckets: Dict[str, List[tuple]] = defaultdict(list)
    for t in tracks_out:
        if t["valence"] is None or not t["played_at"]:
            continue
        d = t["played_at"][:10]
        day_buckets[d].append((t["valence"], t["energy"]))

    points: List[dict] = []
    today = date.today()
    for i in range(29, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        bucket = day_buckets.get(d, [])
        if bucket:
            avg_v = round(sum(b[0] for b in bucket) / len(bucket), 3)
            avg_e = round(sum(b[1] for b in bucket) / len(bucket), 3)
        else:
            avg_v = avg_e = None
        j = j_by_day.get(d, [])
        points.append({
            "date": d,
            "valence": avg_v,
            "energy": avg_e,
            "plays": len(bucket),
            "journal_intensity": round(sum(j) / len(j), 2) if j else None,
        })

    # 6) Quadrant histogram across the 50 plays
    quadrants = Counter()
    for t in tracks_out:
        if t["quadrant"]:
            quadrants[t["quadrant"]] += 1
    quad_total = sum(quadrants.values()) or 1

    # 7) Single centroid for "your music mood right now"
    val_list = [t["valence"] for t in tracks_out if t["valence"] is not None]
    en_list = [t["energy"] for t in tracks_out if t["energy"] is not None]
    centroid = None
    if val_list:
        cv = sum(val_list) / len(val_list)
        ce = sum(en_list) / len(en_list)
        centroid = {
            "valence": round(cv, 3),
            "energy": round(ce, 3),
            "label": _mood_quadrant_label(cv, ce),
        }

    # 8) Discovered genres — even when none map to mood, the user wants to
    # see what Spotify thinks they're listening to. Counter across all
    # primary artists.
    genre_counter = Counter()
    for ag in artist_genres.values():
        for g in ag:
            genre_counter[g] += 1
    discovered_genres = [
        {"name": g, "count": c} for g, c in genre_counter.most_common(12)
    ]

    # 9) Top artists by play count (always works, no genres needed)
    artist_counter = Counter()
    artist_meta: Dict[str, dict] = {}
    for p in plays:
        a = (p.get("artist") or "").split(",")[0].strip()
        if a:
            artist_counter[a] += 1
            if a not in artist_meta:
                artist_meta[a] = {"name": a, "image": p.get("image")}
    top_artists = [
        {**artist_meta[a], "plays": c}
        for a, c in artist_counter.most_common(8)
    ]

    # 10) Listening window summary — first/last play timestamps
    timestamps = sorted(p.get("played_at") for p in plays if p.get("played_at"))
    listening_window = None
    if timestamps:
        listening_window = {
            "first": timestamps[0],
            "last": timestamps[-1],
            "active_days": len(set((t or "")[:10] for t in timestamps)),
        }

    return {
        "tracks": tracks_out,
        "points": points,
        "centroid": centroid,
        "quadrants": [
            {"label": q, "count": c, "share": round(c / quad_total, 3)}
            for q, c in quadrants.most_common()
        ],
        "enriched_share": enriched_share,
        "play_count": len(plays),
        "discovered_genres": discovered_genres,
        "top_artists": top_artists,
        "listening_window": listening_window,
        "genre_table_size": len(GENRE_MOOD),
    }


@app.get("/spotify/daily")
def spotify_daily(days: int = 30):
    """Per-day listening counts + per-day journal mood for an honest
    side-by-side chart. We don't smooth, we don't interpolate, and we don't
    invent correlations — empty days stay empty."""
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")

    n_days = max(7, min(days, 90))

    # Recent plays — Spotify caps at 50
    recent = spotify_get(cfg, "/me/player/recently-played", {"limit": 50}).get("items", [])
    plays_by_day: Dict[str, int] = defaultdict(int)
    for it in recent:
        ts = (it.get("played_at") or "")[:10]
        if ts:
            plays_by_day[ts] += 1

    # Journal per day — top emotion + avg intensity
    entries = load_file(JOURNAL_FILE)
    by_day_emotion: Dict[str, Counter] = defaultdict(Counter)
    by_day_intensity: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        ts = _safe_dt(e.get("timestamp"))
        if not ts:
            continue
        d = ts.date().isoformat()
        by_day_emotion[d][(e.get("emotion") or "neutral").lower()] += 1
        try:
            by_day_intensity[d].append(float(e.get("intensity") or 0))
        except Exception:
            pass

    today = date.today()
    days_out: List[dict] = []
    for i in range(n_days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        plays = plays_by_day.get(d, 0)
        emo_counter = by_day_emotion.get(d)
        emotion = emo_counter.most_common(1)[0][0] if emo_counter else None
        intensities = by_day_intensity.get(d, [])
        avg_intensity = round(sum(intensities) / len(intensities), 2) if intensities else None
        days_out.append({
            "date": d,
            "plays": plays,
            "emotion": emotion,
            "avg_intensity": avg_intensity,
            "entry_count": len(intensities),
        })

    # Honest overlap metric: how many days have BOTH a play and an entry
    overlap = sum(1 for d in days_out if d["plays"] > 0 and d["avg_intensity"] is not None)
    music_days = sum(1 for d in days_out if d["plays"] > 0)
    journal_days = sum(1 for d in days_out if d["avg_intensity"] is not None)

    return {
        "days": days_out,
        "window_days": n_days,
        "plays_total": sum(d["plays"] for d in days_out),
        "music_days": music_days,
        "journal_days": journal_days,
        "overlap_days": overlap,
    }


def fetch_audio_features(cfg, track_ids: List[str]) -> tuple:
    """Return (features, status) where status is 'ok', 'forbidden' (Spotify
    deprecated /audio-features for new apps in Nov 2024), or 'error'."""
    feats = []
    status = "ok"
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        try:
            data = spotify_get(cfg, "/audio-features", {"ids": ",".join(batch)})
            for f in data.get("audio_features", []):
                if f:
                    feats.append(f)
        except HTTPException as he:
            # 401/403/404 typically means the new-app deprecation. Stop trying
            # — the rest of the batches will fail identically.
            if he.status_code in (401, 403, 404):
                status = "forbidden"
                break
            status = "error"
            continue
    if not feats and status == "ok":
        status = "empty"
    return feats, status


def _proxy_features_from_tracks(played: List[dict]) -> Dict[str, dict]:
    """Fallback when /audio-features is unavailable. We can't compute real
    valence/energy without it, but we *can* surface a meaningful 'listening
    signal' from raw metadata: how heavily someone played each day, when
    they listened, and how mainstream the music skews. Returns a per-track
    dict so the aggregator can stay shape-compatible."""
    out = {}
    for p in played:
        pop = p.get("popularity") or 50
        # Map popularity 0..100 → an "obscurity" score (higher = more niche).
        obscurity = round(1.0 - pop / 100.0, 3)
        out[p["id"]] = {"id": p["id"], "popularity": pop, "obscurity": obscurity}
    return out


@app.get("/spotify/mood")
def spotify_mood():
    """
    Aggregate features for recently-played tracks and correlate them with
    journal mood. When Spotify's /audio-features endpoint is unavailable
    (deprecated for new apps Nov 2024), we degrade to a 'listening
    intensity' signal derived from play volume, time-of-day, and popularity
    — the chart stays meaningful instead of going blank.
    """
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")

    # 1. Recent tracks (last 50 plays)
    data = spotify_get(cfg, "/me/player/recently-played", {"limit": 50})
    items = data.get("items", [])
    if not items:
        return {"points": [], "summary": "No recent plays found.",
                "degraded": False, "available_metrics": []}

    played = []
    ids = []
    for it in items:
        tr = it.get("track")
        if not tr or not tr.get("id"):
            continue
        played.append({
            "id": tr["id"],
            "name": tr["name"],
            "artist": ", ".join(a["name"] for a in tr["artists"]),
            "played_at": it.get("played_at"),
            "popularity": tr.get("popularity"),
            "duration_ms": tr.get("duration_ms"),
        })
        ids.append(tr["id"])

    feats, status = fetch_audio_features(cfg, ids)
    degraded = status != "ok"
    feat_by_id = {f["id"]: f for f in feats}
    proxy_by_id = _proxy_features_from_tracks(played) if degraded else {}

    available_metrics = (
        ["valence", "energy", "danceability", "tempo"]
        if not degraded else ["plays_per_day", "obscurity", "evening_share"]
    )

    # 2. Aggregate per day
    by_day = defaultdict(list)
    proxy_by_day = defaultdict(list)
    plays_by_day = Counter()
    hour_by_day: Dict[str, List[int]] = defaultdict(list)
    for p in played:
        d = (p.get("played_at") or "")[:10]
        if not d:
            continue
        plays_by_day[d] += 1
        # Track listening hour for the day so we can build an "evening share".
        try:
            hr = datetime.fromisoformat((p["played_at"] or "").replace("Z", "+00:00")).hour
            hour_by_day[d].append(hr)
        except Exception:
            pass
        f = feat_by_id.get(p["id"])
        if f:
            by_day[d].append(f)
        elif degraded:
            pr = proxy_by_id.get(p["id"])
            if pr:
                proxy_by_day[d].append(pr)

    # 3. Journal intensity per day (last 30 days)
    journal = load_file(JOURNAL_FILE)
    j_by_day = defaultdict(list)
    for e in journal:
        ts = _safe_dt(e.get("timestamp"))
        if not ts:
            continue
        d = ts.date().isoformat()
        if e.get("intensity") is not None:
            j_by_day[d].append(e["intensity"])

    points = []
    today = date.today()
    for i in range(29, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        day_feats = by_day.get(d, [])
        if day_feats:
            valence = round(sum(f["valence"] for f in day_feats) / len(day_feats), 3)
            energy = round(sum(f["energy"] for f in day_feats) / len(day_feats), 3)
            danceability = round(sum(f["danceability"] for f in day_feats) / len(day_feats), 3)
            tempo = round(sum(f["tempo"] for f in day_feats) / len(day_feats), 1)
        else:
            valence = energy = danceability = tempo = None

        # Proxy signal for the degraded mode — meaningful even without features.
        plays = plays_by_day.get(d, 0)
        proxies = proxy_by_day.get(d, [])
        obscurity = round(sum(p["obscurity"] for p in proxies) / len(proxies), 3) if proxies else None
        hrs = hour_by_day.get(d, [])
        evening_share = round(sum(1 for h in hrs if h >= 18 or h < 5) / len(hrs), 3) if hrs else None

        j_intensities = j_by_day.get(d, [])
        journal_intensity = round(sum(j_intensities) / len(j_intensities), 2) if j_intensities else None
        points.append({
            "date": d,
            "valence": valence,
            "energy": energy,
            "danceability": danceability,
            "tempo": tempo,
            "plays": plays,
            "obscurity": obscurity,
            "evening_share": evening_share,
            "journal_intensity": journal_intensity,
            "play_count": len(day_feats) if day_feats else plays,
        })

    # 4. Overall averages — both real and proxy lines, whichever exist.
    all_val = [p["valence"] for p in points if p["valence"] is not None]
    all_en = [p["energy"] for p in points if p["energy"] is not None]
    all_obs = [p["obscurity"] for p in points if p["obscurity"] is not None]
    avg_valence = round(sum(all_val) / len(all_val), 3) if all_val else None
    avg_energy = round(sum(all_en) / len(all_en), 3) if all_en else None
    avg_obscurity = round(sum(all_obs) / len(all_obs), 3) if all_obs else None

    return {
        "points": points,
        "avg_valence": avg_valence,
        "avg_energy": avg_energy,
        "avg_obscurity": avg_obscurity,
        "play_count": len(played),
        "recent": played[:10],
        "degraded": degraded,
        "audio_features_status": status,
        "available_metrics": available_metrics,
    }


@app.get("/spotify/insight")
def spotify_insight():
    """LLM interpretation of the Spotify ↔ journal correlation."""
    data = spotify_mood()
    points = data.get("points", [])
    val = data.get("avg_valence")
    en = data.get("avg_energy")
    degraded = data.get("degraded")

    if degraded:
        lines = []
        for p in points[-14:]:
            if not (p.get("plays") or p.get("journal_intensity")):
                continue
            lines.append(
                f"- {p['date']}: plays={p.get('plays')}, "
                f"obscurity={p.get('obscurity')}, "
                f"evening_share={p.get('evening_share')}, "
                f"journal_intensity={p.get('journal_intensity')}"
            )
        if not lines:
            return {"insight": "Not enough overlapping data yet. Listen a little and journal a little more.",
                    "degraded": True}
        prompt = f"""You are reading a 14-day window of a person's Spotify listening and journal mood.
Spotify's audio-feature data isn't available for this app, so the signals are:
- plays: how many tracks they listened to that day (volume)
- obscurity: 0 = top-40 mainstream, 1 = obscure; high values often track introspective listening
- evening_share: fraction of plays after 6pm or before 5am (late-night/evening tilt)
- journal_intensity: self/AI rated emotional intensity (1-10)

Write a short, warm reflection (4-6 sentences) for the user in second person:
- Note volume changes and what they might mean (heavy days = avoidance? momentum?)
- Whether obscurity rises on heavier journal days
- Whether evening listening lines up with harder days
- One tiny kind observation
No medical advice. No pathologizing.

Data:
{chr(10).join(lines)}
"""
        return {"insight": call_llama(prompt, temperature=0.5), "degraded": True}

    # Full audio-features path
    lines = []
    for p in points[-14:]:
        if p["valence"] is None and p["journal_intensity"] is None:
            continue
        lines.append(
            f"- {p['date']}: valence={p['valence']}, energy={p['energy']}, "
            f"plays={p['play_count']}, journal_intensity={p['journal_intensity']}"
        )
    if not lines:
        return {"insight": "Not enough overlapping data yet. Listen a little and journal a little more.",
                "degraded": False}

    prompt = f"""You are analyzing a 14-day window of listening + journal data.
Valence is Spotify's measure of musical positivity (0=sad, 1=happy).
Journal intensity is the writer's self/AI-rated emotional intensity (1-10).

Write a short, warm reflection (4-6 sentences) for the user in second person:
- What does the music mood pattern suggest
- Note any correlation (or lack) with journal intensity — cite specific days
- One tiny, kind observation about what songs may be doing for them
Do not pathologize. No medical advice.

Data:
{chr(10).join(lines)}

Averages: valence={val}, energy={en}.
"""
    return {"insight": call_llama(prompt, temperature=0.5), "degraded": False}


# ---------------------------------------------------------------------------
# Phase 3 — Insight Engines
#
# These are pattern-detection passes over the journal that produce structured
# findings. Findings are cached in data/insights.json so repeated reads are
# cheap; the user explicitly triggers a recompute.
#
# Engines (built incrementally):
#   - contradictions: stated intentions vs actual behavior
#   - (next) triggers: what tends to precede mood dips
#   - (next) burnout/spiral: rolling time-series rules + LLM verdict
#   - (next) narrative: identity & life-story summary
# ---------------------------------------------------------------------------

def _load_insights() -> dict:
    return load_file(INSIGHTS_FILE, default={})


def _save_insights(data: dict):
    save_file(INSIGHTS_FILE, data)


def _entries_for_window(days: int) -> List[dict]:
    """Return journal entries from the last N days, sorted oldest-first."""
    entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=max(7, days))
    window = [e for e in entries
              if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    window.sort(key=lambda e: e.get("timestamp", ""))
    if len(window) < 6:
        # Not enough recent — fall back to the most recent 25 regardless of date
        window = sorted(entries, key=lambda e: e.get("timestamp", ""))[-25:]
    return window


def _enrich_evidence(entry_ids: List[str], lookup: Dict[str, dict]) -> List[dict]:
    """Resolve a list of short or full entry IDs into citation-pill payloads."""
    out = []
    seen = set()
    for raw in entry_ids or []:
        if not isinstance(raw, str):
            continue
        rid = raw.strip()
        if not rid or rid in seen:
            continue
        # Match either full ID or 8-char prefix
        match = lookup.get(rid)
        if not match:
            for full_id, e in lookup.items():
                if full_id.startswith(rid) or full_id[:8] == rid:
                    match = e
                    break
        if not match:
            continue
        seen.add(match["id"])
        out.append({
            "id": match["id"],
            "title": match.get("title") or match.get("summary") or "",
            "date": (match.get("timestamp") or "")[:10],
            "emotion": match.get("emotion", "neutral"),
        })
    return out


def compute_contradictions(window_days: int = 60) -> dict:
    """LLM pass to surface gaps between stated intentions/values and actual
    behavior. Caches into INSIGHTS_FILE under the 'contradictions' key."""
    window = _entries_for_window(window_days)
    if len(window) < 4:
        return {"items": [], "generated_at": datetime.now().isoformat(),
                "window_days": window_days, "entry_count": len(window),
                "note": "Need a few more entries before contradictions are meaningful."}

    # Compact entry list for the LLM — short id, date, emotion, brief text
    lines = []
    for e in window:
        snippet = (e.get("summary") or e.get("text", ""))[:240].replace("\n", " ").strip()
        lines.append(
            f"[id={e['id'][:8]} | {(e.get('timestamp') or '')[:10]} | "
            f"emotion={e.get('emotion','?')}] {snippet}"
        )
    journal_block = "\n".join(lines)

    schema = (
        '{\n'
        '  "items": [\n'
        '    {\n'
        '      "stated": "what the user said they want / value / will do",\n'
        '      "behavior": "what they actually did that contradicts it",\n'
        '      "evidence": ["id8", "id8"],\n'
        '      "pattern": "1-3 word label e.g. avoidance, intention-drift, perfectionism",\n'
        '      "severity": "low | medium | high",\n'
        '      "honest_note": "one sentence to the user, second person, no fluff"\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    system = (
        "You are an honest pattern-spotter for a personal journal. "
        "Find genuine contradictions between what the user states (intentions, values, plans) "
        "and what they actually do (behavior in entries). Be specific and direct. "
        "Don't moralize, don't motivate — just name the gap. "
        "Output strict JSON only — no prose, no markdown fences."
    )

    prompt = f"""Below are recent journal entries (oldest first).

Find 2–6 contradictions between stated intentions/values and observed behavior.
Rules:
- Each contradiction must cite at least one entry id for "stated" and one for "behavior" — combined in the evidence list.
- Use ONLY the 8-char ids shown in brackets above. Never invent ids.
- If you can't find real contradictions, return an empty items array.
- Quote the user's own framing in "stated" and "behavior" — paraphrase tightly, don't editorialize.
- "honest_note" should sound like a friend who isn't trying to make the user feel better — it's the value-add.

Schema (JSON exactly, no fences, no prose):
{schema}

Entries:
{journal_block}
"""
    raw = call_llama(prompt, system=system, temperature=0.35, timeout=240)
    obj = extract_json(raw) or {}

    # Build entry lookup once
    entries_full = load_file(JOURNAL_FILE)
    by_id = {e["id"]: e for e in entries_full}

    items: List[dict] = []
    for it in (obj.get("items") or [])[:8]:
        if not isinstance(it, dict):
            continue
        stated = str(it.get("stated", "")).strip()
        behavior = str(it.get("behavior", "")).strip()
        if not stated or not behavior:
            continue
        evidence = _enrich_evidence(it.get("evidence") or [], by_id)
        # Drop any contradiction with no real evidence — that's hallucination
        if not evidence:
            continue
        severity = str(it.get("severity", "medium")).lower()
        if severity not in ("low", "medium", "high"):
            severity = "medium"
        items.append({
            "stated": stated[:400],
            "behavior": behavior[:400],
            "evidence": evidence[:6],
            "pattern": str(it.get("pattern", "")).strip()[:40] or "pattern",
            "severity": severity,
            "honest_note": str(it.get("honest_note", "")).strip()[:400],
        })

    result = {
        "items": items,
        "generated_at": datetime.now().isoformat(),
        "window_days": window_days,
        "entry_count": len(window),
    }
    cache = _load_insights()
    cache["contradictions"] = result
    _save_insights(cache)
    return result


@app.get("/insights/contradictions")
def get_contradictions():
    """Return the cached contradiction set. Empty until /refresh is called."""
    cache = _load_insights()
    data = cache.get("contradictions") or {
        "items": [], "generated_at": None, "window_days": 60, "entry_count": 0,
    }
    return data


@app.post("/insights/contradictions/refresh")
def refresh_contradictions(window_days: int = 60):
    """Recompute contradictions over the last N days and persist."""
    return compute_contradictions(window_days=window_days)


# --- Emotional Trigger Map ------------------------------------------------
#
# Hybrid engine. We use stats to identify tags/themes that correlate with
# mood swings (cheap, no LLM, anchored in real data), then ask the LLM to
# translate the loaded labels into human-readable trigger phrases with
# evidence. The stats step prevents the LLM from inventing triggers.

EMOTION_POLARITY = {
    # positive
    "happy": 1.0, "grateful": 1.0, "excited": 1.0, "hopeful": 0.9,
    "proud": 0.9, "content": 0.8, "calm": 0.4, "reflective": 0.1,
    # neutral
    "neutral": 0.0,
    # negative
    "anxious": -1.0, "sad": -1.0, "frustrated": -0.85, "angry": -1.0,
    "lonely": -1.0, "overwhelmed": -1.0, "tired": -0.55,
}


def _mood_score(entry: dict) -> float:
    """Signed mood score in roughly [-10, +10]: negative for low/heavy days,
    positive for energized/grateful days. Combines emotion polarity × the
    user-or-AI-assigned intensity."""
    pol = EMOTION_POLARITY.get((entry.get("emotion") or "neutral").lower(), 0.0)
    try:
        intensity = float(entry.get("intensity") or 5)
    except Exception:
        intensity = 5.0
    return pol * intensity


def _entry_labels(entry: dict):
    """Yield deduped, normalized tag/theme labels for one entry."""
    seen = set()
    for raw in list(entry.get("tags", [])) + list(entry.get("themes", [])):
        t = (str(raw) or "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            yield t


def _candidate_triggers(entries: List[dict], min_count: int = 3,
                        min_abs_delta: float = 1.0) -> List[dict]:
    """Stats pass: find labels whose average mood-score deviates from the
    user's baseline by at least `min_abs_delta`. Returns sorted by |delta|."""
    if not entries:
        return []
    scores = [_mood_score(e) for e in entries]
    baseline = sum(scores) / len(scores)

    by_label_scores: Dict[str, List[float]] = defaultdict(list)
    by_label_entries: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        ms = _mood_score(e)
        for label in _entry_labels(e):
            by_label_scores[label].append(ms)
            by_label_entries[label].append(e)

    out = []
    for label, slist in by_label_scores.items():
        if len(slist) < min_count:
            continue
        avg = sum(slist) / len(slist)
        delta = avg - baseline
        if abs(delta) < min_abs_delta:
            continue
        ents = sorted(by_label_entries[label],
                      key=lambda e: e.get("timestamp", ""), reverse=True)
        out.append({
            "label": label,
            "count": len(slist),
            "avg_mood": round(avg, 2),
            "baseline": round(baseline, 2),
            "delta": round(delta, 2),
            "direction": "negative" if delta < 0 else "positive",
            "entry_ids": [e["id"] for e in ents[:6]],
        })

    out.sort(key=lambda c: abs(c["delta"]), reverse=True)
    return out[:12]


TRIGGER_CATEGORIES = {"social", "work", "health", "sleep", "family",
                      "money", "self", "creative", "other"}


def compute_triggers(window_days: int = 90) -> dict:
    """Find emotional triggers in the last N days. Stats first, then LLM
    characterization with strict evidence requirements."""
    entries = _entries_for_window(window_days)
    if len(entries) < 6:
        return {"items": [], "generated_at": datetime.now().isoformat(),
                "window_days": window_days, "entry_count": len(entries),
                "note": "Need a few more entries before triggers are meaningful."}

    candidates = _candidate_triggers(entries)
    if not candidates:
        return {"items": [], "generated_at": datetime.now().isoformat(),
                "window_days": window_days, "entry_count": len(entries),
                "note": "Your moods are pretty even across topics — no strong triggers stand out yet."}

    by_id = {e["id"]: e for e in load_file(JOURNAL_FILE)}

    # Build a compact candidate block for the LLM with sample entries
    cand_lines = []
    for c in candidates[:8]:
        cand_lines.append(
            f"\n[{c['label']}] direction={c['direction']} "
            f"count={c['count']} avg_mood={c['avg_mood']} (baseline {c['baseline']})"
        )
        for eid in c["entry_ids"][:3]:
            e = by_id.get(eid)
            if not e: continue
            snippet = (e.get("summary") or e.get("text", ""))[:160]
            snippet = snippet.replace("\n", " ").strip()
            cand_lines.append(
                f"  - id={e['id'][:8]} | {(e.get('timestamp') or '')[:10]} | "
                f"emotion={e.get('emotion','?')}/{e.get('intensity',5)} : {snippet}"
            )

    schema = (
        '{\n'
        '  "items": [\n'
        '    {\n'
        '      "label": "human-readable trigger phrase, 2-7 words",\n'
        '      "outcome": "what tends to follow (e.g. anxiety spike, mood lift, energy drain)",\n'
        '      "category": "social | work | health | sleep | family | money | self | creative | other",\n'
        '      "direction": "negative | positive",\n'
        '      "evidence": ["id8"],\n'
        '      "pattern_note": "one honest sentence about the actual pattern"\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    system = (
        "You are an honest pattern-spotter for a personal journal. "
        "Identify what triggers mood shifts — what tends to precede dips or lifts. "
        "Be specific, not motivational. Output strict JSON only — no prose, no fences."
    )

    prompt = f"""Below are candidate trigger labels from the user's journal — pulled by stats, not by you.
Each candidate shows its direction, frequency, and average mood-score versus the user's baseline.

For each meaningful candidate:
- Translate the raw label into a clear trigger phrase (e.g. tag "mom" might become "Calls with mom").
- Pair it with the outcome (the mood shift the data shows).
- Cite the 8-char id of at least one supporting entry from the samples.
- Skip any candidate that doesn't really hold up when you read the entries.
- Return at most 6 items.

Candidates:
{chr(10).join(cand_lines)}

Schema (JSON exactly, no fences):
{schema}
"""
    raw = call_llama(prompt, system=system, temperature=0.4, timeout=240)
    obj = extract_json(raw) or {}

    items: List[dict] = []
    for it in (obj.get("items") or [])[:8]:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label", "")).strip()
        if not label:
            continue
        evidence = _enrich_evidence(it.get("evidence") or [], by_id)
        if not evidence:
            continue  # anti-hallucination: must cite a real entry

        direction = str(it.get("direction", "")).lower()
        if direction not in ("negative", "positive"):
            direction = "negative" if "neg" in direction else "positive"

        category = str(it.get("category", "other")).lower().strip()
        if category not in TRIGGER_CATEGORIES:
            category = "other"

        # Try to attach the matching candidate's stats so the UI can show
        # frequency / avg mood / delta numerically.
        ev_ids = {ev["id"] for ev in evidence}
        match = None
        for c in candidates:
            if any(eid in ev_ids for eid in c["entry_ids"]):
                match = c
                break

        items.append({
            "label": label[:80],
            "outcome": str(it.get("outcome", "")).strip()[:120],
            "category": category,
            "direction": direction,
            "evidence": evidence[:5],
            "pattern_note": str(it.get("pattern_note", "")).strip()[:300],
            "stats": {
                "frequency": match["count"] if match else None,
                "avg_mood": match["avg_mood"] if match else None,
                "delta": match["delta"] if match else None,
            } if match else None,
        })

    result = {
        "items": items,
        "generated_at": datetime.now().isoformat(),
        "window_days": window_days,
        "entry_count": len(entries),
        "baseline_mood": round(sum(_mood_score(e) for e in entries) / len(entries), 2),
    }
    cache = _load_insights()
    cache["triggers"] = result
    _save_insights(cache)
    return result


@app.get("/insights/triggers")
def get_triggers():
    cache = _load_insights()
    return cache.get("triggers") or {
        "items": [], "generated_at": None, "window_days": 90, "entry_count": 0,
    }


@app.post("/insights/triggers/refresh")
def refresh_triggers(window_days: int = 90):
    return compute_triggers(window_days=window_days)


# --- Wellbeing radar: burnout + negative-spiral ---------------------------
#
# Stats first (deterministic, anchored in real data), then a single LLM pass
# to write honest 1-2 sentence assessments for both signals.
#
# We grade severity into four levels: ok | watch | elevated | high. The
# crisis-safety layer in the chat path handles acute self-harm signals — this
# layer is for chronic patterns the user wouldn't notice in any single entry.

WELLBEING_LEVELS = ("ok", "watch", "elevated", "high")
NEG_EMOTIONS = {"anxious", "sad", "frustrated", "angry", "lonely",
                "overwhelmed", "tired"}
EXHAUSTION_EMOTIONS = {"tired", "overwhelmed"}


def _level_burnout(avg_mood: float, exhaustion_ratio: float, neg_ratio: float) -> str:
    if avg_mood <= -3.0 or (exhaustion_ratio >= 0.4 and avg_mood <= -1.0):
        return "high"
    if avg_mood <= -1.5 or exhaustion_ratio >= 0.3 or neg_ratio >= 0.6:
        return "elevated"
    if avg_mood <= -0.5 or exhaustion_ratio >= 0.2:
        return "watch"
    return "ok"


def _burnout_stats(entries: List[dict]) -> dict:
    if not entries:
        return {"level": "ok", "entry_count": 0}

    scores = [_mood_score(e) for e in entries]
    avg_mood = sum(scores) / len(scores)
    exhaustion = sum(1 for e in entries
                     if (e.get("emotion") or "").lower() in EXHAUSTION_EMOTIONS)
    neg = sum(1 for e in entries
              if (e.get("emotion") or "").lower() in NEG_EMOTIONS)

    # Trend: compare first vs last third of the window
    n = len(entries)
    third = max(1, n // 3)
    early = sum(_mood_score(e) for e in entries[:third]) / third
    late = sum(_mood_score(e) for e in entries[-third:]) / third
    drift = round(late - early, 2)

    level = _level_burnout(avg_mood, exhaustion / n, neg / n)

    return {
        "level": level,
        "entry_count": n,
        "avg_mood": round(avg_mood, 2),
        "exhaustion_count": exhaustion,
        "negative_count": neg,
        "drift": drift,  # negative drift = getting heavier
        "evidence_ids": [e["id"] for e in entries[-6:]],
    }


def _spiral_stats(entries: List[dict], lookback_days: int = 7) -> dict:
    """Look for a recent run of declining days. Multiple entries in one day
    are averaged into that day's score."""
    if not entries:
        return {"level": "ok", "entry_count": 0}

    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent = [e for e in entries
              if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    if len(recent) < 2:
        return {"level": "ok", "entry_count": len(recent),
                "consecutive_decline": 0, "evidence_ids": []}

    by_day: Dict[str, List[float]] = defaultdict(list)
    for e in recent:
        d = (e.get("timestamp") or "")[:10]
        by_day[d].append(_mood_score(e))

    days_sorted = sorted(by_day.keys())
    day_scores = [(d, sum(by_day[d]) / len(by_day[d])) for d in days_sorted]

    # Longest run of strictly-declining trailing days
    decline = 0
    for i in range(len(day_scores) - 1, 0, -1):
        if day_scores[i][1] < day_scores[i - 1][1]:
            decline += 1
        else:
            break
    if decline > 0:
        decline += 1  # include the starting day of the run

    # Are the trailing days net-negative too?
    tail_avg = sum(s for _, s in day_scores[-decline:]) / decline if decline else 0.0
    all_negative_tail = all(s < 0 for _, s in day_scores[-decline:]) if decline else False

    if decline >= 4 and all_negative_tail and tail_avg <= -3.0:
        level = "high"
    elif decline >= 3 and tail_avg < 0:
        level = "elevated"
    elif decline >= 3 or (decline == 2 and tail_avg < -1.0):
        level = "watch"
    else:
        level = "ok"

    # Pick the entries that fall on the declining tail as evidence
    tail_dates = {d for d, _ in day_scores[-decline:]} if decline else set()
    evidence = [e for e in recent
                if (e.get("timestamp") or "")[:10] in tail_dates]
    evidence.sort(key=lambda e: e.get("timestamp", ""))

    return {
        "level": level,
        "entry_count": len(recent),
        "consecutive_decline": decline,
        "tail_avg_mood": round(tail_avg, 2),
        "all_negative_tail": all_negative_tail,
        "evidence_ids": [e["id"] for e in evidence[-6:]],
    }


def _wellbeing_llm_pass(burnout: dict, spiral: dict, entries: List[dict]) -> dict:
    """One LLM call to write honest 1-2 sentence assessments for both
    signals. Skipped entirely if both signals are 'ok' — saves the call."""
    if burnout.get("level") == "ok" and spiral.get("level") == "ok":
        return {
            "burnout_summary": "No signs of burnout in the last two weeks.",
            "spiral_summary": "No spiral pattern in your recent days.",
        }

    by_id = {e["id"]: e for e in entries}

    def _ev_lines(ids):
        out = []
        for eid in ids[-5:]:
            e = by_id.get(eid)
            if not e: continue
            snippet = (e.get("summary") or e.get("text", ""))[:160].replace("\n", " ").strip()
            out.append(
                f"  - id={e['id'][:8]} | {(e.get('timestamp') or '')[:10]} | "
                f"emotion={e.get('emotion','?')}/{e.get('intensity',5)} : {snippet}"
            )
        return "\n".join(out) if out else "  (no entries)"

    schema = (
        '{\n'
        '  "burnout_summary": "1–2 honest sentences. Cite [id8] inline.",\n'
        '  "spiral_summary":  "1–2 honest sentences. Cite [id8] inline."\n'
        '}'
    )

    system = (
        "You are an honest, calm observer for a personal journal. "
        "Two signals are flagged below: burnout (chronic, two-week pattern) and "
        "spiral (acute, last-few-days pattern). For each, write 1–2 sentences "
        "the user would respect: specific, grounded, no motivational fluff. "
        "Do NOT diagnose. Do NOT give medical advice. Output strict JSON only."
    )

    prompt = f"""Burnout signal: level={burnout.get('level')} · avg_mood={burnout.get('avg_mood')} · exhaustion_count={burnout.get('exhaustion_count')}/{burnout.get('entry_count')} · drift={burnout.get('drift')}
Recent entries (most recent last):
{_ev_lines(burnout.get('evidence_ids', []))}

Spiral signal: level={spiral.get('level')} · consecutive_decline={spiral.get('consecutive_decline')} days · tail_avg_mood={spiral.get('tail_avg_mood')}
Trailing entries:
{_ev_lines(spiral.get('evidence_ids', []))}

Write the JSON exactly:
{schema}
"""
    raw = call_llama(prompt, system=system, temperature=0.4, timeout=120)
    obj = extract_json(raw) or {}
    return {
        "burnout_summary": str(obj.get("burnout_summary", "")).strip()[:400] or
                           "Some signals of burnout — see the entries below.",
        "spiral_summary": str(obj.get("spiral_summary", "")).strip()[:400] or
                          "A short downward trend — worth a moment of attention.",
    }


def compute_wellbeing(window_days: int = 14) -> dict:
    burnout_entries = _entries_for_window(window_days)
    burnout = _burnout_stats(burnout_entries)

    # Spiral always uses last 7 days regardless of window selection
    all_entries = load_file(JOURNAL_FILE)
    cutoff = datetime.now() - timedelta(days=8)
    recent_for_spiral = [e for e in all_entries
                         if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    recent_for_spiral.sort(key=lambda e: e.get("timestamp", ""))
    spiral = _spiral_stats(recent_for_spiral)

    # If we don't have enough data, return early without an LLM call
    if burnout.get("entry_count", 0) < 4 and spiral.get("entry_count", 0) < 4:
        return {
            "burnout": {**burnout, "level": "ok",
                        "summary": "Not enough recent entries to read a burnout pattern."},
            "spiral": {**spiral, "level": "ok",
                       "summary": "Not enough recent entries to spot a spiral."},
            "generated_at": datetime.now().isoformat(),
            "window_days": window_days,
        }

    by_id = {e["id"]: e for e in all_entries}
    llm = _wellbeing_llm_pass(burnout, spiral, burnout_entries)

    burnout["summary"] = llm.get("burnout_summary", "")
    burnout["evidence"] = _enrich_evidence(burnout.get("evidence_ids", []), by_id)
    burnout.pop("evidence_ids", None)

    spiral["summary"] = llm.get("spiral_summary", "")
    spiral["evidence"] = _enrich_evidence(spiral.get("evidence_ids", []), by_id)
    spiral.pop("evidence_ids", None)

    result = {
        "burnout": burnout,
        "spiral": spiral,
        "generated_at": datetime.now().isoformat(),
        "window_days": window_days,
    }
    cache = _load_insights()
    cache["wellbeing"] = result
    _save_insights(cache)
    return result


@app.get("/insights/wellbeing")
def get_wellbeing():
    cache = _load_insights()
    return cache.get("wellbeing") or {
        "burnout": {"level": "ok", "summary": "", "evidence": []},
        "spiral":  {"level": "ok", "summary": "", "evidence": []},
        "generated_at": None, "window_days": 14,
    }


@app.post("/insights/wellbeing/refresh")
def refresh_wellbeing(window_days: int = 14):
    return compute_wellbeing(window_days=window_days)


# --- Identity & Narrative Layer -------------------------------------------
#
# A self-narrative pass: who the user has been, the arcs they're inside,
# values shown by action, tensions between selves, and where they're heading.
# All claims must cite real entries — anti-hallucination guard rejects items
# whose evidence ids don't resolve.

def compute_narrative(window_days: int = 120) -> dict:
    entries = _entries_for_window(window_days)
    if len(entries) < 6:
        return {"identity_lines": [], "current_arcs": [], "values_in_action": [],
                "tensions": [], "becoming": "",
                "generated_at": datetime.now().isoformat(),
                "window_days": window_days, "entry_count": len(entries),
                "note": "Need a few more entries before a real narrative emerges."}

    # Compact entry lines
    lines = []
    for e in entries:
        snippet = (e.get("summary") or e.get("text", ""))[:220].replace("\n", " ").strip()
        lines.append(
            f"[id={e['id'][:8]} | {(e.get('timestamp') or '')[:10]} | "
            f"emotion={e.get('emotion','?')}/{e.get('intensity',5)}] {snippet}"
        )
    journal_block = "\n".join(lines)

    schema = (
        '{\n'
        '  "identity_lines": ["3-5 first-person I-am lines that capture how the user actually shows up"],\n'
        '  "current_arcs": [\n'
        '    {"label": "2-4 word arc name", "description": "what is in motion right now", "evidence": ["id8"]}\n'
        '  ],\n'
        '  "values_in_action": [\n'
        '    {"value": "single word like honesty / discipline / care", "evidence_note": "how it shows up", "evidence": ["id8"]}\n'
        '  ],\n'
        '  "tensions": [\n'
        '    {"a": "one pull (in their words)", "b": "the opposing pull", "evidence": ["id8"]}\n'
        '  ],\n'
        '  "becoming": "1-2 honest sentences about the direction they are moving in"\n'
        '}'
    )

    system = (
        "You are an honest narrative observer for a personal journal. "
        "Build a self-narrative from the user's entries — who they have been, "
        "what arcs they are inside, what values their actions reveal, what "
        "tensions are in play. Quote their framing where you can; never invent. "
        "Output strict JSON only — no prose, no markdown fences."
    )

    prompt = f"""Below are the user's recent journal entries (oldest first).

Build their narrative. Rules:
- "identity_lines" should sound like the user — first person, specific, no generic affirmations.
- Every arc / value / tension MUST cite at least one 8-char entry id from the brackets above.
- If you cannot ground a claim in entries, drop it.
- "becoming" should describe direction, not destination.
- 3–5 identity lines, 2–4 arcs, 2–5 values, 1–4 tensions max.

Schema (JSON exactly, no fences):
{schema}

Entries:
{journal_block}
"""
    raw = call_llama(prompt, system=system, temperature=0.45, timeout=300)
    obj = extract_json(raw) or {}

    by_id = {e["id"]: e for e in load_file(JOURNAL_FILE)}

    def _enriched_items(arr, fields):
        out = []
        for it in (arr or [])[:6]:
            if not isinstance(it, dict):
                continue
            ev = _enrich_evidence(it.get("evidence") or [], by_id)
            if not ev:
                continue  # require real evidence
            cleaned = {f: str(it.get(f, "")).strip()[:300] for f in fields}
            cleaned["evidence"] = ev[:4]
            out.append(cleaned)
        return out

    identity_lines = []
    for line in (obj.get("identity_lines") or [])[:6]:
        s = str(line).strip()
        if s:
            identity_lines.append(s[:200])

    current_arcs = _enriched_items(obj.get("current_arcs"), ["label", "description"])
    values_in_action = _enriched_items(obj.get("values_in_action"), ["value", "evidence_note"])
    tensions = _enriched_items(obj.get("tensions"), ["a", "b"])
    becoming = str(obj.get("becoming", "")).strip()[:400]

    result = {
        "identity_lines": identity_lines,
        "current_arcs": current_arcs,
        "values_in_action": values_in_action,
        "tensions": tensions,
        "becoming": becoming,
        "generated_at": datetime.now().isoformat(),
        "window_days": window_days,
        "entry_count": len(entries),
    }
    cache = _load_insights()
    cache["narrative"] = result
    _save_insights(cache)
    return result


@app.get("/insights/narrative")
def get_narrative():
    cache = _load_insights()
    return cache.get("narrative") or {
        "identity_lines": [], "current_arcs": [], "values_in_action": [],
        "tensions": [], "becoming": "", "generated_at": None,
    }


@app.post("/insights/narrative/refresh")
def refresh_narrative(window_days: int = 120):
    return compute_narrative(window_days=window_days)


# --- Memory Graph ---------------------------------------------------------
#
# Build a node-edge graph from journal entries. Nodes are entries; edges
# connect entries that share themes/tags/emotions, weighted by Jaccard
# similarity. The frontend lays this out with a force simulation.
#
# Embedding-based edges are also possible but Jaccard on tags/themes is
# fast, deterministic, and produces visibly meaningful clusters.

@app.get("/graph")
def get_graph(limit: int = 250, min_weight: float = 0.22):
    """Return a memory graph of recent entries with named clusters.

    Nodes are entries; edges connect entries that share tags / themes /
    emotion, scored by an **IDF-weighted** Jaccard so common-everywhere
    tags (e.g. 'sleep', 'work') don't dominate similarity. We then run
    connected components, split any mega-cluster (>30% of nodes) by its
    most discriminating sub-feature, and name each cluster after a term
    that *distinguishes* it rather than a term that's globally common.
    """
    entries = load_file(JOURNAL_FILE)
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    entries = entries[:max(10, min(limit, 500))]

    # ---- Feature extraction ---------------------------------------------
    feats: Dict[str, set] = {}
    raw_tags: Dict[str, set] = {}
    raw_themes: Dict[str, set] = {}
    for e in entries:
        s, tg, th = set(), set(), set()
        for t in e.get("tags", []) or []:
            v = (str(t) or "").strip().lower()
            if v:
                s.add(v); tg.add(v)
        for t in e.get("themes", []) or []:
            v = (str(t) or "").strip().lower()
            if v:
                s.add(v); th.add(v)
        emo = (e.get("emotion") or "").strip().lower()
        if emo: s.add(f"emo:{emo}")
        feats[e["id"]] = s
        raw_tags[e["id"]] = tg
        raw_themes[e["id"]] = th

    nodes = [
        {
            "id": e["id"],
            "short": e["id"][:8],
            "title": (e.get("title") or e.get("summary") or "")[:80],
            "date": (e.get("timestamp") or "")[:10],
            "emotion": e.get("emotion", "neutral"),
            "intensity": e.get("intensity") or 5,
            "tags": e.get("tags", []),
            "themes": e.get("themes", []),
        }
        for e in entries
    ]
    ids = list(feats.keys())
    N = len(ids)

    # ---- IDF weights for every feature ----------------------------------
    # Common tags (sleep across 40% of entries) get a low weight; rare
    # themes (specific phrases) get a high weight. Keeps the giant "shared
    # by everyone" tags from creating one mega-cluster.
    feat_df = Counter()
    for s in feats.values():
        for f in s:
            feat_df[f] += 1
    feat_idf: Dict[str, float] = {}
    for f, df in feat_df.items():
        # +1 smoothing; cap to a sensible range
        feat_idf[f] = math.log((N + 1) / (df + 1)) + 1.0

    def _weighted_jaccard(a: set, b: set) -> float:
        if not a or not b: return 0.0
        inter = a & b
        union = a | b
        if not inter: return 0.0
        wi = sum(feat_idf.get(f, 1.0) for f in inter)
        wu = sum(feat_idf.get(f, 1.0) for f in union)
        return wi / wu if wu else 0.0

    # ---- Edges -----------------------------------------------------------
    edges = []
    for i in range(N):
        a = feats[ids[i]]
        if not a: continue
        for j in range(i + 1, N):
            b = feats[ids[j]]
            if not b: continue
            shared = a & b
            if not shared: continue
            w = _weighted_jaccard(a, b)
            if w < min_weight: continue
            edges.append({
                "source": ids[i],
                "target": ids[j],
                "weight": round(w, 3),
                "shared": sorted(shared, key=lambda f: -feat_idf.get(f, 1.0))[:5],
            })

    # ---- Connected components (union-find) -----------------------------
    parent: Dict[str, str] = {nid: nid for nid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for ed in edges:
        union(ed["source"], ed["target"])

    component_of: Dict[str, str] = {nid: find(nid) for nid in ids}
    by_component: Dict[str, List[str]] = defaultdict(list)
    for nid, comp in component_of.items():
        by_component[comp].append(nid)

    # Pull out singletons into a synthetic "solo" bucket
    singletons = [members[0] for members in by_component.values() if len(members) == 1]
    for nid in singletons:
        if nid in by_component:
            del by_component[nid]
    if singletons:
        by_component["__solo__"] = list(singletons)

    # ---- Mega-cluster splitting -----------------------------------------
    # If any one component holds more than MEGA_FRACTION of the (non-solo)
    # nodes, partition it by its most-discriminating tag/theme into
    # sub-clusters. We never split below MIN_SUBCLUSTER_SIZE per piece.
    MEGA_FRACTION = 0.30
    MIN_SUBCLUSTER_SIZE = 3
    non_solo_total = sum(len(v) for k, v in by_component.items() if k != "__solo__")

    def _split_mega(members: List[str]) -> List[List[str]]:
        """Greedy split: find the highest-IDF tag/theme that covers a
        meaningful subset of members, partition by 'has it' / 'doesn't',
        recurse if the larger side is still mega. Returns list of buckets."""
        if len(members) < MIN_SUBCLUSTER_SIZE * 2:
            return [members]
        # Candidate features inside this cluster — by IDF × prevalence
        feat_count = Counter()
        for nid in members:
            for f in feats[nid]:
                if not f.startswith("emo:"):  # emotion is a weak signal here
                    feat_count[f] += 1
        if not feat_count:
            return [members]
        # Score = idf × min(count, M-count): favours splits that cleave roughly in half
        best_f, best_score = None, 0
        for f, c in feat_count.items():
            other = len(members) - c
            if c < MIN_SUBCLUSTER_SIZE or other < MIN_SUBCLUSTER_SIZE:
                continue
            balance = min(c, other) / max(c, other)  # 1.0 = perfectly balanced
            score = feat_idf.get(f, 1.0) * balance * (1.0 + math.log(c))
            if score > best_score:
                best_score = score; best_f = f
        if not best_f:
            return [members]
        with_f, without_f = [], []
        for nid in members:
            (with_f if best_f in feats[nid] else without_f).append(nid)
        # Recurse if the bigger side is still mega-sized
        out = []
        for bucket in (with_f, without_f):
            if len(bucket) > non_solo_total * MEGA_FRACTION and len(bucket) >= MIN_SUBCLUSTER_SIZE * 2:
                out.extend(_split_mega(bucket))
            elif bucket:
                out.append(bucket)
        return out

    # Apply mega-split
    refined: Dict[str, List[str]] = {}
    split_counter = 0
    for cid, members in by_component.items():
        if cid == "__solo__":
            refined[cid] = members
            continue
        if non_solo_total > 0 and len(members) > non_solo_total * MEGA_FRACTION and len(members) >= MIN_SUBCLUSTER_SIZE * 2:
            buckets = _split_mega(members)
            for b in buckets:
                if not b: continue
                key = f"split_{split_counter}_{b[0][:6]}"
                split_counter += 1
                refined[key] = b
        else:
            refined[cid] = members
    by_component = refined

    # ---- Discriminating-feature cluster naming -------------------------
    # A cluster's name should be a feature that's *common inside* but
    # *not in everything else*. We compute per-cluster TF-IDF over the
    # tags/themes and pick the top non-emotion candidate.
    SKIP_TAGS = {"work", "life", "day", "today", "thing", "stuff", "time", "things"}

    # Total presence of each feature across the WHOLE graph (for IDF inside naming)
    GLOBAL_PREVALENCE = {f: df / max(N, 1) for f, df in feat_df.items()}

    def _label_for(members: List[str]) -> Dict[str, Any]:
        if len(members) == 1 and members[0] in singletons:
            return {"name": "Unconnected", "kind": "solo",
                    "size": len(members), "top_terms": [], "top_emotion": None}

        themes_in = Counter()
        tags_in = Counter()
        emotions = Counter()
        for nid in members:
            for t in raw_themes.get(nid, set()):
                themes_in[t] += 1
            for t in raw_tags.get(nid, set()):
                tags_in[t] += 1
        for n in nodes:
            if n["id"] in members:
                emotions[n["emotion"]] += 1

        size = len(members)
        # TF-IDF score per candidate: (count_in_cluster / size) × IDF
        def _score(term: str, count: int) -> float:
            tf = count / size
            global_share = GLOBAL_PREVALENCE.get(term, 0.0001)
            # IDF-like signal; heavily punish terms that appear in >60% of the graph
            if global_share > 0.6:
                return 0.0
            return tf * feat_idf.get(term, 1.0) * (1.0 + math.log(count))

        coverage = max(2, size // 4)
        theme_candidates = [(t, c, _score(t, c)) for t, c in themes_in.items() if c >= coverage]
        theme_candidates.sort(key=lambda x: -x[2])

        tag_candidates = [(t, c, _score(t, c)) for t, c in tags_in.items()
                          if c >= coverage and t not in SKIP_TAGS]
        tag_candidates.sort(key=lambda x: -x[2])

        # Themes preferred — they read better as names
        if theme_candidates and theme_candidates[0][2] > 0:
            name = theme_candidates[0][0].title()
            kind = "theme"
            top_terms = [t for t, _, _ in theme_candidates[:3]]
        elif tag_candidates and tag_candidates[0][2] > 0:
            name = "#" + tag_candidates[0][0]
            kind = "tag"
            top_terms = ["#" + t for t, _, _ in tag_candidates[:3]]
        else:
            top_emo = emotions.most_common(1)
            if top_emo:
                name = top_emo[0][0].capitalize() + " moments"
                kind = "emotion"
                top_terms = [e for e, _ in emotions.most_common(3)]
            else:
                name = "Cluster"
                kind = "other"
                top_terms = []

        return {
            "name": name, "kind": kind, "size": size,
            "top_terms": top_terms,
            "top_emotion": emotions.most_common(1)[0][0] if emotions else None,
        }

    clusters_out: List[dict] = []
    cluster_id_for_node: Dict[str, str] = {}
    used_names = set()
    for comp_root, members in by_component.items():
        info = _label_for(members)
        # Disambiguate if two clusters happen to land on the same primary
        # name (e.g. both pick #sleep). Suffix the second one with the
        # next-best term in its top_terms list.
        name = info["name"]
        if name in used_names and len(info.get("top_terms", [])) > 1:
            for alt in info["top_terms"][1:]:
                alt_name = alt.title() if not alt.startswith("#") else alt
                if alt_name not in used_names:
                    info["name"] = alt_name
                    break
        used_names.add(info["name"])

        cid = comp_root[:8] if comp_root != "__solo__" else "solo"
        cluster_record = {"id": cid, **info, "members": members}
        clusters_out.append(cluster_record)
        for nid in members:
            cluster_id_for_node[nid] = cid

    clusters_out.sort(key=lambda c: (c["id"] == "solo", -c["size"], c["name"]))

    # ---- Per-node degree + cluster assignment --------------------------
    deg = Counter()
    for ed in edges:
        deg[ed["source"]] += 1
        deg[ed["target"]] += 1
    for n in nodes:
        n["degree"] = deg.get(n["id"], 0)
        n["cluster_id"] = cluster_id_for_node.get(n["id"], "solo")

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": [
            {k: v for k, v in c.items() if k != "members"}
            for c in clusters_out
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "cluster_count": sum(1 for c in clusters_out if c["id"] != "solo"),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"app": "Innerbloom", "model": MODEL, "status": "ok"}


@app.get("/health")
def health():
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        return {"ollama": "up", "models": r.json()}
    except Exception as e:
        return {"ollama": "down", "error": str(e)}