"""
Reflect — Advanced AI Journal Backend
-------------------------------------
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

MODEL = os.environ.get("REFLECT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("REFLECT_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("REFLECT_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.environ.get("REFLECT_OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_EMBED_URL = os.environ.get("REFLECT_OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")

DATA_DIR = os.environ.get(
    "REFLECT_DATA_DIR",
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

app = FastAPI(title="Reflect — AI Journal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    personality: Optional[str] = "honest_coach"  # see PERSONALITIES below

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
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z'\-]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2]

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


def retrieve_similar(query: str, items: List[dict], k: int = 5,
                     embed_field: str = "embedding") -> List[dict]:
    """Embed the query, score each item by cosine similarity against its
    stored embedding, return top k. Items missing an embedding are skipped —
    use backfill_*_embeddings to populate them lazily."""
    qvec = embed(query)
    if not qvec:
        return []
    scored = []
    for it in items:
        vec = it.get(embed_field)
        if not vec:
            continue
        s = cosine_similarity(qvec, vec)
        if s > 0:
            scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]


def _entry_embed_text(entry: dict) -> str:
    parts = [entry.get("text", ""), entry.get("summary", "")]
    return "\n".join(p for p in parts if p)


def _chat_embed_text(turn: dict) -> str:
    return f"{turn.get('user','')}\n{turn.get('ai','')}".strip()


def backfill_entry_embeddings(entries: List[dict]) -> bool:
    """Fill any missing entry embeddings in place. Returns True if anything
    was added, so the caller knows to persist."""
    changed = False
    for e in entries:
        if not e.get("embedding"):
            vec = embed(_entry_embed_text(e))
            if vec:
                e["embedding"] = vec
                changed = True
    return changed


def backfill_chat_embeddings(history: List[dict]) -> bool:
    changed = False
    for t in history:
        if not t.get("embedding"):
            vec = embed(_chat_embed_text(t))
            if vec:
                t["embedding"] = vec
                changed = True
    return changed


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

    # Insights are produced by future engines (contradictions, triggers,
    # narrative). For now this stays empty but the slot exists.
    insights: List[dict] = []

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
            block.append(f"Reflect: {_truncate(t.get('ai',''), 240)}")
            block.append("")
        parts.append("\n".join(block).rstrip())

    # Patterns section: future insight engines + active goals (the closest
    # thing we have today to "things known about the user").
    pattern_lines: List[str] = []
    for ins in memory.get("insights", []):
        text = ins.get("text") if isinstance(ins, dict) else str(ins)
        if text:
            pattern_lines.append(f"- {text}")
    try:
        goals = load_file(GOALS_FILE, default=[])
        for g in [g for g in goals if not g.get("done")][:5]:
            cat = (g.get("category") or "goal").upper()
            pattern_lines.append(f"- [GOAL/{cat}] {g.get('text','')}")
    except Exception:
        pass
    if pattern_lines:
        parts.append("=== Known Patterns About User ===\n" + "\n".join(pattern_lines))

    parts.append(f"=== Current Message ===\n{message}\n\nReflect:")
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
# Chat (RAG with citations)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chat — personalities, modes, safety
# ---------------------------------------------------------------------------

PERSONALITIES = {
    "honest_coach": (
        "You are Reflect in 'brutally honest coach' mode. Be direct, clear, and unsparing — "
        "name the gap between what the user says and what they do. Skip motivational fluff. "
        "Care shows up as honesty, not encouragement. Short sentences. No hedging."
    ),
    "calm_therapist": (
        "You are Reflect in 'calm therapist' mode. Be warm, patient, and grounded. "
        "Reflect feelings back, validate before redirecting, ask one gentle question at a time. "
        "Never diagnose or pathologize. Use plain language."
    ),
    "analytical_observer": (
        "You are Reflect in 'analytical observer' mode. Be precise, neutral, evidence-based. "
        "Quote specific entries when claiming a pattern. Distinguish observation from inference. "
        "Avoid emotional framing unless the user asks for it."
    ),
}

JOURNAL_MODE_RULES = (
    "You have access to the user's journal entries and past conversations in the context above. "
    "Ground every claim in specific entries — never invent a memory. "
    "Cite journal entries with [cite:id] (use the short id shown in brackets) when referencing them. "
    "If the context does not contain the answer, say so plainly rather than guessing. "
    "When the user is exploring something fuzzy, ask one clarifying question back instead of guessing."
)

GENERAL_MODE_RULES = (
    "You are in 'general' mode — the user is treating you as a normal assistant, not as their journal. "
    "Do NOT pull from journal entries unless the user explicitly asks. "
    "Help with whatever they bring: questions, ideas, planning, code, writing. "
    "Stay concise. Ask a clarifying question if the request is ambiguous."
)


def build_system_prompt(mode: str, personality: str) -> str:
    p = PERSONALITIES.get(personality, PERSONALITIES["honest_coach"])
    rules = GENERAL_MODE_RULES if mode == "general" else JOURNAL_MODE_RULES
    return p + "\n\n" + rules + "\n\nKeep replies concise unless the user asks for detail."


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
    # Strip literal [cite:...] tokens from the visible reply; UI shows pills.
    return re.sub(r"\s*\[cite:[a-z0-9\-]+\]", "", reply).strip()


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


TOOLS = {
    "search_entries": tool_search_entries,
    "get_entry": tool_get_entry,
    "list_themes": tool_list_themes,
    "period_summary": tool_period_summary,
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
- search_entries(query: str, k: int=5) — semantic search over the user's journal
- get_entry(ref: str) — fetch one entry by 8-char id prefix or by date (YYYY-MM-DD)
- list_themes(period_days: int=30) — top tags / emotions / themes for the period
- period_summary(period: "week"|"month") — entry count + avg intensity + recent summaries

Rules:
- Pick 0–3 tools. Skip tools entirely for small talk like "hi" or "thanks".
- search_entries is the most common pick — call it whenever the user references a feeling, person, topic, or time.
- If the message is genuinely ambiguous (e.g. "what should I do?" with no context), set ask_back to ONE clarifying question and leave tools empty.
- Never invent ids or dates; only the user's question is your input.

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
    obs_block = "\n".join(lines).strip() or "(no tool results — answer from general knowledge or ask a clarifying question)"

    return (
        "=== Tool Observations ===\n"
        f"{obs_block}\n\n"
        "=== User Message ===\n"
        f"{message}\n\n"
        "Now draft your reply. Cite specific entries with [cite:id] using the "
        "short id from the brackets above. If the observations don't fully "
        "answer the question, say what's missing — don't invent.\n\n"
        "Reflect:"
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

    personality = data.personality if data.personality in PERSONALITIES else "honest_coach"
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
            if t["name"] == "search_entries" and isinstance(result, list):
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
    personality = data.personality if data.personality in PERSONALITIES else "honest_coach"
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
    personality = data.personality if data.personality in PERSONALITIES else "honest_coach"
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


def _safe_dt(s):
    try: return datetime.fromisoformat(s)
    except Exception: return None


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
    goals = load_file(GOALS_FILE, default=[])
    active_goals = [g["text"] for g in goals if not g.get("done")][:3]
    goals_context = ""
    if active_goals:
        goals_context = "\nUser's active goals: " + "; ".join(active_goals) + "."

    prompt = f"""You write sharp, personal journal prompts. Based on this person's recent emotional themes,
write ONE journaling prompt — one sentence, under 22 words.
The prompt should invite honest self-examination, not generic reflection.
Connect to a specific theme or tension you notice. No preamble. No quotes. Just the prompt.

Recent mood and topics:
{context}{goals_context}
"""
    out = call_llama(prompt, temperature=0.92).strip().strip('"').strip()
    out = out.splitlines()[0] if out else "What are you tolerating right now that you haven't admitted to yourself?"
    return {"prompt": out}


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
    lines = ["# Reflect — Journal Export\n"]
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
# Grow — self-improvement & self-reflection endpoints
# ---------------------------------------------------------------------------

GOALS_FILE = os.path.join(DATA_DIR, "goals.json")


class GoalCreate(BaseModel):
    text: str


class GoalToggle(BaseModel):
    done: bool


@app.get("/grow/patterns")
def grow_patterns():
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return {"result": "Start journaling — your growth patterns will surface after a few entries."}
    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:30]
    combined = "\n\n".join(
        f"[{e.get('timestamp','')[:10]}] ({e.get('emotion','?')}) {e.get('text','')[:300]}"
        for e in recent
    )
    goals = load_file(GOALS_FILE, default=[])
    goals_context = ""
    if goals:
        active = [g["text"] for g in goals if not g.get("done")][:3]
        if active:
            goals_context = f"\nUser's active goals: {'; '.join(active)}."

    prompt = f"""You are a sharp, empathetic growth coach. Analyze these journal entries for deep patterns.
Write in second person. Be specific — name real themes, not vague categories.
Every sentence should feel like it could only apply to this person.{goals_context}

Respond with these four sections (use **bold** headers):

**Strengths in motion** — 2-3 genuine strengths you see them actually using (not just possessing)
**The growth edge** — the most important area to develop right now, and why it keeps showing up
**Thinking patterns** — one or two cognitive patterns (e.g., perfectionism, avoidance, rumination, reframing) with a specific example from entries
**Your next lever** — one concrete, specific action that would compound their growth most right now

Journal entries:
{combined}
"""
    return {"result": call_llama(prompt, temperature=0.65)}


@app.get("/grow/wins")
def grow_wins():
    entries = load_file(JOURNAL_FILE)
    if not entries:
        return {"wins": [], "message": "Write your first entry and your wins will surface here."}
    cutoff = datetime.now() - timedelta(days=14)
    recent = [e for e in entries if _safe_dt(e.get("timestamp")) and _safe_dt(e["timestamp"]) >= cutoff]
    if not recent:
        recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:10]
    combined = "\n\n".join(
        f"[{e.get('timestamp','')[:10]}] {e.get('text','')[:350]}" for e in recent
    )
    prompt = f"""Read these journal entries and extract genuine wins — achievements, growth moments,
acts of courage, emotional regulation, showing up when it was hard, anything they did well.
Include small wins. Be specific — paraphrase the actual event, don't be vague.
Return 4-8 bullet points. No preamble, no numbering. Start each with the win itself.

Entries:
{combined}
"""
    raw = call_llama(prompt, temperature=0.6)
    wins = [w.strip("-•* ").strip() for w in raw.splitlines() if w.strip() and len(w.strip()) > 10][:8]
    return {"wins": wins}


class StepToggle(BaseModel):
    done: bool


def _generate_goal_plan(goal_text: str, journal_context: str = "") -> dict:
    """Call LLM to generate a structured plan for a goal."""
    categories = "health, career, learning, relationships, mindset, finance, creativity, productivity, other"
    context_block = f"\nUser's recent journal context:\n{journal_context}\n" if journal_context else ""

    prompt = f"""You are a personal coach. Create a structured, actionable plan for this goal.
Return ONLY a compact JSON object — no prose, no markdown fences.{context_block}
Goal: "{goal_text}"

Schema:
{{
  "category": "one of: {categories}",
  "steps": ["3-5 specific, ordered, actionable steps — each a complete action sentence"],
  "habits": ["2-4 daily or weekly habits that directly support this goal"],
  "reflection_prompt": "one thoughtful question (1 sentence) to journal about regarding this goal"
}}

JSON:"""

    raw = call_llama(prompt, temperature=0.4, timeout=120)
    data = extract_json(raw) or {}

    steps_raw = data.get("steps") or []
    habits_raw = data.get("habits") or []
    steps = [
        {"id": str(uuid.uuid4()), "text": str(s).strip(), "done": False}
        for s in steps_raw if str(s).strip()
    ][:6]
    habits = [str(h).strip() for h in habits_raw if str(h).strip()][:5]
    category = str(data.get("category", "other")).lower().strip()
    if category not in categories.replace(",", "").split():
        category = "other"
    reflection = str(data.get("reflection_prompt", "")).strip() or \
        "What would make real progress on this goal feel like this week?"

    return {"category": category, "steps": steps, "habits": habits, "reflection_prompt": reflection}


@app.get("/grow/goals")
def get_goals():
    return {"goals": load_file(GOALS_FILE, default=[])}


@app.post("/grow/goals")
def create_goal(body: GoalCreate):
    goal_text = body.text.strip()[:300]
    if not goal_text:
        raise HTTPException(400, "Goal text is required")

    entries = load_file(JOURNAL_FILE)
    recent = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:5]
    journal_context = "\n".join(
        f"- ({e.get('emotion','?')}) {e.get('summary') or e.get('text','')[:120]}"
        for e in recent
    )

    plan = _generate_goal_plan(goal_text, journal_context)

    goal = {
        "id": str(uuid.uuid4()),
        "text": goal_text,
        "category": plan["category"],
        "steps": plan["steps"],
        "habits": plan["habits"],
        "reflection_prompt": plan["reflection_prompt"],
        "done": False,
        "created": datetime.now().isoformat(),
    }
    goals = load_file(GOALS_FILE, default=[])
    goals.append(goal)
    save_file(GOALS_FILE, goals)
    return {"goal": goal}


@app.patch("/grow/goals/{goal_id}")
def update_goal(goal_id: str, body: GoalToggle):
    goals = load_file(GOALS_FILE, default=[])
    for g in goals:
        if g["id"] == goal_id:
            g["done"] = body.done
            save_file(GOALS_FILE, goals)
            return {"goal": g}
    raise HTTPException(404, "Goal not found")


@app.patch("/grow/goals/{goal_id}/steps/{step_id}")
def toggle_step(goal_id: str, step_id: str, body: StepToggle):
    goals = load_file(GOALS_FILE, default=[])
    for g in goals:
        if g["id"] == goal_id:
            for s in g.get("steps", []):
                if s["id"] == step_id:
                    s["done"] = body.done
                    done_count = sum(1 for st in g["steps"] if st.get("done"))
                    if done_count == len(g["steps"]) and g["steps"]:
                        g["done"] = True
                    save_file(GOALS_FILE, goals)
                    return {"goal": g}
            raise HTTPException(404, "Step not found")
    raise HTTPException(404, "Goal not found")


@app.delete("/grow/goals/{goal_id}")
def delete_goal(goal_id: str):
    goals = load_file(GOALS_FILE, default=[])
    goals = [g for g in goals if g["id"] != goal_id]
    save_file(GOALS_FILE, goals)
    return {"status": "deleted"}


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


def fetch_audio_features(cfg, track_ids: List[str]) -> List[dict]:
    feats = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        try:
            data = spotify_get(cfg, "/audio-features", {"ids": ",".join(batch)})
            for f in data.get("audio_features", []):
                if f:
                    feats.append(f)
        except HTTPException:
            continue
    return feats


@app.get("/spotify/mood")
def spotify_mood():
    """
    Aggregate audio features for recently-played tracks and correlate the
    day-by-day listening 'valence/energy' with the day-by-day journal intensity.
    """
    cfg = spotify_authed()
    if not cfg:
        raise HTTPException(401, "Not connected to Spotify")

    # 1. Recent tracks (last 50 plays)
    data = spotify_get(cfg, "/me/player/recently-played", {"limit": 50})
    items = data.get("items", [])
    if not items:
        return {"points": [], "summary": "No recent plays found."}

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
        })
        ids.append(tr["id"])

    feats = fetch_audio_features(cfg, ids)
    feat_by_id = {f["id"]: f for f in feats}

    # 2. Aggregate per day
    by_day = defaultdict(list)
    for p in played:
        f = feat_by_id.get(p["id"])
        if not f: continue
        d = (p.get("played_at") or "")[:10]
        if d:
            by_day[d].append(f)

    # 3. Journal intensity per day (last 30 days)
    journal = load_file(JOURNAL_FILE)
    j_by_day = defaultdict(list)
    for e in journal:
        try:
            d = datetime.fromisoformat(e["timestamp"]).date().isoformat()
            if e.get("intensity") is not None:
                j_by_day[d].append(e["intensity"])
        except Exception:
            pass

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
        j_intensities = j_by_day.get(d, [])
        journal_intensity = round(sum(j_intensities) / len(j_intensities), 2) if j_intensities else None
        points.append({
            "date": d,
            "valence": valence,
            "energy": energy,
            "danceability": danceability,
            "tempo": tempo,
            "journal_intensity": journal_intensity,
            "play_count": len(day_feats),
        })

    # 4. Overall averages
    all_val = [p["valence"] for p in points if p["valence"] is not None]
    all_en = [p["energy"] for p in points if p["energy"] is not None]
    avg_valence = round(sum(all_val) / len(all_val), 3) if all_val else None
    avg_energy = round(sum(all_en) / len(all_en), 3) if all_en else None

    return {
        "points": points,
        "avg_valence": avg_valence,
        "avg_energy": avg_energy,
        "play_count": len(played),
        "recent": played[:10],
    }


@app.get("/spotify/insight")
def spotify_insight():
    """LLM interpretation of the Spotify ↔ journal correlation."""
    data = spotify_mood()
    points = data.get("points", [])
    val = data.get("avg_valence")
    en = data.get("avg_energy")

    # Build a compact textual summary for the LLM
    lines = []
    for p in points[-14:]:
        if p["valence"] is None and p["journal_intensity"] is None:
            continue
        lines.append(
            f"- {p['date']}: valence={p['valence']}, energy={p['energy']}, "
            f"plays={p['play_count']}, journal_intensity={p['journal_intensity']}"
        )
    if not lines:
        return {"insight": "Not enough overlapping data yet. Listen a little and journal a little more."}

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
    return {"insight": call_llama(prompt, temperature=0.5)}


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

    prompt = f"""Burnout signal: level={burnout.get('level')} · "
"avg_mood={burnout.get('avg_mood')} · exhaustion_count={burnout.get('exhaustion_count')}/{burnout.get('entry_count')} · drift={burnout.get('drift')}
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
def get_graph(limit: int = 250, min_weight: float = 0.18):
    """Return a memory graph of recent entries.

    - `limit`: cap on number of nodes (most recent first)
    - `min_weight`: minimum Jaccard similarity for an edge to be drawn
    """
    entries = load_file(JOURNAL_FILE)
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    entries = entries[:max(10, min(limit, 500))]

    # Feature set per entry: lowercased tags + themes + emotion
    feats: Dict[str, set] = {}
    for e in entries:
        s = set()
        for t in e.get("tags", []) or []:
            v = (str(t) or "").strip().lower()
            if v: s.add(v)
        for t in e.get("themes", []) or []:
            v = (str(t) or "").strip().lower()
            if v: s.add(v)
        emo = (e.get("emotion") or "").strip().lower()
        if emo: s.add(f"emo:{emo}")
        feats[e["id"]] = s

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

    edges = []
    ids = list(feats.keys())
    for i in range(len(ids)):
        a = feats[ids[i]]
        if not a: continue
        for j in range(i + 1, len(ids)):
            b = feats[ids[j]]
            if not b: continue
            shared = a & b
            if not shared: continue
            w = len(shared) / len(a | b)  # Jaccard
            if w < min_weight: continue
            edges.append({
                "source": ids[i],
                "target": ids[j],
                "weight": round(w, 3),
                "shared": sorted(list(shared))[:5],
            })

    # Per-node degree so the UI can size hubs larger
    deg = Counter()
    for ed in edges:
        deg[ed["source"]] += 1
        deg[ed["target"]] += 1
    for n in nodes:
        n["degree"] = deg.get(n["id"], 0)

    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"app": "Reflect", "model": MODEL, "status": "ok"}


@app.get("/health")
def health():
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        return {"ollama": "up", "models": r.json()}
    except Exception as e:
        return {"ollama": "down", "error": str(e)}