"""
retriever.py

Signal-based retrieval over a personal event stream.
No embeddings required — uses keyword overlap, recency decay,
urgency signals, and source weighting to rank events.

Design rationale:
  - In production with 10k+ events, this approach stays fast (<50ms)
    because it avoids a round-trip to an embedding API.
  - For larger corpora, swap score_events() with a two-stage pipeline:
    ANN retrieval (FAISS/Qdrant) → rerank with this scorer.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_NOW = datetime(2026, 4, 13, 3, 0, 0, tzinfo=timezone.utc)

# How quickly older events lose relevance. Higher = steeper decay.
RECENCY_DECAY_DAYS = 14

# Sources we trust more for commitments / deadlines
HIGH_TRUST_SOURCES = {"email", "calendar", "reminder"}

# Words that spike urgency score
URGENCY_KEYWORDS = {
    "urgent", "asap", "deadline", "overdue", "by eod", "by today",
    "by tomorrow", "due", "must", "critical", "important", "confirm",
    "waiting", "blocked", "follow up", "followup", "follow-up",
    "reminder", "don't forget", "need to", "have to",
}

# Words that mark procrastination / stalled items
STALL_KEYWORDS = {
    "still", "haven't", "not yet", "keep forgetting", "procrastinat",
    "delayed", "postponed", "pending", "stuck", "waiting on",
    "need to", "should", "meant to",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load and lightly normalise the event stream."""
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)

    for ev in events:
        # Parse timestamp once, store as datetime object
        raw_ts = ev.get("timestamp", "")
        try:
            ev["_dt"] = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ev["_dt"] = SCENARIO_NOW  # fallback

        # Lower-case content for matching
        ev["_text"] = (ev.get("content", "") + " " + ev.get("source", "")).lower()

    return events


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def recency_score(event: dict) -> float:
    """
    Exponential decay: events from today score ~1.0,
    events 14 days ago score ~0.37, a month ago ~0.1.
    """
    delta_days = (SCENARIO_NOW - event["_dt"]).total_seconds() / 86400
    # Future events (scheduled) also score high
    if delta_days < 0:
        return 1.0
    return math.exp(-delta_days / RECENCY_DECAY_DAYS)


def keyword_score(event: dict, query_terms: set[str]) -> float:
    """Fraction of query terms found in the event text."""
    if not query_terms:
        return 0.0
    text = event["_text"]
    hits = sum(1 for t in query_terms if t in text)
    return hits / len(query_terms)


def urgency_score(event: dict) -> float:
    """Boost events that contain urgency signals."""
    text = event["_text"]
    hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text)
    return min(hits * 0.15, 1.0)


def stall_score(event: dict) -> float:
    """Score events that hint at procrastination or stalled work."""
    text = event["_text"]
    hits = sum(1 for kw in STALL_KEYWORDS if kw in text)
    return min(hits * 0.2, 1.0)


def source_weight(event: dict) -> float:
    """Trust multiplier by source type."""
    src = event.get("source", "").lower()
    if src in HIGH_TRUST_SOURCES:
        return 1.3
    if src == "slack":
        return 1.1
    return 1.0


# ---------------------------------------------------------------------------
# Query → keyword expansion
# ---------------------------------------------------------------------------

# Maps query intent to relevant search terms
#
# "requires_topic_match": when True, an event MUST contain at least one
# topic_terms hit to be eligible at all. This prevents high-recency,
# high-source-weight noise (e.g. unrelated calendar invites around the
# same dates) from crowding out genuinely relevant but lower-recency
# events. Without this gate, a same-week "Interview calibration" meeting
# can outscore a real UIE Slack thread purely on recency + source weight.
QUERY_PROFILES = {
    "focus_today": {
        "terms": {"today", "urgent", "deadline", "due", "meeting", "call",
                  "review", "send", "submit", "finish", "complete", "prepare"},
        "boost_urgency": True,
        "boost_stall": False,
        "requires_topic_match": False,
        "topic_terms": set(),
    },
    "commitments_at_risk": {
        "terms": {"deadline", "due", "commit", "promise", "by", "confirm",
                  "send", "deliver", "submit", "waiting", "overdue", "eod"},
        "boost_urgency": True,
        "boost_stall": True,
        "requires_topic_match": False,
        "topic_terms": set(),
    },
    "procrastination": {
        "terms": {"still", "haven't", "pending", "keep forgetting", "delayed",
                  "postponed", "stuck", "should", "meant to", "not yet"},
        "boost_urgency": False,
        "boost_stall": True,
        "requires_topic_match": False,
        "topic_terms": set(),
    },
    "uie_proposal": {
        "terms": {"uie", "proposal", "client", "contract",
                  "deck", "slides", "pricing", "scope", "pitch", "sign"},
        "boost_urgency": True,
        "boost_stall": True,
        # Topic-specific query: gate strictly on "uie" or "proposal" so
        # unrelated same-week calendar noise (interview loops, other
        # deals, unrelated standups) can't ride in on recency alone.
        "requires_topic_match": True,
        "topic_terms": {"uie", "proposal"},
    },
}


def extract_query_terms(query: str) -> set[str]:
    """Tokenise a free-text query into search terms."""
    stopwords = {"what", "should", "i", "am", "at", "is", "are", "the",
                 "a", "an", "on", "in", "of", "to", "for", "me", "my",
                 "have", "been", "do", "how", "when", "everything", "related"}
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return {t for t in tokens if t not in stopwords and len(t) > 2}


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    events: list[dict],
    profile_key: str,
    free_query: str = "",
    top_k: int = 20,
) -> list[dict]:
    """
    Score and rank events for a given query profile.

    Returns top_k events with their scores attached (_score).

    Scaling note:
      With 10k+ events, pre-filter by source or time window first,
      then score only the candidate set. This keeps latency sub-100ms
      even without vector indexing.
    """
    profile = QUERY_PROFILES.get(profile_key, {})
    base_terms = profile.get("terms", set())
    boost_urgency = profile.get("boost_urgency", False)
    boost_stall = profile.get("boost_stall", False)
    requires_topic_match = profile.get("requires_topic_match", False)
    topic_terms = profile.get("topic_terms", set())

    # Merge profile terms with any free-text terms from the query
    free_terms = extract_query_terms(free_query)
    all_terms = base_terms | free_terms

    scored = []
    for ev in events:
        # Hard gate: for topic-specific queries, an event must contain at
        # least one topic term to be eligible at all. This stops recency
        # and source-weight from smuggling in unrelated same-week events
        # (e.g. an unrelated interview loop or SOW negotiation) just
        # because they happen to be on a trusted source near "now".
        if requires_topic_match:
            if not any(term in ev["_text"] for term in topic_terms):
                continue

        r = recency_score(ev)
        k = keyword_score(ev, all_terms)
        u = urgency_score(ev) if boost_urgency else 0.0
        s = stall_score(ev) if boost_stall else 0.0
        w = source_weight(ev)

        # Weighted composite score
        score = w * (0.35 * r + 0.40 * k + 0.15 * u + 0.10 * s)

        if score > 0.05:  # ignore near-zero matches
            ev_copy = dict(ev)
            ev_copy["_score"] = round(score, 4)
            ev_copy["_score_breakdown"] = {
                "recency": round(r, 3),
                "keyword": round(k, 3),
                "urgency": round(u, 3),
                "stall": round(s, 3),
                "source_weight": round(w, 2),
            }
            scored.append(ev_copy)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]