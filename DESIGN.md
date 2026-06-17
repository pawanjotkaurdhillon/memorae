# Memorae — Design Document

**Personal Memory Query Engine**  
Pawanjot kaur 
Date: 17 june 2026

---

## Overview

Memorae is a query engine that turns a raw personal event stream (messages, emails, notes, reminders) into actionable answers. Given a noisy stream of events with no labels or ground truth, the system infers what matters based on content, timing, and source signals.

The core design philosophy is: **choose the right context, not the biggest context.** Dumping everything into a language model is not retrieval — it is abdication of engineering responsibility.

---

## 1. Retrieval Architecture

### Why not embeddings?

The first question I had to answer was whether to use semantic embeddings or a signal-based ranker. For this dataset and use case, I chose a **signal-based hybrid scorer** for three reasons:

1. **Speed**: No round-trip to an embedding API. Scoring 500 events takes under 10ms locally.
2. **Interpretability**: Every score is decomposable into recency, keyword, urgency, and stall sub-scores — easy to debug and tune.
3. **Control**: Personal memory queries are often highly structured ("what is at risk?", "what about the UIE proposal?") — keyword and intent matching handles these well.

For a production system with 50k+ events and free-text queries with no clear intent, I would add a first-stage ANN retrieval (FAISS or Qdrant) using sentence embeddings, then rerank the top-100 candidates with the signal scorer.

### Scoring Function

Each event receives a composite score:

```
score = source_weight × (0.35×recency + 0.40×keyword + 0.15×urgency + 0.10×stall)
```

**Recency** uses exponential decay with a 14-day half-life. Events from today score ~1.0. Events from 30 days ago score ~0.12. Future events (scheduled meetings, deadlines) score 1.0 regardless.

**Keyword** measures overlap between the event text and a query-specific term set. Each query profile defines its own vocabulary (e.g., "commitments at risk" includes "deadline", "overdue", "waiting on").

**Urgency** detects words like "ASAP", "deadline", "critical", "must", "EOD" — signals that something needs action soon.

**Stall** detects procrastination language — "still", "haven't", "pending", "keep forgetting" — used to surface items the user is avoiding.

**Source weight** applies a trust multiplier: calendar and email events get 1.3×, Slack gets 1.1×, other sources get 1.0×. Calendar invites and emails represent external commitments; Slack is often more signal-dense than WhatsApp.

### Query Profiles

Rather than a single retriever, each query type has a named profile that configures which signals to boost. This keeps retrieval intent-aware without complex NLP:

| Profile | Key Signal Boosts |
|---|---|
| `focus_today` | recency, urgency |
| `commitments_at_risk` | urgency + stall, deadline keywords |
| `procrastination` | stall signals, repeated mentions |
| `uie_proposal` | topic-specific keywords (uie, proposal, pitch, scope) |

---

## 2. Memory Architecture

### Event Normalisation

On load, each event is:
- Parsed into a UTC `datetime` object
- Lower-cased into a `_text` field for matching
- Left otherwise untouched (no labels added to raw data)

### No Persistent State (by design, for now)

The current implementation is stateless — it re-scores the full event stream on each query. This is intentional for a v1:

- It is correct: no risk of stale cached scores
- It is fast enough: scoring 500 events takes <10ms
- It is simple to test and reason about

For production, I would add:

1. **A persistent index** (SQLite or Qdrant) with pre-computed embeddings, updated incrementally as new events arrive
2. **Session memory**: track what the user has already been told in this session to avoid repetition
3. **Entity extraction**: identify people, projects, and deadlines as named entities, then maintain a live "entity state" (e.g., "UIE proposal: status=pending, deadline=April 15")

---

## 3. Context Construction Strategy

### Budget-Aware Packing

The production system is assumed to have a 100k-token context budget. We use a small fraction of it deliberately:

- **Target**: ~6,000 tokens per query
- **Max events in context**: 15 (configurable)
- **Content truncation**: 400 chars per event, trimmed at word boundaries

This leaves ~94k tokens available for conversation history, system instructions, and multi-turn dialogue — essential for a real assistant.

### Deduplication

Events are fingerprinted by their first 120 characters of content. Duplicates (e.g., the same reminder synced from two sources) are dropped before packing. This matters especially for notification-heavy sources like WhatsApp.

### Chronological Ordering

After scoring (which determines *which* events to include), selected events are re-sorted chronologically before being sent to the LLM. This preserves narrative coherence — the model can follow the progression of a situation over time rather than seeing it in random order.

---

## 4. Contradiction and Recency Handling

### Recency Wins by Default

If two events contradict each other (e.g., "meeting at 3pm" vs. "meeting rescheduled to 5pm"), the recency score naturally surfaces the newer event higher. The LLM is instructed to be time-aware and treat later events as updates to earlier ones.

### Explicit Uncertainty

The generation prompt instructs the model: *"If something is uncertain, say what is uncertain and explain your reasoning."* This prevents false confidence when the event stream is incomplete or ambiguous.

### Known Limitation

The current system does not explicitly detect contradictions — it relies on recency scoring to surface the right event and on the LLM to resolve conflicts in language. A stronger approach would be to cluster events by entity/topic, detect conflicts within clusters, and pass the conflict explicitly to the LLM ("Event A says X, but Event B 3 days later says Y — treat B as the update").

---

## 5. Failure Modes

| Failure Mode | Cause | Mitigation |
|---|---|---|
| Silent completions | Task completed but no event recorded | Uncertainty flagging in prompts |
| Duplicate noise | Same event from multiple sources | Content deduplication on load |
| Stale urgency | Old urgent event never resolved | Recency decay reduces old events' score |
| Missing context | Relevant event uses different vocabulary than query profile | Add free-text query term expansion |
| LLM hallucination | Model invents details not in events | Low temperature (0.2), "grounded in events only" instruction |
| Keyword mismatch | User says "deck" but event says "presentation" | Query profiles include synonyms; future: embeddings |

---

## 6. Scaling to Larger Datasets

### From 500 → 50,000 events

| Component | Change |
|---|---|
| Retrieval | Add FAISS/Qdrant ANN index; score top-100 candidates only |
| Embeddings | Generate offline with `sentence-transformers` (free) or `text-embedding-3-small` |
| Index updates | Incremental — embed new events as they arrive, no full re-index |
| Deduplication | Move to hash-based dedup at write time, not query time |
| Context | Same budget strategy works; just source from index instead of full scan |

### Entity State Layer

At scale, maintain a live entity state store:
- Extract entities (people, projects, deadlines) from events using a small NER model
- Maintain a "current state" per entity (e.g., UIE proposal → last status, next action, owner)
- Query this state layer first, then fall back to raw event retrieval for detail

This reduces per-query token cost significantly — instead of packing 15 raw events, you pack 1 concise entity summary + 5 supporting raw events.

---

## 7. Optimization: Under 2 Seconds, 80% Cost Reduction

The current system makes 2 LLM calls per query (answer + reasoning). At Groq's pricing, the cost is already very low, but the latency target forces architectural changes.

### Changes

**Retrieval**: Already fast (<10ms). No change needed.

**Merge answer + reasoning into one call**: Use structured output (JSON mode) to get both the answer and reasoning explanation in a single LLM call. Saves ~300–500ms and halves API cost.

**Model routing**: Use `llama-3.1-8b-instant` (faster, cheaper) for queries where the context is small (<10 events) or the query is simple. Reserve `llama-3.3-70b` for complex, multi-event queries.

**Precompute entity summaries**: Run a background job every 15 minutes to refresh entity state (UIE proposal status, pending commitments). At query time, inject the summary directly — no need to retrieve and re-read raw events.

**Cache recent answers**: For stable queries like "UIE proposal status", cache the answer for 10 minutes. Invalidate if a new relevant event arrives.

### Tradeoffs

| Change | Gain | Cost |
|---|---|---|
| Merge LLM calls | -40% latency, -50% cost | Slightly less separation of concerns |
| Smaller model | -30% latency, -60% cost | Slight quality drop on complex queries |
| Precomputed summaries | -50% latency | Risk of serving stale data; adds infra complexity |
| Answer caching | Near-zero latency for cached queries | Stale if events arrive between cache refresh |

A realistic target: merge calls + route to 8b model for simple queries + 10-min cache = **under 1.5s, ~75% cost reduction** with acceptable quality tradeoff.