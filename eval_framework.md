# Memorae — Evaluation Framework

---

## Philosophy

The goal of evaluation is not to prove the system works on happy-path examples.
It is to find the edges where it breaks, measure how bad the breaks are, and
track whether things get better or worse as the system evolves.

This framework has three layers:

| Layer | Speed | LLM Required | Purpose |
|---|---|---|---|
| Offline evals | Fast (<1s) | No | Catch regressions, verify retrieval logic |
| Rubric evals | Slow (~30s) | Yes | Assess answer quality |
| Regression tests | Medium | Optional | Detect silent quality regressions |

---

## 1. Offline Evals

These are deterministic unit tests that run without any LLM calls.
They should run on every commit.

### Retrieval Tests

| Test | What it checks | Pass condition |
|---|---|---|
| `retriever_returns_results` | Every query profile returns ≥1 event | len(results) > 0 |
| `uie_profile_finds_uie_events` | UIE profile surfaces UIE-relevant content | ≥1 event with "uie" or "proposal" in top-20 |
| `scores_sorted_descending` | Results are properly ranked | scores[i] >= scores[i+1] for all i |
| `future_events_score_high` | Scheduled future events are surfaced | recency_score(future_event) >= 1.0 |
| `old_events_score_low` | Events from 60 days ago decay | recency_score(old_event) < 0.05 |
| `urgency_keywords_boost_score` | Urgent events rank higher than neutral ones | urgent_event._score > neutral_event._score |
| `high_trust_sources_boosted` | Email/calendar outrank WhatsApp for same content | email_score > whatsapp_score |

### Context Construction Tests

| Test | What it checks | Pass condition |
|---|---|---|
| `deduplication_works` | Duplicate events appear only once | content appears exactly once in context block |
| `long_content_truncated` | Events over 400 chars are trimmed | no event content exceeds MAX_CONTENT_CHARS in block |
| `events_sorted_chronologically` | Context block is time-ordered | timestamps increase monotonically in block |
| `context_respects_max_events` | Hard cap on events is honoured | len(context_events) <= MAX_CONTEXT_EVENTS |

Run all offline evals:
```bash
python eval.py --data memorae_mock_events.json --mode offline
```

---

## 2. Rubric Evals (LLM-as-Judge)

For subjective queries, we use a secondary LLM call to score the answer
against a structured rubric. The judge model sees the answer and the context
block (not the full event stream) to avoid unfair advantage.

### What Makes a Good Answer?

A good answer is **specific**, **grounded**, **time-aware**, and **honest about uncertainty**.

Vague answers like *"You should focus on your pending tasks and meetings"* score 0. Specific answers like *"Your top priority today is confirming the UIE proposal pricing with the client — they're waiting and the deadline is tomorrow"* score high.

### Rubric: "What should I focus on today?"

| Criterion | Weight |
|---|---|
| Mentions specific tasks by name (not generic categories) | High |
| Items are prioritised (not a flat undifferentiated list) | Medium |
| References actual names, deadlines, or event content | High |
| Does not fabricate details absent from the event stream | High |
| Appropriate for the current date (April 13, 2026) | Medium |

**Minimum passing score**: 3 out of 5 criteria met.

### Rubric: "What commitments am I at risk of missing?"

| Criterion | Weight |
|---|---|
| Identifies specific commitments (not generic risks) | High |
| Names who the commitment is with | Medium |
| Explains why each item is at risk | High |
| Flags items that are overdue or near-deadline | High |
| Reasoning is grounded in the event stream | High |

**Minimum passing score**: 3 out of 5 criteria met.

### Rubric: "What have I been procrastinating on?"

| Criterion | Weight |
|---|---|
| Identifies recurring or unresolved items | High |
| Stays factual; avoids moralising | Low |
| Groups related items together | Medium |
| Items are genuinely pending (not already completed) | High |
| Answer is specific, not a list of platitudes | High |

**Minimum passing score**: 3 out of 5 criteria met.

### Rubric: "Summarise the UIE proposal."

| Criterion | Weight |
|---|---|
| Covers current status | High |
| Identifies what is outstanding | High |
| Includes deadlines or pricing if present in events | Medium |
| Presented as a coherent briefing (not raw event dump) | Medium |
| Identifies any risks or blockers | Medium |

**Minimum passing score**: 3 out of 5 criteria met.

Run rubric evals:
```bash
python eval.py --data memorae_mock_events.json --mode rubric
```

---

## 3. Regression Tests

Regression tests compare a current run against a saved baseline to detect
silent quality degradation — when a change to scoring weights or prompt
wording makes answers shorter, less specific, or covers fewer events.

### How to create a baseline

```bash
python main.py --data memorae_mock_events.json --output baseline.json
```

### What is checked

| Signal | Regression if… |
|---|---|
| Answer length | Current answer < 50% of baseline length |
| Events in context | Δ > 5 events from baseline |
| Key entities mentioned | A named entity present in baseline is absent in current |

### Example regression test case

**Query**: "Summarise the UIE proposal"  
**Baseline answer length**: 420 characters  
**Current answer length**: 180 characters  
→ **REGRESSION**: Answer is 43% of baseline length — likely a prompt or retrieval regression.

Run regression tests:
```bash
python eval.py --data memorae_mock_events.json --mode regression --baseline baseline.json
```

---

## 4. Metrics Summary

| Metric | How measured | Target |
|---|---|---|
| Retrieval precision | Fraction of top-k events that are relevant (manual spot-check) | >70% |
| Rubric pass rate | Criteria met / total criteria | ≥3/5 per query |
| Answer specificity | Named entities per answer (count noun phrases) | ≥3 per answer |
| Hallucination rate | Claims in answer not traceable to any event | 0 |
| Latency (p50) | Seconds from query to printed answer | <5s |
| Latency (p95) | Same, at 95th percentile | <8s |
| Context efficiency | Tokens used / 100k budget | <10% |

---

## 5. Edge Cases and Adversarial Inputs

| Scenario | Expected behaviour |
|---|---|
| Query about a topic with no events | Answer says "no relevant events found" — does not hallucinate |
| All events are from 60+ days ago | Recency decay makes all scores low; answer acknowledges staleness |
| Same event appears from 3 sources | Deduplication keeps 1; answer is not repetitive |
| Contradictory events (meeting rescheduled) | Newer event wins; answer reflects updated time |
| Very short event content (<10 chars) | Still scored, but low keyword overlap keeps rank low |
| Empty event stream | Graceful error — "No events to analyse" |

---

## 6. Manual Evaluation Checklist

For each query, a human reviewer should check:

- [ ] Would a real user find this answer useful right now?
- [ ] Is every specific claim traceable to an event in `selected_context`?
- [ ] Does the answer reflect the correct date (April 13, 2026)?
- [ ] Are the top 3 selected events genuinely the most relevant ones?
- [ ] Does the reasoning explanation match what the answer actually uses?