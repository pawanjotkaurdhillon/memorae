# Memorae — Personal Memory Query Engine

A signal-based query engine that answers natural-language questions about your personal event stream. Built as a take-home assignment response.

---

## What it does

Given a raw stream of personal events (Slack messages, emails, WhatsApp, calendar, reminders), it answers queries like:

- *What should I focus on today?*
- *What commitments am I at risk of missing?*
- *What have I been procrastinating on?*
- *Summarise everything related to the UIE proposal.*

For each query, it shows:
- The answer
- Which events were selected and why
- What was ignored and why
- What is uncertain

---

## Architecture (quick summary)

```
Event stream (JSON)
       │
       ▼
  retriever.py          ← Signal-based scoring: recency + keyword + urgency + stall
       │
       ▼
  memory_engine.py      ← Context construction, LLM call (Groq), reasoning explanation
       │
       ▼
  main.py               ← CLI runner, JSON output
```

See [`DESIGN.md`](DESIGN.md) for the full design document.  
See [`EVAL_FRAMEWORK.md`](EVAL_FRAMEWORK.md) for the evaluation framework.

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd memorae
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Two dependencies: `groq` (the official Groq Python SDK) and `python-dotenv` (loads the `.env` file automatically).

### 4. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com).

Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

This loads automatically every run — no need to `export` it manually each session. `.env` is already in `.gitignore` so it never gets committed.

### 5. Place the dataset

This repo expects the dataset at `data/memorae_mock_events.json`. If your file is elsewhere, just pass its path via `--data`.

---

## Running

### Run all four queries

```bash
python main.py --data data/memorae_mock_events.json
```

### Run a single query

```bash
python main.py --data data/memorae_mock_events.json --query uie_proposal
python main.py --data data/memorae_mock_events.json --query focus_today
python main.py --data data/memorae_mock_events.json --query commitments_at_risk
python main.py --data data/memorae_mock_events.json --query procrastination
```

### Save results to JSON

```bash
python main.py --data data/memorae_mock_events.json --output results.json
```

---

## Evaluation

### Offline evals (no LLM required, fast)

```bash
python eval.py --data data/memorae_mock_events.json --mode offline
```

### Rubric evals (LLM-as-judge, uses Groq)

```bash
python eval.py --data data/memorae_mock_events.json --mode rubric
```

### Regression tests (compare against a saved baseline)

First, save a baseline:
```bash
python main.py --data data/memorae_mock_events.json --output baseline.json
```

Then run regression:
```bash
python eval.py --data data/memorae_mock_events.json --mode regression --baseline baseline.json
```

---

## Output format

Each query returns:

```json
{
  "query": "What should I focus on today?",
  "answer": "...",
  "selected_context": [
    {
      "timestamp": "2026-04-12T09:00:00+00:00",
      "source": "email",
      "content": "...",
      "relevance_score": 0.847,
      "score_breakdown": {
        "recency": 0.93,
        "keyword": 0.8,
        "urgency": 0.3,
        "stall": 0.0,
        "source_weight": 1.3
      }
    }
  ],
  "context_stats": {
    "events_retrieved": 18,
    "events_in_context": 15,
    "total_events": 87,
    "estimated_tokens": 1842
  },
  "reasoning": {
    "why_selected": "...",
    "why_ignored": "...",
    "uncertainty": "..."
  }
}
```

---

## Hallucination guard (grounding check)

Every generated answer is scanned for numeric claims (dollar amounts, percentages, times, dates) that don't appear anywhere in the retrieved context. If the LLM states a number not traceable to a source event, it's flagged in the console as a warning and recorded in the output JSON under `grounding_check`.

This is a heuristic, not a guarantee — it catches the most damaging failure mode for a personal-memory assistant (invented prices, deadlines, percentages) but won't catch invented names or causal claims. During development this caught a real hallucinated licensing figure the model invented in an early run; the fix and the resulting regression tests are documented in `eval.py` and discussed under "Failure Modes" in `DESIGN.md`.

---

## API usage and cost

- **Model**: `llama-3.3-70b-versatile` via Groq
- **Calls per query**: 2 (answer generation + reasoning explanation, the latter using JSON mode for reliable parsing)
- **Tokens per query**: ~2,000–4,000 input + ~800 output
- **Estimated cost**: <$0.01 per query at Groq's pricing (as of April 2026)
- **Groq docs**: [console.groq.com/docs](https://console.groq.com/docs)

---

## Project structure

```
memorae/
├── main.py              # CLI entry point
├── retriever.py         # Signal-based event scoring and retrieval
├── memory_engine.py     # Pipeline: context construction, LLM calls, grounding check
├── eval.py              # Evaluation framework (offline, rubric, regression)
├── requirements.txt     # groq, python-dotenv
├── .env                 # GROQ_API_KEY (not committed — see .gitignore)
├── .gitignore
├── data/
│   └── memorae_mock_events.json
├── DESIGN.md            # Architecture and design decisions
├── EVAL_FRAMEWORK.md    # Detailed evaluation methodology
└── README.md            # This file
```

---

## Design decisions (brief)

- **No embeddings in v1**: Keyword + recency scoring is fast, interpretable, and sufficient for structured queries. Embeddings would be added as a first-stage ANN retriever for free-text queries at scale.
- **Topic gating for entity-specific queries**: For queries like "summarise the UIE proposal," recency and source trust alone let unrelated same-week calendar events crowd out genuinely relevant ones. Topic-specific profiles now require at least one topic-term match before an event is eligible at all — see `requires_topic_match` in `retriever.py`.
- **Low LLM temperature (0.2)**: Personal memory answers should be factual, not creative.
- **Structured JSON output for reasoning**: An earlier version parsed free-text labels (`WHY_SELECTED:`) out of the model's reasoning response. This silently broke when the model varied its formatting. Switched to Groq's `response_format: json_object` mode so the schema is enforced, not guessed at.
- **Explicit uncertainty**: The model is instructed to flag uncertainty rather than guess, and a post-hoc grounding check flags numeric claims not traceable to source events.
- **Context budget discipline**: We use ~6k tokens of a 100k budget — leaving room for conversation history in production.

See `DESIGN.md` for the full discussion.