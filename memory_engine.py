"""
memory_engine.py

Orchestrates the four-step pipeline for each query:
  1. Source & Signal Selection  — retriever.py handles this
  2. Context Construction       — budget-aware, deduped, time-sorted
  3. Answer Generation          — Groq LLM call
  4. Reasoning Explanation      — separate structured LLM call

Context budget strategy:
  - We target ~6,000 tokens per query (well under a 100k budget)
  - This leaves room for conversation history in production
  - Events are trimmed to their most informative ~300 chars before packing
"""

import json
import os
import re
import textwrap
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from retriever import SCENARIO_NOW, retrieve

load_dotenv()  # reads .env in the project root if present

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "llama-3.3-70b-versatile"   # fast, capable, cheap on Groq
MAX_CONTEXT_EVENTS = 15             # hard cap on events sent to LLM
MAX_CONTENT_CHARS = 400             # truncate long event content
GROQ_CLIENT = None                  # initialised lazily


def get_client() -> Groq:
    global GROQ_CLIENT
    if GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Export it before running:\n"
                "  export GROQ_API_KEY=gsk_..."
            )
        GROQ_CLIENT = Groq(api_key=api_key)
    return GROQ_CLIENT


# ---------------------------------------------------------------------------
# Step 2: Context Construction
# ---------------------------------------------------------------------------

def build_context_block(events: list[dict], max_events: int = MAX_CONTEXT_EVENTS) -> str:
    """
    Convert retrieved events into a compact, LLM-readable context block.

    Design decisions:
      - Sort chronologically so the LLM sees the narrative arc
      - Truncate long content (300 chars) — enough signal, less noise
      - Deduplicate by content hash to handle duplicate messages
      - Label each event with source and timestamp for traceability
    """
    seen_content = set()
    unique_events = []

    for ev in events:
        # Deduplicate by first 120 chars of content
        fingerprint = ev.get("content", "")[:120].strip().lower()
        if fingerprint and fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        unique_events.append(ev)

    # Sort chronologically
    unique_events.sort(key=lambda x: x["_dt"])

    # Limit to budget
    unique_events = unique_events[:max_events]

    lines = []
    for i, ev in enumerate(unique_events, 1):
        ts = ev["_dt"].strftime("%Y-%m-%d %H:%M UTC")
        source = ev.get("source", "unknown").upper()
        content = ev.get("content", "").strip()

        # Truncate long content but preserve sentence boundaries where possible
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS].rsplit(" ", 1)[0] + "…"

        lines.append(f"[{i}] {ts} | {source}\n    {content}")

    return "\n\n".join(lines)


def format_scenario_context() -> str:
    return (
        f"Current time: {SCENARIO_NOW.strftime('%A, %B %d %Y at %H:%M UTC')}\n"
        "You are a personal memory assistant. Answer based only on the events provided.\n\n"
        "STRICT GROUNDING RULE: every specific fact, number, date, name, or dollar "
        "amount in your answer must come directly from the event stream below. "
        "Do not infer, estimate, or invent numbers (e.g. prices, percentages, "
        "durations) that are not explicitly stated in an event. If a detail "
        "the user might expect (such as a price, owner, or exact figure) is "
        "not present in the events, explicitly say it is not mentioned in the "
        "available events rather than filling it in.\n\n"
        "Be specific and time-aware about what IS present. If something is "
        "uncertain or missing, say so explicitly rather than guessing."
    )


# ---------------------------------------------------------------------------
# Step 3: Answer Generation
# ---------------------------------------------------------------------------

QUERY_PROMPTS = {
    "focus_today": textwrap.dedent("""
        Based on the events below, what should the user focus on today ({date})?
        
        Prioritise:
        - Items with explicit deadlines or due dates falling today or soon
        - Commitments made to other people
        - Meetings or calls scheduled today
        - High-urgency items flagged in recent messages
        
        Be specific. Reference actual tasks, names, and deadlines from the events.
        Format your answer as a prioritised list with brief context for each item.
    """).strip(),

    "commitments_at_risk": textwrap.dedent("""
        Based on the events below, what commitments is the user at risk of missing?
        
        Look for:
        - Promised deliverables with deadlines that are approaching or past
        - Items people are waiting on from the user
        - Follow-ups that haven't happened yet
        - Anything marked urgent that hasn't been resolved
        
        For each risk, note: what the commitment is, who it's with, and why it's at risk.
    """).strip(),

    "procrastination": textwrap.dedent("""
        Based on the events below, what has the user been procrastinating on?
        
        Look for:
        - Tasks mentioned multiple times without resolution
        - Items described as pending, delayed, or forgotten
        - Things the user said they "need to" or "should" do but hasn't
        - Patterns of repeated reminders or follow-ups on the same thing
        
        Be honest and specific. Group related items together.
    """).strip(),

    "uie_proposal": textwrap.dedent("""
        Summarise everything related to the UIE proposal based on the events below.
        
        Cover:
        - What the proposal is for and who the client is
        - Current status and what has been done
        - What is still outstanding or pending
        - Any deadlines, pricing, or scope details mentioned
        - Any risks or blockers
        
        Present this as a concise briefing, chronologically where relevant.
    """).strip(),
}


def generate_answer(
    query_key: str,
    context_block: str,
    free_query: str = "",
) -> str:
    """Call Groq to generate an answer grounded in the context block."""
    client = get_client()

    date_str = SCENARIO_NOW.strftime("%B %d, %Y")
    prompt_template = QUERY_PROMPTS.get(query_key, "Answer this query: " + free_query)
    prompt = prompt_template.format(date=date_str)

    system_msg = format_scenario_context()
    user_msg = f"{prompt}\n\n---\nEVENT STREAM:\n\n{context_block}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,   # low temp = more factual, less hallucination
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


def check_grounding(answer: str, context_block: str) -> list[str]:
    """
    Lightweight hallucination guard: flag numeric tokens in the answer
    (dollar amounts, percentages, durations) that don't appear anywhere
    in the source context block.

    This is a heuristic, not a guarantee — it catches invented numbers
    (the most common and most damaging hallucination type for a
    commitments/deadlines assistant) but won't catch invented names or
    invented causal claims. Treat this as a tripwire for manual review,
    not a correctness proof.

    Known false-positive source we explicitly guard against:
      The model is told the scenario's "current time" in the system
      prompt and restates it in varying formats across calls: ISO
      ("2026-04-13"), written ("April 13, 2026"), with or without a
      leading "as of", etc. Three separate phrasing-specific patches
      were tried and each broke on a new variant. Pattern-matching the
      *sentence* is fundamentally brittle because there are unbounded
      ways to phrase the same fact in English.

      The robust fix: stop trying to recognize phrasings, and instead
      exclude every individual numeric component of the scenario
      clock (year, zero-padded and bare month, zero-padded and bare
      day, hour, minute, and the combined date/time strings) as
      known-safe tokens. Any number that equals one of these values is
      definitionally not a hallucination — it's metadata we gave the
      model ourselves — regardless of which format or sentence it
      appears in.
    """
    # Every numeric fragment the scenario clock could be decomposed
    # into, across any phrasing the model might choose. This list is
    # intentionally generous: false negatives here (treating a real
    # event's number as "safe" because it happens to equal e.g. "13")
    # are an acceptable tradeoff against the repeated false-positive
    # whack-a-mole of pattern-matching sentences.
    known_safe_tokens = {
        SCENARIO_NOW.strftime("%Y-%m-%d"),   # 2026-04-13
        SCENARIO_NOW.strftime("%H:%M"),       # 03:00
        SCENARIO_NOW.strftime("%Y"),          # 2026
        SCENARIO_NOW.strftime("%m"),          # 04
        str(SCENARIO_NOW.month),              # 4
        SCENARIO_NOW.strftime("%d"),          # 13
        str(SCENARIO_NOW.day),                # 13 (no leading zero, same value here)
        SCENARIO_NOW.strftime("%H"),          # 03
        str(SCENARIO_NOW.hour),               # 3
        SCENARIO_NOW.strftime("%M"),          # 00
    }

    # Match ISO dates (2026-04-13) and HH:MM times as single atomic
    # units BEFORE the generic number pattern runs, so neither gets
    # fragmented into stray digit pieces ("2026", "04", "13" from a
    # date; "14", "30" from a time). Order matters: most specific
    # patterns first.
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    time_pattern = r"\d{1,2}:\d{2}(?::\d{2})?"
    number_pattern = r"\$?\d+(?:\.\d+)?[kKmM%]?"

    def extract_tokens(text: str) -> set[str]:
        dates = set(re.findall(date_pattern, text))
        text_without_dates = re.sub(date_pattern, " ", text)

        times = set(re.findall(time_pattern, text_without_dates))
        text_without_times = re.sub(time_pattern, " ", text_without_dates)

        numbers = set(re.findall(number_pattern, text_without_times))
        return dates | times | numbers

    answer_tokens = extract_tokens(answer)
    context_tokens = extract_tokens(context_block)

    unverified = sorted(answer_tokens - context_tokens - known_safe_tokens)

    # Filter out trivial/common numbers that aren't really "claims"
    # (single digits, list markers, etc.)
    unverified = [n for n in unverified if len(n.lstrip("$")) > 1]

    return unverified


# ---------------------------------------------------------------------------
# Step 4: Reasoning Explanation
# ---------------------------------------------------------------------------

def generate_reasoning(
    query_key: str,
    selected_events: list[dict],
    all_event_count: int,
    answer: str,
) -> dict[str, str]:
    """
    Ask the LLM to explain its context selection and reasoning.
    Returns a structured dict for the output JSON.

    Uses Groq's JSON mode (response_format) instead of free-text label
    parsing. Free-text parsing (looking for lines starting with
    "WHY_SELECTED:") is fragile — LLM output formatting drifts across
    calls (different casing, markdown bold, numbered headers), and a
    parser mismatch fails silently, returning empty strings with no
    error. JSON mode forces the model to return a schema we control.
    """
    client = get_client()

    # Build a compact summary of selected events for the reasoning prompt
    selected_summary = "\n".join(
        f"- [{ev['_dt'].strftime('%Y-%m-%d')}] {ev.get('source','?').upper()}: "
        f"{ev.get('content','')[:100]}… (score: {ev.get('_score', 0):.3f})"
        for ev in selected_events[:MAX_CONTEXT_EVENTS]
    )

    prompt = textwrap.dedent(f"""
        A personal memory assistant was asked: "{query_key.replace('_', ' ')}"

        It selected {len(selected_events)} events from a stream of {all_event_count} total events.

        Selected events (with relevance scores):
        {selected_summary}

        The answer it generated was:
        {answer[:500]}

        Explain the assistant's behaviour. Respond with ONLY a JSON object,
        no markdown fences, no preamble, matching exactly this schema:

        {{
          "why_selected": "2-3 sentences on why these specific events were chosen and what signals made them relevant",
          "why_ignored": "2-3 sentences on what types of events were likely ignored and why",
          "uncertainty": "2-3 sentences on what is uncertain or ambiguous in the answer, and what information is missing"
        }}
    """).strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        return {
            "why_selected": parsed.get("why_selected", ""),
            "why_ignored": parsed.get("why_ignored", ""),
            "uncertainty": parsed.get("uncertainty", ""),
        }
    except json.JSONDecodeError:
        # Surface the failure instead of silently returning empty strings.
        # In production this would be logged to monitoring; here we keep
        # the raw text so a developer can see exactly what went wrong.
        return {
            "why_selected": "",
            "why_ignored": "",
            "uncertainty": "",
            "_parse_error": True,
            "_raw_response": raw,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_query(
    events: list[dict],
    profile_key: str,
    display_name: str,
    free_query: str = "",
) -> dict[str, Any]:
    """
    Run the full four-step pipeline for a single query.
    Returns a structured result dict.
    """
    print(f"\n{'='*60}")
    print(f"  Query: {display_name}")
    print(f"{'='*60}")

    # Step 1: Signal Selection
    print("  [1/4] Retrieving relevant events…")
    selected = retrieve(events, profile_key, free_query=free_query, top_k=20)
    print(f"        → {len(selected)} events selected from {len(events)} total")

    # Step 2: Context Construction
    print("  [2/4] Building context block…")
    context_block = build_context_block(selected)
    token_estimate = len(context_block.split()) * 1.3  # rough estimate
    print(f"        → ~{int(token_estimate)} tokens in context")

    # Step 3: Answer Generation
    print("  [3/4] Generating answer…")
    answer = generate_answer(profile_key, context_block, free_query)

    # Grounding check: flag any numeric claims not traceable to context.
    # This is a heuristic tripwire, not a guarantee — see check_grounding()
    # docstring. Surfaced in output so a reviewer can manually verify.
    unverified_numbers = check_grounding(answer, context_block)
    if unverified_numbers:
        print(f"        ⚠ WARNING: unverified numeric claims in answer: "
              f"{unverified_numbers}")

    # Step 4: Reasoning Explanation
    print("  [4/4] Generating reasoning explanation…")
    reasoning = generate_reasoning(profile_key, selected, len(events), answer)

    # Build the selected_context list for output (clean, no internal fields)
    selected_context = [
        {
            "timestamp": ev["_dt"].isoformat(),
            "source": ev.get("source", "unknown"),
            "content": ev.get("content", ""),
            "relevance_score": ev.get("_score", 0),
            "score_breakdown": ev.get("_score_breakdown", {}),
        }
        for ev in selected[:MAX_CONTEXT_EVENTS]
    ]

    return {
        "query": display_name,
        "answer": answer,
        "selected_context": selected_context,
        "context_stats": {
            "events_retrieved": len(selected),
            "events_in_context": len(selected_context),
            "total_events": len(events),
            "estimated_tokens": int(token_estimate),
        },
        "reasoning": reasoning,
        "grounding_check": {
            "unverified_numeric_claims": unverified_numbers,
            "note": (
                "Numbers in the answer not found in the selected context. "
                "Heuristic check, not a guarantee — review manually."
            ) if unverified_numbers else "No unverified numeric claims detected.",
        },
    }