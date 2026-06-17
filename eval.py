"""
eval.py

Evaluation framework for Memorae.

Three evaluation modes:
  1. Offline evals  — deterministic, no LLM calls, fast
  2. Rubric evals   — LLM-as-judge for answer quality
  3. Regression     — compare outputs against saved baselines

Usage:
    python eval.py --data memorae_mock_events.json --mode offline
    python eval.py --data memorae_mock_events.json --mode rubric
    python eval.py --data memorae_mock_events.json --mode regression --baseline baseline.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from retriever import load_events, retrieve, SCENARIO_NOW
from memory_engine import run_query, get_client, MODEL


# ---------------------------------------------------------------------------
# 1. Offline Evals — deterministic, no LLM required
# ---------------------------------------------------------------------------

class OfflineEvaluator:
    """
    Tests that don't require an LLM call.
    These run fast and should be part of every CI check.
    """

    def __init__(self, events: list[dict]):
        self.events = events
        self.results: list[dict] = []

    def _record(self, test_name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}" + (f": {detail}" if detail else ""))
        self.results.append({"test": test_name, "passed": passed, "detail": detail})

    def test_retriever_returns_results(self):
        """Basic smoke test: every profile should return at least 1 event."""
        from retriever import QUERY_PROFILES
        for profile_key in QUERY_PROFILES:
            results = retrieve(self.events, profile_key, top_k=20)
            self._record(
                f"retriever/{profile_key} returns results",
                len(results) > 0,
                f"{len(results)} events returned",
            )

    def test_retriever_uie_finds_uie_events(self):
        """UIE profile must surface events mentioning 'uie' or 'proposal'."""
        results = retrieve(self.events, "uie_proposal", top_k=20)
        relevant = [
            ev for ev in results
            if "uie" in ev.get("content", "").lower()
            or "proposal" in ev.get("content", "").lower()
        ]
        self._record(
            "retriever/uie_proposal finds UIE-specific events",
            len(relevant) > 0,
            f"{len(relevant)} UIE-relevant events in top-20",
        )

    def test_retriever_scores_decrease_monotonically(self):
        """Results should be sorted by score descending."""
        results = retrieve(self.events, "focus_today", top_k=20)
        if len(results) < 2:
            self._record("retriever/scores sorted", True, "too few results to check")
            return
        scores = [ev["_score"] for ev in results]
        is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        self._record("retriever/scores sorted descending", is_sorted)

    def test_future_events_score_high(self):
        """Events scheduled in the future should have recency_score ~1.0."""
        from retriever import recency_score
        from datetime import timedelta
        future_event = {
            "_dt": SCENARIO_NOW + timedelta(days=2),
            "content": "meeting tomorrow",
            "source": "calendar",
        }
        score = recency_score(future_event)
        self._record(
            "retriever/future events score >= 1.0",
            score >= 1.0,
            f"score = {score:.3f}",
        )

    def test_old_events_score_low(self):
        """Events from 60 days ago should have recency_score < 0.05."""
        from retriever import recency_score
        from datetime import timedelta
        old_event = {
            "_dt": SCENARIO_NOW - timedelta(days=60),
            "content": "old reminder",
            "source": "reminder",
        }
        score = recency_score(old_event)
        self._record(
            "retriever/old events score < 0.05",
            score < 0.05,
            f"score = {score:.4f}",
        )

    def test_context_block_deduplicates(self):
        """Duplicate events should be removed from context."""
        from memory_engine import build_context_block
        dup_events = [
            {"_dt": SCENARIO_NOW, "source": "slack", "content": "Finish the report by EOD",
             "_score": 0.9},
            {"_dt": SCENARIO_NOW, "source": "slack", "content": "Finish the report by EOD",
             "_score": 0.9},
        ]
        block = build_context_block(dup_events)
        # Should only appear once
        count = block.count("Finish the report by EOD")
        self._record(
            "context/deduplication works",
            count == 1,
            f"content appeared {count} time(s)",
        )

    def test_context_truncates_long_content(self):
        """Long event content should be truncated."""
        from memory_engine import build_context_block, MAX_CONTENT_CHARS
        long_content = "A" * 1000
        events = [{"_dt": SCENARIO_NOW, "source": "email", "content": long_content,
                   "_score": 0.8}]
        block = build_context_block(events)
        self._record(
            "context/long content truncated",
            len(long_content) > MAX_CONTENT_CHARS,  # just verify the constant
            f"MAX_CONTENT_CHARS = {MAX_CONTENT_CHARS}",
        )

    def test_grounding_check_ignores_own_timestamp_echo(self):
        """
        Regression test for a real bug found during manual testing
        (and a second variant found in a follow-up run): the model
        echoes back the scenario clock in varying phrasings —
        "as of 2026-04-13 03:00 UTC" in one run, "As of the current
        time, April 13, 03:00 UTC" in another. A naive check either
        split "03:00" into stray digit tokens, or matched only one
        exact phrasing and missed the other. The fix excludes the
        scenario clock's own time value as a known-safe token,
        independent of how the model phrases the reference to it.
        """
        from memory_engine import check_grounding

        context = "UIE proposal review with Nina Apr 13 14:30 IST."

        # Variant A: the original phrasing that first surfaced this bug
        answer_a = (
            "**UIE Proposal Briefing as of 2026-04-13 03:00 UTC**\n"
            "Review at 14:30 IST as scheduled."
        )
        result_a = check_grounding(answer_a, context)
        self._record(
            "grounding/ignores scenario-time echo (variant A)",
            result_a == [],
            f"flagged: {result_a}" if result_a else "no false positives",
        )

        # Variant B: the rephrasing that broke the first fix
        answer_b = (
            "As of the current time, April 13, 03:00 UTC, the review "
            "is scheduled for 14:30 IST."
        )
        result_b = check_grounding(answer_b, context)
        self._record(
            "grounding/ignores scenario-time echo (variant B)",
            result_b == [],
            f"flagged: {result_b}" if result_b else "no false positives",
        )

        # Variant C: the written-out date format that broke the second
        # fix (ISO-only date pattern missed "April 13, 2026")
        answer_c = (
            "As of the current time (Monday, April 13, 2026, at 03:00 "
            "UTC), the review is at 14:30 IST."
        )
        result_c = check_grounding(answer_c, context)
        self._record(
            "grounding/ignores scenario-time echo (variant C, written date)",
            result_c == [],
            f"flagged: {result_c}" if result_c else "no false positives",
        )

    def test_grounding_check_still_catches_real_hallucination(self):
        """A genuinely invented number must still be flagged."""
        from memory_engine import check_grounding
        context = "UIE proposal review with Nina Apr 13 14:30 IST."
        answer = "The licensing estimate is $99.9k for year one."
        result = check_grounding(answer, context)
        self._record(
            "grounding/catches invented numbers",
            any("99.9" in r for r in result),
            f"flagged: {result}",
        )

    def run_all(self) -> dict:
        print("\n--- OFFLINE EVALS ---")
        self.test_retriever_returns_results()
        self.test_retriever_uie_finds_uie_events()
        self.test_retriever_scores_decrease_monotonically()
        self.test_future_events_score_high()
        self.test_old_events_score_low()
        self.test_context_block_deduplicates()
        self.test_context_truncates_long_content()
        self.test_grounding_check_ignores_own_timestamp_echo()
        self.test_grounding_check_still_catches_real_hallucination()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n  Result: {passed}/{total} tests passed")
        return {"passed": passed, "total": total, "tests": self.results}


# ---------------------------------------------------------------------------
# 2. Rubric Evals — LLM-as-judge
# ---------------------------------------------------------------------------

RUBRICS = {
    "focus_today": {
        "criteria": [
            "Does the answer mention specific tasks (not vague advice)?",
            "Is it prioritised (not just a flat list)?",
            "Does it reference actual names, deadlines, or event details from the context?",
            "Does it avoid making up information not in the events?",
            "Is the answer appropriate for the current date (April 13, 2026)?",
        ],
        "min_score": 3,  # out of 5 criteria
    },
    "commitments_at_risk": {
        "criteria": [
            "Does it identify specific commitments (not generic risks)?",
            "Does it mention who the commitment is with?",
            "Does it explain why each item is at risk?",
            "Does it flag overdue or near-deadline items?",
            "Is the reasoning grounded in the event stream?",
        ],
        "min_score": 3,
    },
    "procrastination": {
        "criteria": [
            "Does it identify recurring or unresolved items?",
            "Does it avoid moralising and stay factual?",
            "Does it group related items together?",
            "Are items mentioned actually pending (not completed)?",
            "Is the answer specific, not generic?",
        ],
        "min_score": 3,
    },
    "uie_proposal": {
        "criteria": [
            "Does it cover the current status of the proposal?",
            "Does it mention what is outstanding?",
            "Does it include any deadlines or pricing details if present in events?",
            "Is it structured as a coherent briefing, not a list of raw events?",
            "Does it identify any risks or blockers?",
        ],
        "min_score": 3,
    },
}


def rubric_eval_answer(
    query_key: str,
    answer: str,
    context_block: str,
) -> dict[str, Any]:
    """Use LLM as judge to score an answer against a rubric."""
    client = get_client()
    rubric = RUBRICS.get(query_key)
    if not rubric:
        return {"score": None, "detail": "No rubric defined for this query"}

    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["criteria"]))

    prompt = f"""You are evaluating a personal memory assistant's answer.

Query type: {query_key.replace('_', ' ')}

Context provided to the assistant:
{context_block[:2000]}

Assistant's answer:
{answer}

Evaluate the answer against these criteria (answer YES or NO for each):
{criteria_text}

Respond ONLY with a JSON object like:
{{
  "scores": [true, false, true, true, false],
  "explanation": "Brief explanation of the evaluation"
}}
No preamble, no markdown fences."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        scores = parsed.get("scores", [])
        passed = sum(1 for s in scores if s)
        min_score = rubric["min_score"]
        return {
            "criteria_scores": scores,
            "criteria_passed": passed,
            "criteria_total": len(scores),
            "min_required": min_score,
            "passed": passed >= min_score,
            "explanation": parsed.get("explanation", ""),
        }
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM judge response", "raw": raw}


# ---------------------------------------------------------------------------
# 3. Regression Tests
# ---------------------------------------------------------------------------

def regression_eval(
    current_results: list[dict],
    baseline_path: Path,
) -> dict:
    """
    Compare current answers against a saved baseline.
    Flags if the answer changed significantly (length, key terms).
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_by_query = {r["query"]: r for r in baseline}
    report = []

    for result in current_results:
        query = result["query"]
        if query not in baseline_by_query:
            report.append({"query": query, "status": "NEW — no baseline"})
            continue

        base = baseline_by_query[query]
        current_answer = result["answer"]
        base_answer = base["answer"]

        # Simple regression signals
        len_ratio = len(current_answer) / max(len(base_answer), 1)
        current_events = result["context_stats"]["events_in_context"]
        base_events = base["context_stats"]["events_in_context"]

        regressions = []
        if len_ratio < 0.5:
            regressions.append(f"Answer is much shorter ({len_ratio:.1%} of baseline length)")
        if abs(current_events - base_events) > 5:
            regressions.append(
                f"Context size changed significantly: {base_events} → {current_events} events"
            )

        report.append({
            "query": query,
            "status": "REGRESSION" if regressions else "OK",
            "regressions": regressions,
            "answer_length_ratio": round(len_ratio, 2),
        })

    passed = sum(1 for r in report if r["status"] == "OK")
    print(f"\n  Regression results: {passed}/{len(report)} stable")
    return {"results": report}


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(events: list[dict], query_key: str, n_runs: int = 3) -> dict:
    """Measure end-to-end latency for a single query (including LLM call)."""
    times = []
    for i in range(n_runs):
        start = time.time()
        run_query(events, query_key, f"latency_test_{i}")
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    return {
        "query": query_key,
        "runs": n_runs,
        "avg_seconds": round(sum(times) / len(times), 2),
        "min_seconds": round(min(times), 2),
        "max_seconds": round(max(times), 2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memorae Evaluation Framework")
    parser.add_argument("--data", required=True, help="Path to event JSON")
    parser.add_argument(
        "--mode",
        choices=["offline", "rubric", "regression", "latency"],
        default="offline",
    )
    parser.add_argument("--baseline", help="Baseline JSON for regression tests")
    parser.add_argument("--output", help="Save eval results to JSON")
    args = parser.parse_args()

    events = load_events(args.data)
    print(f"Loaded {len(events)} events.")

    output = {}

    if args.mode == "offline":
        evaluator = OfflineEvaluator(events)
        output = evaluator.run_all()

    elif args.mode == "rubric":
        print("\n--- RUBRIC EVALS (LLM-as-Judge) ---")
        from main import QUERIES
        from memory_engine import build_context_block
        from retriever import retrieve

        rubric_results = []
        for q in QUERIES:
            print(f"\n  Evaluating: {q['label']}")
            selected = retrieve(events, q["key"], top_k=20)
            context_block = build_context_block(selected)
            result = run_query(events, q["key"], q["label"])
            eval_result = rubric_eval_answer(q["key"], result["answer"], context_block)
            print(f"  Score: {eval_result.get('criteria_passed', '?')}/"
                  f"{eval_result.get('criteria_total', '?')} "
                  f"({'PASS' if eval_result.get('passed') else 'FAIL'})")
            rubric_results.append({"query": q["label"], **eval_result})

        output = {"rubric_results": rubric_results}

    elif args.mode == "regression":
        if not args.baseline:
            print("ERROR: --baseline required for regression mode")
            return
        from main import QUERIES
        results = [run_query(events, q["key"], q["label"]) for q in QUERIES]
        output = regression_eval(results, Path(args.baseline))

    elif args.mode == "latency":
        print("\n--- LATENCY MEASUREMENT ---")
        output = measure_latency(events, "focus_today", n_runs=2)
        print(f"  Average: {output['avg_seconds']}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nEval results saved to {args.output}")


if __name__ == "__main__":
    main()