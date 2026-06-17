"""
main.py

CLI entry point for the Memorae personal memory engine.

Usage:
    python main.py --data memorae_mock_events.json
    python main.py --data memorae_mock_events.json --query uie_proposal
    python main.py --data memorae_mock_events.json --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

from retriever import load_events
from memory_engine import run_query

# ---------------------------------------------------------------------------
# Query registry — add more here to extend the system
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "key": "focus_today",
        "label": "What should I focus on today?",
    },
    {
        "key": "commitments_at_risk",
        "label": "What commitments am I at risk of missing?",
    },
    {
        "key": "procrastination",
        "label": "What have I been procrastinating on?",
    },
    {
        "key": "uie_proposal",
        "label": "Summarise everything related to the UIE proposal.",
    },
]


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    print(f"\n{'#'*60}")
    print(f"  QUERY: {result['query']}")
    print(f"{'#'*60}\n")

    print("ANSWER")
    print("-" * 40)
    print(result["answer"])

    print("\nCONTEXT STATS")
    print("-" * 40)
    stats = result["context_stats"]
    print(f"  Events retrieved : {stats['events_retrieved']}")
    print(f"  Events in context: {stats['events_in_context']}")
    print(f"  Total in dataset : {stats['total_events']}")
    print(f"  Est. tokens used : ~{stats['estimated_tokens']}")

    print("\nREASONING")
    print("-" * 40)
    r = result["reasoning"]
    print(f"  Why selected:\n    {r.get('why_selected', 'N/A')}")
    print(f"\n  Why ignored:\n    {r.get('why_ignored', 'N/A')}")
    print(f"\n  Uncertainty:\n    {r.get('uncertainty', 'N/A')}")

    print("\nTOP SELECTED EVENTS")
    print("-" * 40)
    for i, ev in enumerate(result["selected_context"][:5], 1):
        print(f"  {i}. [{ev['timestamp'][:10]}] {ev['source'].upper()} "
              f"(score: {ev['relevance_score']:.3f})")
        print(f"     {ev['content'][:120]}…")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Memorae — Personal Memory Query Engine"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to memorae_mock_events.json",
    )
    parser.add_argument(
        "--query",
        choices=[q["key"] for q in QUERIES] + ["all"],
        default="all",
        help="Which query to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results",
    )
    args = parser.parse_args()

    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading events from {data_path}…")
    events = load_events(data_path)
    print(f"Loaded {len(events)} events.")

    # Select queries to run
    if args.query == "all":
        queries_to_run = QUERIES
    else:
        queries_to_run = [q for q in QUERIES if q["key"] == args.query]

    # Run pipeline
    results = []
    for q in queries_to_run:
        result = run_query(
            events=events,
            profile_key=q["key"],
            display_name=q["label"],
        )
        print_result(result)
        results.append(result)

    # Save output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            # Convert datetime objects to strings for serialisation
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()