#!/usr/bin/env python3
"""Run evaluation on synchronous LLM providers.

This script queries sync providers (like Ether0) in real-time and writes
results directly to CSV in the same format as batch provider outputs.

Usage:
    python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl
    python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl --limit 10 --workers 3
"""

import argparse
import json
import sys
from pathlib import Path

from providers import SYNC_PROVIDER_MAP


def load_questions(questions_files: list[str], chemiq_only: bool = True, limit: int | None = None) -> list:
    """Load questions from one or more JSONL files."""
    questions = []
    for questions_file in questions_files:
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    if not chemiq_only or q.get('ChemIQ', False):
                        questions.append(q)
                        if limit and len(questions) >= limit:
                            return questions
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on synchronous LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full Ether0 evaluation
  python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl

  # Run with limit for testing
  python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl --limit 5

  # Run with 3 parallel workers
  python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl --workers 3

  # Custom output path
  python run_sync_eval.py --provider ether0 --questions questions/chemiq.jsonl \\
    --output my_results.csv
        """
    )
    
    parser.add_argument(
        '--provider',
        choices=sorted(SYNC_PROVIDER_MAP.keys()),
        required=True,
        help='Sync provider to use'
    )
    parser.add_argument(
        '--questions',
        required=True,
        nargs='+',
        help='Path(s) to questions JSONL file(s)')
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path (default: model_responses/{provider}-responses.csv)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1 for sequential)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh instead of resuming'
    )
    parser.add_argument(
        '--all-questions',
        action='store_true',
        help='Include all questions, not just ChemIQ subset'
    )
    
    args = parser.parse_args()
    
    # Initialize provider
    provider_cls = SYNC_PROVIDER_MAP[args.provider]
    client = provider_cls()
    
    # Load questions
    print(f"Loading questions from {len(args.questions)} file(s): {', '.join(args.questions)}")
    questions = load_questions(
        args.questions,
        chemiq_only=not args.all_questions,
        limit=args.limit
    )
    print(f"Loaded {len(questions)} questions")
    
    # Determine output path
    output_csv = args.output
    if output_csv is None:
        output_csv = f"model_responses/{args.provider}-responses.csv"
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    client.run_evaluation(
        questions=questions,
        output_csv=output_csv,
        resume=not args.no_resume,
        max_workers=args.workers,
    )
    
    print(f"\nDone! Results saved to: {output_csv}")
    print(f"Run evaluation with: python evaluate.py --csv-dir model_responses --questions {' '.join(args.questions)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

