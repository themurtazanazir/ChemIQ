#!/usr/bin/env python3
"""Download batch results from LLM APIs and convert to CSV."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from convert_jsonl_to_csv import convert_results_to_csv
from providers import (
    GoogleProviderClient,
    OpenAIProviderClient,
    AnthropicProviderClient,
    PROVIDER_MAP,
)


def read_batch_id(batch_id_file: str) -> str:
    """Read batch ID from a text file.
    
    Args:
        batch_id_file: Path to batch ID file
        
    Returns:
        Batch ID string
    """
    with open(batch_id_file, 'r') as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(
        description="Download batch results and convert to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Google batch using batch ID file
  python download_results.py --batch-id-file batch_ids/gemini-3-flash-2025-12-27.txt \\
    --provider google \\
    --api-key $GOOGLE_API_KEY
  
  # Download OpenAI batch
  python download_results.py --batch-id-file batch_ids/o3-mini-2025-12-27.txt \\
    --provider openai \\
    --api-key $OPENAI_API_KEY
  
  # Download Anthropic batch
  python download_results.py --batch-id-file batch_ids/claude-3-5-sonnet-2025-12-27.txt \\
    --provider anthropic \\
    --api-key $ANTHROPIC_API_KEY
  
  # Download using batch ID directly
  python download_results.py --batch-id batches/abc123 \\
    --provider openai \\
    --api-key $OPENAI_API_KEY
  
  # Force re-download
  python download_results.py --batch-id-file batch_ids/gemini-3-flash-2025-12-27.txt \\
    --provider google \\
    --api-key $GOOGLE_API_KEY --force
        """
    )
    
    parser.add_argument(
        '--batch-id-file',
        help='Path to batch ID file (alternative to --batch-id)'
    )
    parser.add_argument(
        '--batch-id',
        help='Batch ID directly (alternative to --batch-id-file)'
    )
    parser.add_argument(
        '--provider',
        choices=sorted(PROVIDER_MAP.keys()),
        default='google',
        help='LLM provider (default: google)'
    )
    parser.add_argument(
        '--api-key',
        help='API key (or set GOOGLE_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY env var)',
        default=None
    )
    parser.add_argument(
        '--env-file',
        help='Path to environment file containing API keys (dotenv format)',
        default=None
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory for JSONL results (default: results)'
    )
    parser.add_argument(
        '--csv-dir',
        default='model_responses',
        help='Directory for CSV output (default: model_responses)'
    )
    parser.add_argument(
        '--submissions-dir',
        default='submissions',
        help='Directory for submission files (default: submissions)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if files exist'
    )
    parser.add_argument(
        '--model-name',
        help='Override model name in CSV (default: extract from results)'
    )
    parser.add_argument(
        '--output-name',
        help='Custom output file name prefix (overrides inferred name from batch ID)'
    )
    parser.add_argument(
        '--submission-file',
        help='Path to submission JSONL file (for accurate thinking_budget values)'
    )
    
    args = parser.parse_args()
    

    if args.env_file:
        if not os.path.exists(args.env_file):
            print(f"Error: Environment file not found: {args.env_file}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(args.env_file)
        print(f"Loaded environment from {args.env_file}")
    
    # Get API key (validated later with batch_id)
    provider_cls = PROVIDER_MAP[args.provider]
    
    # Get batch ID
    if args.batch_id_file:
        batch_id = read_batch_id(args.batch_id_file)
        # Try to infer model name, timestamp, and thinking_budget from filename
        batch_id_file_stem = Path(args.batch_id_file).stem
        parts = batch_id_file_stem.split('-')
        
        # Check for thinking_budget pattern
        # New format: model-tb8192 (no timestamp)
        # Old format: model-tb8192-timestamp
        inferred_thinking_budget = 0
        inferred_model = None
        inferred_timestamp = None
        
        # Find the tb* part if present
        tb_idx = None
        for i, part in enumerate(parts):
            if part.startswith('tb') and len(part) > 2:
                try:
                    # Try int first
                    inferred_thinking_budget = int(part[2:])
                except ValueError:
                    # If not int, treat as string level (e.g. tbminimal)
                    inferred_thinking_budget = part[2:]
                tb_idx = i
                break
        
        if tb_idx is not None:
            # Model is everything before tb*
            inferred_model = '-'.join(parts[:tb_idx])
        else:
            inferred_model = batch_id_file_stem
        
    elif args.batch_id:
        batch_id = args.batch_id
        inferred_model = None
        inferred_model = None
        inferred_thinking_budget = 0
    else:
        print("Error: Must provide either --batch-id-file or --batch-id", file=sys.stderr)
        sys.exit(1)
    
    print(f"Batch ID: {batch_id}")
    
    # Validate and Create
    client = provider_cls.validate_download(args.api_key, batch_id)
    
    # Try to fetch actual metadata from the batch job (source of truth)
    try:
        print(f"Checking batch status for metadata...")
        status_obj = client.check_batch_status(batch_id)
        if status_obj.metadata and 'thinking_budget' in status_obj.metadata:
            fetched_budget = status_obj.metadata['thinking_budget']
            print(f"Found thinking_budget in batch metadata: {fetched_budget}")
            
            # Prioritize metadata over filename inference
            # Handle int vs string
            try:
                thinking_budget = int(fetched_budget)
            except ValueError:
                thinking_budget = fetched_budget
                
            inferred_thinking_budget = thinking_budget
    except Exception as e:
        print(f"Could not fetch batch metadata (using inferred values): {e}")
    
    # Determine model name and budget
    model_name = inferred_model or 'unknown-model'
    thinking_budget = inferred_thinking_budget
    
    # Use --output-name if provided, otherwise use inferred model name
    if args.output_name:
        safe_output_name = args.output_name.replace("/", "-").replace("_", "-")
    else:
        safe_output_name = model_name.replace("/", "-").replace("_", "-")
        # Include thinking_budget in filenames if specified (no dates in filenames)
        has_budget = False
        if isinstance(thinking_budget, int) and thinking_budget > 0:
            has_budget = True
        elif isinstance(thinking_budget, str) and thinking_budget and thinking_budget != '0':
            has_budget = True
        if has_budget:
            safe_output_name = f"{safe_output_name}-tb{thinking_budget}"
    
    results_jsonl = Path(args.results_dir) / f"{safe_output_name}-results.jsonl"
    submission_jsonl = Path(args.submissions_dir) / f"{safe_output_name}-submission.jsonl"
    output_csv = Path(args.csv_dir) / f"{safe_output_name}-responses.csv"
    
    # Download results using provider client
    success = client.download_results(
        batch_id=batch_id,
        output_jsonl=str(results_jsonl),
        force=args.force
    )
    
    if not success:
        sys.exit(1)
    
    # Convert to CSV
    # Use explicit submission file if provided, otherwise use inferred path
    actual_submission_file = args.submission_file if args.submission_file else str(submission_jsonl)
    if not Path(actual_submission_file).exists():
        print(f"Warning: Submission file not found: {actual_submission_file}")
        print("CSV conversion may not have accurate thinking_budget values")
        actual_submission_file = str(results_jsonl)  # Fallback
    
    print(f"\nConverting to CSV...")
    try:
        success = convert_results_to_csv(
            results_jsonl_path=str(results_jsonl),
            submission_jsonl_path=actual_submission_file,
            output_csv_path=str(output_csv),
            provider_cls=provider_cls,
            model_name=args.model_name
        )
        print(f"CSV saved to: {output_csv}")
    except Exception as e:
        print(f"Error converting to CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
