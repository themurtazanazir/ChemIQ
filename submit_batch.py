#!/usr/bin/env python3
"""Submit batch jobs to LLM APIs."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from providers import (
    GoogleProviderClient,
    GoogleThinkingLevel,
    OpenAIProviderClient,
    OpenAIThinkingLevel,
    AnthropicProviderClient,
    PROVIDER_MAP,
)


def load_questions(questions_files: list[str], chemiq_only: bool = True) -> list:
    """Load questions from one or more JSONL files."""
    questions = []
    for questions_file in questions_files:
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    if not chemiq_only or q.get('ChemIQ', False):
                        questions.append(q)
    return questions


def parse_thinking_arg(provider: str, model: str, thinking_str: str):
    """Parse thinking argument and return appropriate enum or int.
    
    Args:
        provider: 'openai', 'google', or 'anthropic'
        model: Model identifier
        thinking_str: Raw thinking budget string from CLI
        
    Returns:
        ThinkingLevel enum, int budget, or None
        
    Raises:
        ValueError: If the value is invalid for the provider/model
    """
    if not thinking_str or thinking_str == '0':
        return None
    
    provider_cls = PROVIDER_MAP.get(provider)
    if not provider_cls:
        raise ValueError(f"Unknown provider: {provider}")
        
    if model not in provider_cls.SUPPORTED_MODELS:
        valid_models = sorted(provider_cls.SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Model '{model}' is not supported by {provider}. "
            f"Supported models: {', '.join(valid_models)}"
        )
        
    config = provider_cls.SUPPORTED_MODELS[model]
    thinking_mode = config.thinking_mode
    
    if thinking_mode == "budget":
        return int(thinking_str)
        
    if isinstance(thinking_mode, type):
        # It's an enum class
        return thinking_mode(thinking_str)
    
    # thinking_mode is None (no thinking supported)
    raise ValueError(f"Model '{model}' does not support thinking parameters.")

def main():
    parser = argparse.ArgumentParser(
        description="Submit batch jobs to LLM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit Google Gemini 3 with thinking level
  python submit_batch.py --model gemini-3-flash-preview \\
    --questions questions/chemiq.jsonl \\
    --provider google --env-file .env \\
    --thinking medium
  
  # Submit OpenAI o3-mini with reasoning effort
  python submit_batch.py --model o3-mini-2025-01-31 \\
    --questions questions/chemiq.jsonl \\
    --provider openai --env-file .env \\
    --thinking high
  
  # Submit Google Gemini 2.x with integer budget
  python submit_batch.py --model gemini-2.0-flash-thinking-exp \\
    --questions questions/chemiq.jsonl \\
    --provider google --env-file .env \\
    --thinking 8192

Thinking Parameter Notes:
  - OpenAI (o1, o3-mini): low, medium, high
  - Google Gemini 3.x: minimal, low, medium, high
  - Google Gemini 2.x: integer token budget (e.g., 1024, 8192)
  - Anthropic (Claude 3.7+, 4.x): integer token budget (e.g., 1024, 32768)
        """
    )
    
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--questions', required=True, nargs='+', help='Path(s) to questions JSONL file(s)')
    parser.add_argument(
        '--provider',
        choices=sorted(PROVIDER_MAP.keys()),
        default='google',
        help='LLM provider (default: google)'
    )
    parser.add_argument('--api-key', help='API key (or use env var)', default=None)
    parser.add_argument('--env-file', help='Path to .env file', default=None)
    parser.add_argument(
        '--thinking',
        type=str,
        default=None,
        help='Thinking level/budget (level string for OpenAI/Gemini3, int for Gemini2/Anthropic)'
    )
    parser.add_argument('--output-dir', default='submissions', help='Output directory')
    parser.add_argument('--output-name', help='Custom output file name prefix (overrides auto-generated name)')
    parser.add_argument('--completion-window', default='24h', help='Completion window')
    parser.add_argument('--all-questions', action='store_true', help='Include all questions')
    
    args = parser.parse_args()

    # Load env file
    if args.env_file:
        if not os.path.exists(args.env_file):
            print(f"Error: Environment file not found: {args.env_file}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(args.env_file)
        print(f"Loaded environment from {args.env_file}")

    # Parse thinking argument
    thinking = parse_thinking_arg(args.provider, args.model, args.thinking or '0')
    
    # Get API key & Validate Inputs
    provider_cls = PROVIDER_MAP[args.provider]
    
    # Validation and Client Creation
    client = provider_cls.validate_submission(args.api_key, args.model, thinking)
    
    # Load questions
    print(f"Loading questions from {len(args.questions)} file(s): {', '.join(args.questions)}")
    questions = load_questions(args.questions, chemiq_only=not args.all_questions)
    print(f"Loaded {len(questions)} questions")
    
    # Generate submission file path
    if args.output_name:
        output_name = args.output_name
    else:
        safe_model_name = args.model.replace("/", "-").replace("_", "-")
        thinking_suffix = ""
        if thinking is not None:
            if hasattr(thinking, 'name'):
                thinking_suffix = f"-tb{thinking.name.lower()}"
            else:
                thinking_suffix = f"-tb{thinking}"
        output_name = f"{safe_model_name}{thinking_suffix}"
    
    submission_file = Path(args.output_dir) / f"{output_name}-submission.jsonl"
    
    # Create submission file
    print(f"Creating submission file: {submission_file}")
    client.create_submission_file(
        questions=questions,
        output_path=str(submission_file),
        model_name=args.model,
        thinking=thinking
    )
    print(f"Created submission file with {len(questions)} requests")
    
    # Submit batch
    print("Submitting batch...")
    batch_metadata = {'submission_file': str(submission_file)}
    if thinking is not None:
        batch_metadata['thinking'] = str(thinking.value if hasattr(thinking, 'value') else thinking)

    batch_id = client.submit_batch(
        submission_file=str(submission_file),
        completion_window=args.completion_window,
        metadata=batch_metadata
    )
    print(f"Batch submitted successfully!")
    print(f"Batch ID: {batch_id}")
    
    # Save batch ID
    if args.output_name:
        batch_id_dir = Path('batch_ids')
        batch_id_dir.mkdir(parents=True, exist_ok=True)
        batch_id_file = batch_id_dir / f"{output_name}.txt"
        with open(batch_id_file, 'w') as f:
            f.write(batch_id)
    else:
        batch_id_file = client.save_batch_id(batch_id, args.model, thinking)
    print(f"Batch ID saved to: {batch_id_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
