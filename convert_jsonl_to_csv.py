#!/usr/bin/env python3
"""
Convert JSONL batch results to CSV format compatible with analysis.ipynb

Based on the old 2_process_results.ipynb conversion logic.
"""

import json
import re
import pandas as pd
from pathlib import Path

def read_jsonl(file_path):
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_results_to_csv(*, results_jsonl_path, submission_jsonl_path, output_csv_path, provider_cls, model_name=None):
    """
    Convert batch results JSONL to CSV format.
    
    Args:
        results_jsonl_path: Path to results JSONL file
        submission_jsonl_path: Path to submission JSONL file
        output_csv_path: Path to output CSV file
        provider_cls: Provider class (e.g. AnthropicProviderClient)
        model_name: Override model name
    """

    # Read results
    results_list = read_jsonl(results_jsonl_path)
    
    # Read submission file to map UUIDs to thinking_budget
    submission_list = read_jsonl(submission_jsonl_path)
    uuid_to_budget = {}
    for sub in submission_list:
        uuid = sub.get('custom_id')
        
        body = sub.get('body', {})
        # Extract thinking budget from body if present
        # 1. Standard OpenAI param (e.g. for some models)
        thinking_budget = body.get('thinking_budget', 0)
        
        # 2. Google extra_body nested param
        if thinking_budget == 0:
            extra_body = body.get('extra_body', {})
            # Check Google thinking config
            if 'google' in extra_body and 'thinking_config' in extra_body['google']:
                tc = extra_body['google']['thinking_config']
                if 'thinking_budget' in tc:
                    thinking_budget = tc['thinking_budget']
                elif 'thinking_level' in tc:
                    thinking_budget = tc['thinking_level']
            
        # 3. Check for reasoning_effort (OpenAI o1/o3)
        reasoning_effort_val = body.get('reasoning_effort', None)
        if reasoning_effort_val and thinking_budget == 0:
            thinking_budget = reasoning_effort_val
        
        # 4. Check for Anthropic thinking budget
        # Anthropic stores params at top level in submission file
        params = sub.get('params', {})
        if params and thinking_budget == 0:
            thinking = params.get('thinking', {})
            if thinking.get('type') == 'enabled' and 'budget_tokens' in thinking:
                thinking_budget = thinking['budget_tokens']

        uuid_to_budget[uuid] = thinking_budget
    
    # Use provider class for parsing (static method)
    records = []
    
    for api_response in results_list:
        parsed = provider_cls.parse_batch_result(api_response)
        
        uuid = parsed['custom_id']
            
        model = parsed.get('model')
        raw_answer = parsed['raw_answer']
        prompt_tokens = parsed['prompt_tokens']
        reasoning_tokens = parsed['reasoning_tokens']
        reasoning_trace = parsed['reasoning_trace']
        completion_tokens = parsed.get('completion_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens

        # Strict lookup from submission file map
        thinking_budget = uuid_to_budget.get(uuid, 0)
        
        if model_name:
            model = model_name
        elif not model:
            model = 'unknown'

        records.append({
            'model': model,
            'uuid': uuid,
            'thinking_budget': thinking_budget,
            'prompt_tokens': prompt_tokens,
            'reasoning_trace': reasoning_trace,
            'reasoning_tokens': reasoning_tokens,
            'raw_model_answer': raw_answer,
            'total_tokens': total_tokens
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Ensure directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"Converted {len(records)} records to {output_csv_path}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nModel: {df['model'].iloc[0]}")
    print(f"Thinking budgets: {sorted(df['thinking_budget'].unique())}")
    
    return df

if __name__ == '__main__':
    import sys
    
    # Fail fast if args are missing
    if len(sys.argv) < 3:
        print("Usage: python convert_jsonl_to_csv.py <results.jsonl> <submission.jsonl> [output.csv] [model_name] [provider]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    submission_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'output.csv'
    model_name = sys.argv[4] if len(sys.argv) > 4 else None
    
    if len(sys.argv) < 6:
        print("Usage: python convert_jsonl_to_csv.py <results.jsonl> <submission.jsonl> <output.csv> <model_name> <provider>")
        sys.exit(1)
    
    from providers import PROVIDER_MAP
    
    provider_arg = sys.argv[5]
    provider_cls = PROVIDER_MAP[provider_arg]
        
    convert_results_to_csv(
        results_jsonl_path=results_path, 
        submission_jsonl_path=submission_path, 
        output_csv_path=output_path, 
        provider_cls=provider_cls,
        model_name=model_name
    )
