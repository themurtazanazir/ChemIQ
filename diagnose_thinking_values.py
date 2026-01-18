#!/usr/bin/env python3
"""Diagnose thinking_budget values in CSVs vs expected provider values."""

import pandas as pd
from pathlib import Path
from providers import PROVIDER_MAP
from providers.google import ThinkingLevel as GoogleThinkingLevel
from providers.openai import ThinkingLevel as OpenAIThinkingLevel

def main():
    csv_dir = Path("model_responses")
    
    # Load all CSVs and build model -> set(thinking_budget) mapping
    model_to_values = {}
    
    for csv_file in csv_dir.glob("*.csv"):
        if csv_file.name == "combined_model_responses.csv":
            continue  # Skip combined file
        
        df = pd.read_csv(csv_file)
        if 'model' not in df.columns or 'thinking_budget' not in df.columns:
            print(f"Skipping {csv_file.name}: missing required columns")
            continue
        
        for model in df['model'].unique():
            if model not in model_to_values:
                model_to_values[model] = set()
            
            values = df[df['model'] == model]['thinking_budget'].unique()
            for v in values:
                model_to_values[model].add(str(v) if pd.notna(v) else 'NaN')
    
    print("=" * 60)
    print("MODEL -> THINKING_BUDGET VALUES IN CSVs")
    print("=" * 60)
    for model, values in sorted(model_to_values.items()):
        print(f"{model}: {values}")
    
    print("\n" + "=" * 60)
    print("EXPECTED VALUES FROM PROVIDERS")
    print("=" * 60)
    
    # Google ThinkingLevel values
    print("\nGoogle ThinkingLevel enum values:")
    for level in GoogleThinkingLevel:
        print(f"  {level.name} = '{level.value}'")
    
    # OpenAI ThinkingLevel values
    print("\nOpenAI ThinkingLevel enum values:")
    for level in OpenAIThinkingLevel:
        print(f"  {level.name} = '{level.value}'")
    
    print("\n" + "=" * 60)
    print("ANALYSIS: CHECKING FOR MISMATCHES")
    print("=" * 60)
    
    # Check each model
    google_valid = {l.value for l in GoogleThinkingLevel}
    openai_valid = {l.value for l in OpenAIThinkingLevel}
    
    for model, values in sorted(model_to_values.items()):
        issues = []
        
        for v in values:
            if v in ('0', '0.0', 'NaN', 'nan'):
                continue  # No thinking is fine
            
            # Check if it's a valid integer (budget)
            try:
                int(float(v))
                continue  # Integer budget is fine
            except (ValueError, TypeError):
                pass
            
            # Check if it's a valid enum value
            if v in google_valid or v in openai_valid:
                continue  # Valid uppercase enum
            
            # Check for lowercase versions
            if v.upper() in google_valid:
                issues.append(f"'{v}' should be '{v.upper()}' (Google)")
            elif v.upper() in openai_valid:
                issues.append(f"'{v}' should be '{v.upper()}' (OpenAI)")
            else:
                issues.append(f"'{v}' is unknown")
        
        if issues:
            print(f"\n{model}:")
            for issue in issues:
                print(f"  ❌ {issue}")

if __name__ == "__main__":
    main()
