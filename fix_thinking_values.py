#!/usr/bin/env python3
"""Fix legacy model names and thinking values in CSVs."""

import json
import pandas as pd
from pathlib import Path

# Model name renames: legacy -> canonical
MODEL_RENAMES = {
    "gpt-4o-2024-11-20": "gpt-4o",
    "o3-mini-2025-01-31": "o3-mini",
    "gemini-flash-2-5": "gemini-2.5-flash",
    "gemini-pro-2-5": "gemini-2.5-pro",
}

# Google models use uppercase, OpenAI uses lowercase
# Only fix if value is wrong case for the model
GOOGLE_THINKING_FIX = {
    'low': 'LOW',
    'medium': 'MEDIUM', 
    'high': 'HIGH',
    'minimal': 'MINIMAL',
}

OPENAI_THINKING_FIX = {
    'LOW': 'low',
    'MEDIUM': 'medium', 
    'HIGH': 'high',
}

def fix_csv(csv_path):
    """Fix model names and thinking_budget values in CSV."""
    df = pd.read_csv(csv_path)
    modified = False
    
    # Rename models
    if 'model' in df.columns:
        for old, new in MODEL_RENAMES.items():
            mask = df['model'] == old
            if mask.any():
                df.loc[mask, 'model'] = new
                print(f"  Renamed {old} -> {new} ({mask.sum()} rows)")
                modified = True
    
    # Fix thinking_budget case based on model
    if 'thinking_budget' in df.columns and 'model' in df.columns:
        # Google models: lowercase -> uppercase
        google_mask = df['model'].str.contains('gemini', case=False, na=False)
        for old, new in GOOGLE_THINKING_FIX.items():
            mask = google_mask & (df['thinking_budget'] == old)
            if mask.any():
                df.loc[mask, 'thinking_budget'] = new
                print(f"  Fixed thinking_budget {old} -> {new} for Google ({mask.sum()} rows)")
                modified = True
        
        # OpenAI models: uppercase -> lowercase
        openai_mask = df['model'].str.contains('o1|o3', case=False, regex=True, na=False)
        for old, new in OPENAI_THINKING_FIX.items():
            mask = openai_mask & (df['thinking_budget'] == old)
            if mask.any():
                df.loc[mask, 'thinking_budget'] = new
                print(f"  Fixed thinking_budget {old} -> {new} for OpenAI ({mask.sum()} rows)")
                modified = True
    
    if modified:
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    else:
        print(f"  No changes needed: {csv_path}")

def main():
    csv_dir = Path("model_responses")
    
    for csv_file in csv_dir.glob("*.csv"):
        print(f"\nProcessing {csv_file.name}...")
        fix_csv(csv_file)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
