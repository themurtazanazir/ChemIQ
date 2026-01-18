
import os
import re
import argparse
import pandas as pd
from pathlib import Path

def normalize_budget(val):
    try:
        if pd.isna(val):
            return 0
        if isinstance(val, str):
             # Try to parse if it's a number string
            if val.replace('.','',1).isdigit():
                return int(float(val))
            return 0 
        return int(val)
    except (ValueError, TypeError):
        return 0

def infer_metadata(row):
    model_orig = str(row.get('model', ''))
    budget_raw = row.get('thinking_budget', 0)
    
    # 1. Model Variant (strip dates)
    variant = re.sub(r'-20\d{2}-\d{2}-\d{2}.*$', '', model_orig)
    if "DeepSeek" in variant:
         variant = re.sub(r'-\d{4}$', '', variant)

    # 2. Thinking Budget
    budget_tokens = normalize_budget(budget_raw)
    
    # 3. Model Family
    family = "standard"
    reasoning_keywords = ["o1", "o3", "r1", "reasoning", "thinking", "gpt-5"]
    if any(k in variant.lower() for k in reasoning_keywords):
        family = "reasoning"
    elif budget_tokens > 0:
        family = "reasoning"
    
    # We return the key identifiers for grouping + the new metadata columns
    return pd.Series({
        'model_family': family,
        'model_variant': variant,
        'thinking_budget': budget_tokens,
        # We need to return other columns to keep them? 
        # Actually we will use these to group the DF.
    })

def main():
    parser = argparse.ArgumentParser(description="Consolidate legacy ChemIQ CSVs to v2 schema")
    parser.add_argument('--input-dir', default='model_responses', help='Source directory')
    parser.add_argument('--output-dir', default='model_responses_v2', help='Destination directory')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all data
    all_dfs = []
    
    print(f"Scanning {input_dir}...")
    for csv_file in input_dir.glob("*.csv"):
        # Exclude combined and additional files
        if "combined" in csv_file.name or "additional" in csv_file.name:
            print(f"  Skipping {csv_file.name} (combined/additional)")
            continue
            
        print(f"  Reading {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            
    if not all_dfs:
        print("No files found to migrate.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(full_df)}")
    
    # 2. Augment with metadata
    meta = full_df.apply(infer_metadata, axis=1)
    full_df['model_variant'] = meta['model_variant']
    full_df['model_family'] = meta['model_family']
    # Use normalized budget
    full_df['thinking_budget'] = meta['thinking_budget']
    
    # 3. Group by (model_variant, thinking_budget)
    # We want to merge runs.
    # Logic: Deduplicate on UUID within the group (Latest Wins is tricky if we don't know file order here)
    # Since we loaded glob order (usually alphabetical), timestamps likely sorted? 
    # Let's assume glob order is roughly chronological or random. 
    # Best effort deduplication: keep 'last'
    
    groups = full_df.groupby(['model_variant', 'thinking_budget'])
    
    print("\nWriting canonical files...")
    for (variant, budget), group_df in groups:
        # Deduplicate UUIDs within this canonical group
        # Keeping duplicates? No, user said "let it replace?". 
        # But if we have mixed older/newer files, 'last' depend on glob order.
        # It's safer to dedup.
        deduped = group_df.drop_duplicates(subset=['uuid'], keep='last')
        
        filename = f"{variant}-budget{budget}.csv"
        out_path = output_dir / filename
        
        deduped.to_csv(out_path, index=False)
        print(f"  -> {filename} ({len(deduped)} records)")

if __name__ == "__main__":
    main()
