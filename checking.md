# ChemIQ Pipeline - Quick Start Guide

This guide walks you through running the ChemIQ evaluation pipeline from start to finish.

## Prerequisites

1. **Python Environment**: Python 3.11+ with required packages:
   ```bash
   conda create -n ChemIQ python=3.11 numpy pandas matplotlib scipy requests openai rdkit -c conda-forge
   conda activate ChemIQ
   ```

2. **API Keys**: Set environment variables for your LLM provider:
   ```bash
   # For Google/Gemini models
   export GOOGLE_API_KEY="your-api-key-here"
   
   # For OpenAI models
   export OPENAI_API_KEY="your-api-key-here"
   
   # For Anthropic Claude models
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   
   **Note**: For Anthropic, you also need to install the SDK:
   ```bash
   pip install anthropic
   ```

3. **Questions File**: Ensure `questions/chemiq.jsonl` exists with your benchmark questions.

## Workflow Overview

The pipeline consists of three main steps:

1. **Submit batches** → Create and submit batch jobs to LLM APIs
2. **Download results** → Check status and download completed batch results
3. **Evaluate** → Process results, compute metrics, and generate plots

---

## Step 1: Submit Batch Jobs

Use `submit_batch.py` to create and submit batch jobs to your chosen LLM provider.

### Google/Gemini Models

```bash
python submit_batch.py \
  --model gemini-3-flash-preview \
  --questions questions/chemiq.jsonl \
  --provider google \
  --thinking-budget 1024 \
  --api-key $GOOGLE_API_KEY
```

**Options:**
- `--model`: Model name (e.g., `gemini-3-flash-preview`, `gemini-pro-2-5`)
- `--questions`: Path to questions JSONL file
- `--provider`: `google`, `openai`, or `anthropic` (default: `google`)
- `--thinking-budget`: Thinking budget in tokens (default: 0)
- `--api-key`: API key (or set env var)
- `--output-dir`: Directory for submission files (default: `submissions/`)
- `--completion-window`: Expected completion time (default: `24h`)
- `--all-questions`: Include all questions, not just ChemIQ ones

**What it does:**
- Creates a submission JSONL file in `submissions/`
- Uploads the file to the provider's API
- Submits a batch job
- Saves the batch ID to `batch_ids/{model}-{date}.txt`

**Example output:**
```
Loading questions from questions/chemiq.jsonl...
Loaded 816 questions
Creating submission file: submissions/gemini-3-flash-preview-2025-12-28-submission.jsonl
Created submission file with 816 requests
Submitting batch...
Batch submitted successfully!
Batch ID: batches/abc123xyz
Batch ID saved to: batch_ids/gemini-3-flash-preview-2025-12-28.txt
```

### OpenAI Models

```bash
python submit_batch.py \
  --model o3-mini-2025-01-31 \
  --questions questions/chemiq.jsonl \
  --provider openai \
  --thinking-budget 2 \
  --api-key $OPENAI_API_KEY
```

**Note:** For OpenAI o3-mini models, `--thinking-budget` maps to:
- `0` = low
- `1` = medium  
- `2` = high

### Anthropic Claude Models

```bash
python submit_batch.py \
  --model claude-3-5-sonnet-20241022 \
  --questions questions/chemiq.jsonl \
  --provider anthropic \
  --api-key $ANTHROPIC_API_KEY
```

**Note:** 
- Supported models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
- Requires `anthropic` Python package: `pip install anthropic`

**With thinking budget:**
```bash
python submit_batch.py \
  --model claude-3-5-sonnet-20241022 \
  --questions questions/chemiq.jsonl \
  --provider anthropic \
  --thinking-budget 1024 \
  --api-key $ANTHROPIC_API_KEY
```

**Thinking Budget Notes:**
- **OpenAI**: Use `0`=low, `1`=medium, `2`=high for o3-mini models (maps to `reasoning_effort`)
- **Google**: Use integer values (e.g., `1024`, `32768`) for reasoning models (maps to `thinking_budget`)
- **Anthropic**: Use integer values (e.g., `1024`, `32768`) for reasoning models (maps to `budget_tokens`)

---

## Step 2: Download Results

Use `download_results.py` to check batch status and download completed results.

### Basic Usage

```bash
python download_results.py \
  --batch-id-file batch_ids/gemini-3-flash-preview-2025-12-28.txt \
  --provider google \
  --api-key $GOOGLE_API_KEY
```

**Or with batch ID directly:**

```bash
python download_results.py \
  --batch-id batches/abc123xyz \
  --provider google \
  --api-key $GOOGLE_API_KEY
```

**Options:**
- `--batch-id-file`: Path to batch ID file (or use `--batch-id`)
- `--batch-id`: Batch ID directly
- `--provider`: `google`, `openai`, or `anthropic` (default: `google`)
- `--api-key`: API key (or set env var)
- `--results-dir`: Directory for JSONL results (default: `results/`)
- `--csv-dir`: Directory for CSV output (default: `model_responses/`)
- `--submissions-dir`: Directory for submission files (default: `submissions/`)
- `--force`: Force re-download even if files exist
- `--model-name`: Override model name in CSV

**What it does:**
- Checks batch status
- Downloads results when ready (saves to `results/`)
- Converts JSONL to CSV format (saves to `model_responses/`)

**Example output:**
```
Batch ID: batches/abc123xyz
Checking batch status: batches/abc123xyz
Status: completed
Progress: 816/816 completed
Downloading results...
Downloaded 816 records to results/gemini-3-flash-preview-2025-12-28-results.jsonl

Converting to CSV...
Converted 816 records to model_responses/gemini-3-flash-preview-2025-12-28-responses.csv
CSV saved to: model_responses/gemini-3-flash-preview-2025-12-28-responses.csv
```

**If batch is not ready:**
```
Status: in_progress
Progress: 450/816 completed
Batch not yet completed. Please wait and try again later.
```

---

## Step 3: Evaluate Results

Use `evaluate.py` to process all CSV files, compute metrics, and generate comparison plots.

### Basic Usage

```bash
python evaluate.py \
  --csv-dir model_responses \
  --questions questions/chemiq.jsonl \
  --output-dir figures
```

**Options:**
- `--csv-dir`: Directory containing CSV files to combine (default: `model_responses`)
- `--questions`: Path to questions JSONL file (required)
- `--output-dir`: Directory for output plots (default: `figures`)
- `--save-processed`: Path to save processed DataFrame (optional)

**What it does:**
- Finds all CSV files in the specified directory
- Combines them into a single DataFrame (removes duplicates)
- Parses answers and checks correctness
- Adds question metadata
- Generates comprehensive comparison plots:
  1. Combined radar/bar grid (by provider)
  2. Task-specific comparisons (one plot per task, all models)
  3. All models thinking budget comparison
  4. Gemini Flash 2.5 specific plot (if applicable)
  5. Shortest path and atom mapping comparison
- Prints summary statistics

**Example output:**
```
Scanning model_responses for CSV files...
Found 3 CSV files:
  - gemini-3-flash-2025-12-28-responses.csv
  - o3-mini-2025-12-28-responses.csv
  - gemini-pro-2-5-2025-12-28-responses.csv

Combined 2448 total responses
Removed 0 duplicate entries

Parsing answers...
Checking answers...
Adding question metadata...

Generating plots...
  1. Combined radar/bar grid (by provider)...
  2. Task-specific comparisons (all models)...
     - shortest_path...
     - counting_ring...
     - counting_carbon...
     - nmr_elucidation...
     - reaction...
     - sar...
     - smiles_to_iupac...
     - atom_mapping...
  3. All models thinking budget comparison...
  4. Gemini Flash 2.5 plot...
  5. Shortest path and atom mapping comparison...
  6. SMILES to IUPAC breakdown...

All plots saved to figures/

============================================================
SUMMARY STATISTICS
============================================================

Total responses: 2448
Unique models: 3
Unique questions: 816

Overall success rate: 45.2%

Success rate by model:
  gemini-3-flash-preview           52.1% (n=816)
  gemini-pro-2-5                   48.3% (n=816)
  o3-mini-2025-01-31               35.1% (n=816)

Success rate by task category:
  Shortest Path                     62.5% (n=102)
  Counting Ring                     58.3% (n=102)
  Counting Carbon                   55.1% (n=102)
  NMR Elucidation                  48.2% (n=102)
  Reaction                          45.3% (n=102)
  SAR                               42.1% (n=102)
  SMILES to IUPAC                  38.9% (n=102)
  Atom Mapping                      35.2% (n=102)
```

---

## Complete Example Workflow

Here's a complete example for running the full pipeline:

```bash
# 1. Submit batch for Gemini Flash
python submit_batch.py \
  --model gemini-3-flash-preview \
  --questions questions/chemiq.jsonl \
  --provider google \
  --thinking-budget 1024

# Wait for batch to complete (check status periodically)
# 2. Download results
python download_results.py \
  --batch-id-file batch_ids/gemini-3-flash-preview-2025-12-28.txt \
  --provider google

# 3. Submit batch for o3-mini
python submit_batch.py \
  --model o3-mini-2025-01-31 \
  --questions questions/chemiq.jsonl \
  --provider openai \
  --thinking-budget 2

# Wait and download
python download_results.py \
  --batch-id-file batch_ids/o3-mini-2025-01-31-2025-12-28.txt \
  --provider openai

# 4. Evaluate all results together
python evaluate.py \
  --csv-dir model_responses \
  --questions questions/chemiq.jsonl \
  --output-dir figures
```

---

## File Structure

After running the pipeline, your directory structure will look like:

```
ChemIQ/
├── batch_ids/                    # Batch ID files (one per submission)
│   ├── gemini-3-flash-preview-2025-12-28.txt
│   └── o3-mini-2025-01-31-2025-12-28.txt
├── submissions/                  # Submission JSONL files
│   ├── gemini-3-flash-preview-2025-12-28-submission.jsonl
│   └── o3-mini-2025-01-31-2025-12-28-submission.jsonl
├── results/                      # Raw JSONL results
│   ├── gemini-3-flash-preview-2025-12-28-results.jsonl
│   └── o3-mini-2025-01-31-2025-12-28-results.jsonl
├── model_responses/              # Processed CSV files
│   ├── gemini-3-flash-preview-2025-12-28-responses.csv
│   └── o3-mini-2025-01-31-2025-12-28-responses.csv
└── figures/                      # Generated plots
    ├── combined_radar_bar_grid_updated.png
    ├── combined_radar_bar_grid_updated.pdf
    ├── all_models_comparison.png
    ├── fig_shortestpath_atommapping_updated.png
    └── ...
```

---

## Tips

1. **Multiple Thinking Budgets**: Submit multiple batches with different thinking budgets for the same model to compare performance:
   ```bash
   for budget in 0 1024 2048 4096; do
     python submit_batch.py --model gemini-3-flash-preview \
       --questions questions/chemiq.jsonl \
       --provider google \
       --thinking-budget $budget
   done
   ```

2. **Check Batch Status**: You can run `download_results.py` multiple times - it will tell you if the batch is still processing.

3. **Combining Results**: `evaluate.py` automatically finds and combines all CSV files in the directory. No need to manually merge files.

4. **Force Re-download**: Use `--force` if you need to re-download results (e.g., if the file was corrupted).

5. **Save Processed Data**: Use `--save-processed` to save the processed DataFrame for further analysis.

---

## Troubleshooting

**Batch submission fails:**
- Check your API key is set correctly
- Verify the model name is correct for your provider
- Ensure you have sufficient API credits/quota

**Download fails:**
- Batch may not be completed yet - wait and try again
- Check batch status manually using the provider's dashboard
- Verify batch ID is correct

**Evaluation fails:**
- Ensure CSV files exist in the specified directory
- Check that questions file path is correct
- Verify CSV files have required columns (`model`, `uuid`, `raw_model_answer`, etc.)

---

## Additional Scripts

- `convert_jsonl_to_csv.py`: Standalone converter for JSONL to CSV (usually called automatically by `download_results.py`)

## Architecture Notes

- **Provider Clients**: All providers use native SDKs (no OpenAI-compatible wrappers)
  - OpenAI: Native OpenAI SDK
  - Google: Native Google genai SDK
  - Anthropic: Native Anthropic SDK
- **Data Models**: Output models (`BatchStatus`, `RequestCounts`) provide type safety for return values
- **File Formats**: 
  - Submission files: JSONL format (OpenAI-compatible structure for Google/OpenAI)
  - Results files: JSONL format (provider-specific structure)
  - Processed files: CSV format (standardized across all providers)

