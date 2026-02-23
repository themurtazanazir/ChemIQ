# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChemIQ is a benchmark for evaluating LLMs on chemistry reasoning tasks. It provides a pipeline for submitting questions to LLM APIs, evaluating responses, and generating comparative visualizations. Paper: https://arxiv.org/abs/2505.07735

## Key Commands

```bash
# Environment setup
conda create -n ChemIQ python=3.11 numpy pandas matplotlib scipy requests openai anthropic google-generativeai rdkit jupyterlab ipykernel

# Run full evaluation (generates figures + stats)
python evaluate.py --csv-dir model_responses --questions questions/chemiq.jsonl --output-dir figures --individual-plots-dir figures/individual --save-results figures/results

# Evaluate specific tasks only
python evaluate.py --csv-dir model_responses --questions questions/chemiq.jsonl --tasks nmr_elucidation,reaction --no-radar

# Exclude specific models
python evaluate.py --csv-dir model_responses --questions questions/chemiq.jsonl --exclude-models ether0

# Submit batch evaluation
python submit_batch.py --provider google --model gemini-3-flash-preview --questions questions/chemiq.jsonl --thinking-budget HIGH

# Download batch results
python download_results.py --provider google --batch-id <id>

# Run sync evaluation (e.g., Ether0)
python run_sync_eval.py --provider ether0 --model ether0 --questions questions/chemiq.jsonl --workers 4
```

## Architecture

### Data Flow

```
questions/*.jsonl → submit_batch.py/run_sync_eval.py → [LLM APIs]
    → results/*.jsonl → download_results.py → model_responses/*.csv
    → evaluate.py → figures/ (heatmaps, bar charts, radar plots, leaderboards)
```

### Provider System

Providers in `providers/` implement two base classes from `providers/base.py`:
- **`ProviderClient`** (batch APIs): OpenAI, Google Gemini, Anthropic Claude
- **`SyncProviderClient`** (real-time): Ether0

Models are registered via `SUPPORTED_MODELS` dict in each provider. All models aggregate into `ALL_MODELS` in `evaluate.py` via `PROVIDER_MAP` and `SYNC_PROVIDER_MAP` from `providers/__init__.py`.

Thinking budgets differ by provider:
- **OpenAI** (o3-mini): `ThinkingLevel` enum (LOW/MEDIUM/HIGH)
- **Google Gemini 3+**: `ThinkingLevel` enum (MINIMAL/LOW/MEDIUM/HIGH)
- **Google Gemini 2.x**: Integer token budgets (e.g., 8192, 24576)
- **Anthropic Claude**: Integer `budget_tokens` (e.g., 5000, 16000)

### Answer Processing Pipeline

1. **Parse** (`utils/parser.py`): Extract answers from model output (handles \boxed{}, `<thought>` tags, Ether0 `<|answer_start|>` format)
2. **Verify** (`utils/answer_verifier.py`): Check correctness using format-specific methods (exact match, SMILES canonicalization via RDKit, OPSIN API for IUPAC names)

### Question Format (JSONL)

Each question has: `uuid`, `question_category`, `prompt`, `answer`, `answer_format` (integer/float/smiles/iupac/list_of_tuples), `verification_method`

8 task categories: `counting_carbon`, `counting_ring`, `shortest_path`, `atom_mapping`, `smiles_to_iupac`, `sar` (Free-Wilson), `reaction`, `nmr_elucidation`

### evaluate.py CLI Flags

- `--tasks`: Comma-separated task filter
- `--models` / `--exclude-models`: Model inclusion/exclusion filters
- `--no-radar`: Skip radar plots (useful when evaluating few tasks)
- `--individual-plots-dir`: Save per-model charts separately
- `--save-results`: Save CSV reports (summary, by_model, by_task, model_task_matrix) and heatmap

## Environment Variables

API keys are loaded from `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
