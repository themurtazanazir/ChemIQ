#!/usr/bin/env python3
"""Evaluate all model responses: combine CSVs and generate comprehensive comparison plots."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import binomtest
from rdkit import RDLogger
from tqdm import tqdm

from utils.answer_verifier import AnswerVerifier
from utils import parser

try:
    from statsmodels.stats.contingency import mcnemar as sm_mcnemar
except ImportError:
    sm_mcnemar = None

RDLogger.DisableLog('rdApp.*')
tqdm.pandas()


# ───────────────────────────────────────────────────────────
#  Constants
# ───────────────────────────────────────────────────────────

QUESTION_ORDER = [
    "shortest_path", "counting_ring", "counting_carbon",
    "nmr_elucidation", "reaction", "reaction_ether0_task", "sar",
    "smiles_to_iupac", "atom_mapping"
]

QUESTION_LABELS = {
    "counting_carbon": "Carbon Counting",
    "counting_ring": "Ring Counting",
    "sar": "Free-Wilson\nAnalysis",
    "shortest_path": "Shortest\nPath",
    "atom_mapping": "Atom Mapping",
    "smiles_to_iupac": "SMILES to IUPAC",
    "reaction": "Product of\nReaction",
    "reaction_ether0_task": "Ether0\nReaction",
    "nmr_elucidation": "NMR\nElucidation",
}


# ───────────────────────────────────────────────────────────
#  Data loading and processing
# ───────────────────────────────────────────────────────────

def load_questions(questions_files: str | list[str]) -> dict:
    """Load questions into a dictionary keyed by UUID from one or more files."""
    if isinstance(questions_files, str):
        questions_files = [questions_files]
    
    questions = []
    for questions_file in questions_files:
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    return {q["uuid"]: q for q in questions}


def find_and_combine_csvs(csv_dir: str) -> pd.DataFrame:
    """Find all CSV files in directory and combine them."""
    csv_dir_path = Path(csv_dir)
    if not csv_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")
    
    csv_files = list(csv_dir_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    
    print(f"Found {len(csv_files)} CSV files:")
    dfs = []
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined {len(combined)} total responses")
    
    # Build combined model config from all providers
    # ALL_MODELS is now global
    
    # Filter out unsupported models
    unsupported = combined[~combined['model'].isin(ALL_MODELS.keys())]
    if len(unsupported) > 0:
        import warnings
        unsupported_models = unsupported['model'].unique()
        for m in unsupported_models:
            count = len(unsupported[unsupported['model'] == m])
            warnings.warn(f"Ignoring {count} rows with unsupported model: {m}")
        combined = combined[combined['model'].isin(ALL_MODELS.keys())]
        print(f"Remaining: {len(combined)} rows")
    
    def _to_thinking_budget(row):
        val = row['thinking_budget']
        model = row['model']
        raw_config = ALL_MODELS.get(model)
        
        # Normalize config to thinking mode
        thinking_mode = raw_config
        if isinstance(raw_config, ModelConfig):
            thinking_mode = raw_config.thinking_mode
        
        if thinking_mode is None:
            if pd.isna(val) or val == 0 or val == '0' or val == '':
                return 0
            raise ValueError(f"Model {model} doesn't support thinking, but got {val}")
        elif thinking_mode == "budget":
            return int(float(val))
        else:
            # ThinkingLevel enum - handle 0/empty as None (missing)
            if pd.isna(val) or val == 0 or val == '0' or val == '':
                return None
            return thinking_mode(str(val))
              
    combined['thinking_budget'] = combined.apply(_to_thinking_budget, axis=1)
    return combined



def process_results(df: pd.DataFrame, questions_files: str | list[str]) -> pd.DataFrame:
    """Process model responses: parse answers and check correctness."""
    question_dict = load_questions(questions_files)
    valid_uuids = set(question_dict.keys())
    
    invalid_uuids = set(df["uuid"].unique()) - valid_uuids
    if invalid_uuids:
        print(f"Warning: Found {len(invalid_uuids)} UUIDs not in main questions file. Dropping them.")
        df = df[df["uuid"].isin(valid_uuids)].copy()
        
    if df.empty:
        raise ValueError("No valid questions remaining after filtering!")
    
    # Parse responses
    print("\nParsing answers...")
    answer_parser = parser.AnswerParser(questions_files, doClean=True)
    df["parsed_answer"] = df.progress_apply(
        lambda row: answer_parser.parse(row["uuid"], row["raw_model_answer"]), axis=1
    )
    
    # Check answers
    print("Checking answers...")
    answer_checker = AnswerVerifier(questions_files)
    df[['is_correct', 'opsin_smiles']] = df.progress_apply(
        lambda row: pd.Series(answer_checker.check_answer(row["uuid"], row["parsed_answer"])),
        axis=1
    )
    
    # Also check raw answers (without parsing)
    print("Checking raw answers (without parsing)...")
    raw_parser = parser.AnswerParser(questions_files, doClean=False)
    df["raw_parsed_answer"] = df.progress_apply(
        lambda row: raw_parser.parse(row["uuid"], row["raw_model_answer"]), axis=1
    )
    df[['raw_is_correct', 'raw_opsin_smiles']] = df.progress_apply(
        lambda row: pd.Series(answer_checker.check_answer(row["uuid"], row["raw_parsed_answer"])),
        axis=1
    )
    
    # Add question metadata
    print("Adding question metadata...")
    df["question_category"] = df["uuid"].apply(lambda x: question_dict.get(x, {}).get("question_category", "unknown"))
    df["sub_category"] = df["uuid"].apply(lambda x: question_dict.get(x, {}).get("sub_category", "unknown"))
    df["expected_answer"] = df["uuid"].apply(lambda x: question_dict.get(x, {}).get("answer", ""))
    
    # thinking_budget already normalized to enums/ints by find_and_combine_csvs
    df = create_model_label(df)
    
    df["output_tokens"] = df["total_tokens"] - df["prompt_tokens"]
    df["reasoning_tokens"] = df["output_tokens"]
    
    return df


def create_model_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create a standardized model label column for plotting."""
    df = df.copy()
    
    def _make_label(row):
        model = row['model']
        budget = row.get('thinking_budget', 0)
        
        # Handle enum (has .name attribute)
        if hasattr(budget, 'name'):
            return f"{model}-{budget.name}"
        
        # Handle int
        if isinstance(budget, int) and budget > 0:
            return f"{model}-{budget}"
        
        # Try to convert to int
        try:
            budget_int = int(budget)
            if budget_int > 0:
                return f"{model}-{budget_int}"
        except (ValueError, TypeError):
            pass
        
        return model

    df["model_label"] = df.apply(_make_label, axis=1)
    return df


# ───────────────────────────────────────────────────────────
#  Plotting helpers
# ───────────────────────────────────────────────────────────

def _budget_sort_key(val):
    """Sort key for thinking budget values. Enums use their order, ints sort numerically."""
    from providers.base import BaseThinkingLevel
    
    if val is None:
        return (0, 0)  # None (no thinking) comes first
    
    if isinstance(val, BaseThinkingLevel):
        return (1, val.order)  # Enums sort by their order
    
    if isinstance(val, (int, float)):
        return (2, val)  # Ints after enums
    
    return (3, str(val))  # Unknown last


def _pick_colour(val):
    """Pick a color for a thinking level. Enums use their color property."""
    from providers.base import BaseThinkingLevel
    
    if val is None:
        return "#E2C6FF"  # violet for no thinking
    
    if isinstance(val, BaseThinkingLevel):
        return val.color
    
    # Fallback: stable color from hash
    hash_val = sum(ord(c) for c in str(val))
    return cm.tab10(hash_val % 10)


def _darken(col, f=0.7):
    rgb = np.array(mcolors.to_rgb(col))
    return tuple(np.clip(rgb * f, 0, 1))


def _mcnemar_one_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    if sm_mcnemar is not None:
        p_two = sm_mcnemar([[0, b], [c, 0]], exact=True).pvalue
        return p_two / 2 if c > b else 1.0
    return binomtest(c, n=n, alternative="greater").pvalue


def _p_to_stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def _mcnemar_adjacent_pvals(df, category_col, ordered):
    pvals = []
    for l, r in zip(ordered[:-1], ordered[1:]):
        pv = df[df[category_col].isin([l, r])].pivot(
            index="uuid", columns=category_col, values="is_correct"
        ).dropna()
        if pv.empty:
            pvals.append(1.0)
            continue
        b = ((pv[l] == 1) & (pv[r] == 0)).sum()
        c = ((pv[l] == 0) & (pv[r] == 1)).sum()
        pvals.append(_mcnemar_one_sided(b, c))
    return pvals


def _add_significance_brackets(ax, bar_x, perf, pvals, bar_w=0.75, base_offset=0.08, step=0.04):
    for i, p in enumerate(pvals):
        x1, x2 = bar_x[i], bar_x[i + 1]
        y = max(perf[i], perf[i + 1]) + base_offset + i * step
        ax.plot([x1, x1, x2, x2], [y - 0.005, y, y, y - 0.005], color="black", lw=1, zorder=6)
        ax.text((x1 + x2) / 2, y + 0.005, _p_to_stars(p), ha="center", va="bottom", fontsize=8, zorder=6)


def _calc_numeric(df_subset, keys=None):
    """Calculate performance stats for a set of budgets."""
    df = df_subset.copy()
    if keys is None:
        unique_budgets = list(df["thinking_budget"].unique())
        keys = sorted(unique_budgets, key=_budget_sort_key)
    
    n = df["uuid"].nunique()
    # Reindex with explicit keys to ensure alignment
    perf = df.groupby("thinking_budget")["is_correct"].mean().reindex(keys)
    ci = np.sqrt(perf * (1 - perf) / n) * 1.96
    
    avg = df.groupby("thinking_budget")["reasoning_tokens"].mean().reindex(keys)
    std = df.groupby("thinking_budget")["reasoning_tokens"].std().reindex(keys).fillna(0)
    
    return keys, perf.values, ci.values, avg.values, std.values


def _prepare_model_stats(df, model_name, task_order=None):
    """
    Prepare all statistics needed for plotting a model's performance.
    Returns a dictionary of derived stats.
    """
    task_order = task_order or QUESTION_ORDER
    m_df = df[df["model"] == model_name]
    budgets = list(m_df["thinking_budget"].unique())
    
    # Filter for positive budgets if applicable
    def _is_positive_budget(b):
        if isinstance(b, (int, float)):
            return b > 0
        if hasattr(b, 'name'): # Enum
            return True
        if isinstance(b, str) and b.lower() in ["minimal", "low", "medium", "high"]:
            return True
        return False
        
    has_budgets = any(_is_positive_budget(b) for b in budgets)
    keys = sorted(budgets, key=_budget_sort_key) if has_budgets else [0]
    
    # Fallback to existing unique values if derived keys aren't in data
    if not any(k in budgets for k in keys):
        keys = sorted(budgets, key=_budget_sort_key)

    # Calculate bar stats
    keys, perf, ci, avg, sd = _calc_numeric(m_df, keys)
    
    # Calculate p-values if we have multiple bars
    pvals = _mcnemar_adjacent_pvals(m_df, "thinking_budget", keys) if len(keys) > 1 else None
    
    # Radar lookup (using task_order)
    radar_data = {}
    for k in keys:
        sub = m_df[m_df["thinking_budget"] == k]
        rates = _success_rates(m_df, sub, task_order=task_order)
        radar_data[k] = rates
        
    # Styling preparation
    colors = [_pick_colour(k) for k in keys]
    labels = [str(k.name if hasattr(k, 'name') else k) for k in keys]
    
    return {
        'model': model_name,
        'df': m_df,
        'keys': keys,
        'perf': perf,
        'ci': ci,
        'avg': avg,
        'std': sd,
        'pvals': pvals,
        'radar_data': radar_data,
        'colors': colors,
        'labels': labels,
        'max_token_val': np.nanmax(avg + sd) if len(avg) > 0 else 0,
        'task_order': task_order,
    }


def _plot_bars_from_stats(ax, stats, **kwargs):
    """
    Plot the bar chart component using the pre-calculated stats object.
    Supported kwargs: tok_max, show_tok_ylabel, show_token_line, bar_w, offset
    """
    x = np.arange(len(stats['keys']))
    bar_w = kwargs.get('bar_w', 0.75)
    offset = kwargs.get('offset', 0.1)
    tok_max = kwargs.get('tok_max', None)
    
    # 1. Bars
    ax.bar(x, stats['perf'], width=bar_w, color=stats['colors'], edgecolor="black", zorder=3)
    ax.errorbar(x, stats['perf'], yerr=stats['ci'], fmt="none", capsize=3, ecolor="black", zorder=4)
    
    # 2. X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(stats['labels'], fontsize=8)
    
    # 3. Y-axis (Primary)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    # Grid
    for y in np.arange(0.2, 1, 0.2):
        ax.axhline(y, ls="--", lw=0.5, color="grey", alpha=0.4, zorder=1)
        
    # 4. Token Line (Secondary Axis)
    if kwargs.get('show_token_line', True):
        ax2 = ax.twinx()
        ax2.errorbar(x + offset, stats['avg'], yerr=stats['std'], marker="o", ms=4, lw=1, capsize=3, color="blue", zorder=5)
        
        if tok_max is not None:
             ax2.set_ylim(0, tok_max)
        elif stats['max_token_val'] > 0:
             ax2.set_ylim(0, stats['max_token_val'] * 1.1)
             
        ax2.tick_params(axis="y", colors="blue")
        ax2.spines["right"].set_color("blue")
        ax2.tick_params(axis="y", labelsize=8)
        
        if not kwargs.get('show_tok_ylabel', True):
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("Reasoning Tokens", color="blue", fontsize=8)

    # 5. Significance
    if stats['pvals']:
        _add_significance_brackets(ax, x, stats['perf'], stats['pvals'], bar_w)

    # 6. Formatting Labels
    diff_labels = len(stats['labels']) > 4 or any(len(str(l)) > 4 for l in stats['labels'])
    if diff_labels:
        ax.tick_params(axis='x', rotation=90)
    else:
         ax.tick_params(axis='x', rotation=0)

    ax.set_box_aspect(1)


def _plot_radar_from_stats(ax, stats):
    """Plot the radar component using pre-calculated stats."""
    task_order = stats.get('task_order', QUESTION_ORDER)
    
    # 1. Sort keys by mean performance (biggest areas at bottom z-order)
    key_means = []
    for k in stats['keys']:
        vals = stats['radar_data'][k]
        key_means.append((k, np.mean(vals)))
        
    key_means.sort(key=lambda x: x[1], reverse=True)
    sorted_keys = [k for k, _ in key_means]
    
    for i, key in enumerate(sorted_keys):
        vals = np.asarray(stats['radar_data'][key])
        
        # Close the loop
        ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False)
        vals_c = np.r_[vals, vals[0]]
        ang_c = np.r_[ang, ang[0]]
        
        # Color match
        color_idx = stats['keys'].index(key)
        base_col = stats['colors'][color_idx]
        
        # Plot
        ax.fill(ang_c, vals_c, fc=base_col, ec=None, alpha=0.4, zorder=i)
        ax.plot(ang_c, vals_c, lw=1, color=_darken(base_col), zorder=i + len(sorted_keys))
        
    # Axes setup with task_order
    _beautify_radar_axis(ax, task_order=task_order, show_xt=True)
    

def _beautify_radar_axis(ax, task_order=None, show_xt=False):
    """Setup radar axis with optional custom task order."""
    task_order = task_order or QUESTION_ORDER
    N = len(task_order)
    label_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ax.set_xticks(label_angles)
    ax.set_xticklabels([])
    
    if show_xt:
        offset_r = 1.05
        for txt, ang in zip(task_order, label_angles):
            lbl = QUESTION_LABELS.get(txt, txt)
            deg = np.degrees(ang)
            ha = "right" if 90 < deg < 270 else ("center" if deg in (90, 270) else "left")
            ax.text(ang, offset_r, lbl, ha=ha, va="center", fontsize=8)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_linestyle("--")
        gl.set_alpha(0.7)


def _success_rates(base, grp, task_order=None, correct_col="is_correct"):
    """Calculate success rates for each task category."""
    task_order = task_order or QUESTION_ORDER
    rates = []
    for q in task_order:
        if q == "smiles_to_iupac":
            base_mask = (base["question_category"] == "smiles_to_iupac") & (
                base["sub_category"].isin(["zinc_canonical", "zinc_random"])
            )
            grp_mask = (grp["question_category"] == "smiles_to_iupac") & (
                grp["sub_category"].isin(["zinc_canonical", "zinc_random"])
            )
        else:
            base_mask = base["question_category"] == q
            grp_mask = grp["question_category"] == q
        denom = base.loc[base_mask, "uuid"].nunique()
        num = grp.loc[grp_mask, correct_col].sum()
        rates.append(num / denom if denom else 0)
    return rates


def plot_model_pair(ax_bar, ax_radar, stats, **kwargs):
    """
    Orchestrate plotting both bar and radar for a single model on provided axes.
    """
    _plot_bars_from_stats(ax_bar, stats, **kwargs)
    _plot_radar_from_stats(ax_radar, stats)


# Build combined model config from all providers (batch + sync)
from providers import PROVIDER_MAP, SYNC_PROVIDER_MAP
from providers.base import ModelConfig

ALL_MODELS = {}
for provider_cls in PROVIDER_MAP.values():
    ALL_MODELS.update(provider_cls.SUPPORTED_MODELS)
for provider_cls in SYNC_PROVIDER_MAP.values():
    ALL_MODELS.update(provider_cls.SUPPORTED_MODELS)


def _prepare_leaderboard_stats(df, models):
    """
    Constructs a stats object for the 'Best Config' comparison.
    """
    best_configs = []
    
    for m in models:
        m_df = df[df['model'] == m]
        if m_df.empty: continue
        
        budgets = m_df["thinking_budget"].unique()
        best_b, best_score = None, -1.0
        
        # Determine best budget
        for b in budgets:
            score = m_df[m_df["thinking_budget"] == b]["is_correct"].mean()
            if score > best_score:
                best_score, best_b = score, b
                
        if best_b is not None:
            best_configs.append({
                'model': m, 'budget': best_b, 'score': best_score,
                'df': m_df[m_df["thinking_budget"] == best_b]
            })
            
    best_configs.sort(key=lambda x: x['score'], reverse=False)
    
    # We need to build the arrays expected by stats
    labels, perf, cis, avgs, stds, colors = [], [], [], [], [], []
    keys = []
    
    for i, cfg in enumerate(best_configs):
        sub = cfg['df']
        n = sub["uuid"].nunique()
        p = sub["is_correct"].mean()
        ci = np.sqrt(p * (1 - p) / n) * 1.96
        avg_tok = sub["reasoning_tokens"].mean()
        std_tok = sub["reasoning_tokens"].std()
        if pd.isna(std_tok): std_tok = 0
        
        budget_label = cfg['budget'].name if hasattr(cfg['budget'], 'name') else str(cfg['budget'])
        label = f"{cfg['model']}\n({budget_label})"
        
        keys.append(i) # Dummy key
        labels.append(label)
        perf.append(p)
        cis.append(ci)
        avgs.append(avg_tok)
        stds.append(std_tok)
        
        # Use Model Color from config
        m_cfg = ALL_MODELS.get(cfg['model'])
        if isinstance(m_cfg, ModelConfig):
            colors.append(m_cfg.color)
        else:
             # Fallback
            import matplotlib.cm as cm
            hash_val = sum(ord(c) for c in cfg['model'])
            colors.append(cm.tab10(hash_val % 10))
        
    return {
        'model': 'Leaderboard',
        'keys': keys,
        'perf': np.array(perf),
        'ci': np.array(cis),
        'avg': np.array(avgs),
        'std': np.array(stds),
        'pvals': None, 
        'radar_data': {}, # populated later if needed
        'colors': colors,
        'labels': labels,
        'max_token_val': np.nanmax(np.array(avgs) + np.array(stds)) if len(avgs) > 0 else 0,
        'configs': best_configs
    }


def _prepare_summary_radar_stats(df, leaderboard_stats, task_order=None):
    """Extracts all models from leaderboard stats and formats for radar plotting."""
    task_order = task_order or QUESTION_ORDER
    configs = leaderboard_stats.get('configs', [])
    if not configs: return None
    
    # Use all models, not just top 3
    all_configs = configs
    
    keys = []
    radar_data = {}
    colors = []
    
    for cfg in all_configs:
        key = cfg['model']
        keys.append(key)
        radar_data[key] = _success_rates(df, cfg['df'], task_order=task_order)
        
        m_cfg = ALL_MODELS.get(cfg['model'])
        if isinstance(m_cfg, ModelConfig):
            colors.append(m_cfg.color)
        else:
            configs_c = leaderboard_stats['colors']
            # Try to find matching color from leaderboard stats
            # This is hard because leaderboard stats colors are by index.
            # Simple fallback
            import matplotlib.cm as cm
            hash_val = sum(ord(c) for c in cfg['model'])
            colors.append(cm.tab10(hash_val % 10))
        
    return {
        'keys': keys,
        'radar_data': radar_data,
        'colors': colors,
        'task_order': task_order,
    }

# ... (inside plot_combined_radar_bar_grid where it calls this)
# Be careful with the context matching for replacement.

# Let's target the function definition separately from the call site if needed, 
# or do a large block replacement if they are close.
# They are slightly separated. I will replace the function first.



def plot_combined_radar_bar_grid(df, output_path="figures/combined_radar_bar_grid_updated.png", individual_plots_dir=None, no_radar=False):
    """
    Refactored main plotter.
    1. Pre-calculates all stats.
    2. Determines global scaling.
    3. Plots grid.
    4. Optionally saves individual plots to a folder.

    Args:
        df: DataFrame with evaluation results.
        output_path: Path for combined plot.
        individual_plots_dir: Optional directory to save individual model plots.
        no_radar: If True, skip radar plots in individual model charts.
    """
    models = sorted(df['model'].unique())
    valid_models = [m for m in models if len(df[df['model'] == m]) > 0]
    n_rows = len(valid_models)
    if n_rows == 0:
        return

    # Determine active task order from data
    active_tasks = df['question_category'].unique()
    task_order = [t for t in QUESTION_ORDER if t in active_tasks]
    if not task_order:
        print("Warning: No valid tasks found in data")
        return
    print(f"  Radar plot using {len(task_order)} tasks: {task_order}")

    # 1. Prepare Data & Stats (pass task_order)
    all_stats = {}
    global_max_tokens = 0
    
    for m in valid_models:
        stats = _prepare_model_stats(df, m, task_order=task_order)
        all_stats[m] = stats
        if stats['max_token_val'] > global_max_tokens:
            global_max_tokens = stats['max_token_val']
            
    tok_max = 1.1 * global_max_tokens if global_max_tokens > 0 else 100

    # 2. Setup Figure
    total_rows = n_rows + 1
    fig = plt.figure(figsize=(10, 3.5 * total_rows), dpi=300)
    gs = gridspec.GridSpec(
        total_rows, 2, figure=fig,
        width_ratios=[1, 1], height_ratios=[1] * total_rows,
        hspace=0.6, wspace=0.1
    )
    
    # 3. Plot Each Model
    for i, m in enumerate(valid_models):
        stats = all_stats[m]
        
        ax_bar = fig.add_subplot(gs[i, 0])
        ax_radar = fig.add_subplot(gs[i, 1], projection="polar")
        
        plot_model_pair(ax_bar, ax_radar, stats, 
                       tok_max=tok_max, show_tok_ylabel=True, 
                       bar_w=0.75)
        
        # Add titles and labels
        ax_bar.set_title(f"{m}", fontsize=10, loc='left')
        ax_bar.set_ylabel("Success Rate", fontsize=8)
        if i == n_rows - 1:
            ax_bar.set_xlabel("Thinking Budget", fontsize=8)
            
        # Grid letter label
        label_char = chr(97 + i)
        fig.text(ax_bar.get_position().x0 - 0.035, ax_bar.get_position().y1 + 0.01,
                 f"({label_char})", fontsize=10, ha="left", va="bottom")
        
        # Save individual model plot if directory specified
        if individual_plots_dir:
            if no_radar:
                ind_fig = plt.figure(figsize=(6, 4.5), dpi=300)
                ind_ax_bar = ind_fig.add_subplot(111)
                _plot_bars_from_stats(ind_ax_bar, stats,
                                     tok_max=tok_max, show_tok_ylabel=True)
            else:
                ind_fig = plt.figure(figsize=(12, 4.5), dpi=300)
                ind_gs = gridspec.GridSpec(1, 2, figure=ind_fig, width_ratios=[1, 1], wspace=0.4)
                ind_ax_bar = ind_fig.add_subplot(ind_gs[0, 0])
                ind_ax_radar = ind_fig.add_subplot(ind_gs[0, 1], projection="polar")
                plot_model_pair(ind_ax_bar, ind_ax_radar, stats,
                               tok_max=tok_max, show_tok_ylabel=True, bar_w=0.75)

            ind_ax_bar.set_title(f"{m}", fontsize=12, loc='left', fontweight='bold')
            ind_ax_bar.set_ylabel("Success Rate", fontsize=10)
            ind_ax_bar.set_xlabel("Thinking Budget", fontsize=10)

            # Sanitize model name for filename
            safe_name = m.replace("/", "_").replace(":", "_").replace(" ", "_")
            ind_dir = Path(individual_plots_dir)
            ind_dir.mkdir(parents=True, exist_ok=True)

            ind_fig.tight_layout()
            ind_fig.savefig(ind_dir / f"{safe_name}.png", bbox_inches="tight", dpi=300)
            ind_fig.savefig(ind_dir / f"{safe_name}.pdf", bbox_inches="tight")
            plt.close(ind_fig)

    # 4. Summary / Leaderboard Row (Best of Each)
    best_stats = _prepare_leaderboard_stats(df, models)
    
    lb_max = best_stats['max_token_val']
    final_tok_max = max(tok_max, 1.1 * lb_max if lb_max > 0 else 100)
    
    ax_lb_bar = fig.add_subplot(gs[n_rows, 0])
    ax_lb_radar = fig.add_subplot(gs[n_rows, 1], projection="polar")
    
    _plot_bars_from_stats(ax_lb_bar, best_stats, 
                          tok_max=final_tok_max, show_tok_ylabel=True, 
                          show_token_line=False)
    
    ax_lb_bar.set_title("Best Performance by Model (Optimal Budget)", fontsize=10, loc='left')
    ax_lb_bar.set_ylabel("Success Rate", fontsize=8)
    ax_lb_bar.tick_params(axis='x', rotation=90, labelsize=7)
    
    summary_radar_stats = _prepare_summary_radar_stats(df, best_stats, task_order=task_order)
    if summary_radar_stats:
        _plot_radar_from_stats(ax_lb_radar, summary_radar_stats)
        ax_lb_radar.set_title("All Models Comparison", fontsize=9, y=1.1)
        
    label_char = chr(97 + n_rows)
    fig.text(ax_lb_bar.get_position().x0 - 0.035, ax_lb_bar.get_position().y1 + 0.01,
             f"({label_char})", fontsize=10, ha="left", va="bottom")

    # Remove tight_layout which ignores gridspec hspace
    # Increase vertical spacing manually
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.8, wspace=0.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Save individual summary plot if directory specified
    if individual_plots_dir:
        if no_radar:
            ind_fig = plt.figure(figsize=(8, 5), dpi=300)
            ind_ax_bar = ind_fig.add_subplot(111)
        else:
            ind_fig = plt.figure(figsize=(14, 5), dpi=300)
            ind_gs = gridspec.GridSpec(1, 2, figure=ind_fig, width_ratios=[1.3, 1], wspace=0.4)
            ind_ax_bar = ind_fig.add_subplot(ind_gs[0, 0])
            ind_ax_radar = ind_fig.add_subplot(ind_gs[0, 1], projection="polar")

            if summary_radar_stats:
                _plot_radar_from_stats(ind_ax_radar, summary_radar_stats)
                ind_ax_radar.set_title("All Models Comparison", fontsize=11, y=1.1)

        _plot_bars_from_stats(ind_ax_bar, best_stats,
                              tok_max=final_tok_max, show_tok_ylabel=True,
                              show_token_line=False)

        ind_ax_bar.set_title("Best Performance by Model (Optimal Budget)", fontsize=12, loc='left', fontweight='bold')
        ind_ax_bar.set_ylabel("Success Rate", fontsize=10)
        ind_ax_bar.tick_params(axis='x', rotation=90, labelsize=7)

        ind_dir = Path(individual_plots_dir)
        ind_dir.mkdir(parents=True, exist_ok=True)

        ind_fig.tight_layout()
        ind_fig.savefig(ind_dir / "summary_leaderboard.png", bbox_inches="tight", dpi=300)
        ind_fig.savefig(ind_dir / "summary_leaderboard.pdf", bbox_inches="tight")
        plt.close(ind_fig)
        
        print(f"  Saved {len(valid_models) + 1} individual plots to {individual_plots_dir}/")


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def main():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate all model responses: combine CSVs and generate comprehensive plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --csv-dir model_responses \\
    --questions questions/chemiq.jsonl \\
    --output-dir figures

  # Evaluate only SMILES-output tasks (for Ether0 comparison)
  python evaluate.py --csv-dir model_responses \\
    --questions questions/chemiq.jsonl \\
    --tasks nmr_elucidation,reaction

Available tasks: {}
        """.format(", ".join(QUESTION_ORDER))
    )
    
    arg_parser.add_argument('--csv-dir', default='model_responses',
                           help='Directory containing CSV files (default: model_responses)')
    arg_parser.add_argument('--questions', required=True, nargs='+',
                           help='Path(s) to questions JSONL file(s)')
    arg_parser.add_argument('--output-dir', default='figures',
                           help='Directory for output plots (default: figures)')
    arg_parser.add_argument('--individual-plots-dir',
                           help='Directory to save individual model plots (optional, e.g., figures/individual)')
    arg_parser.add_argument('--tasks',
                           help=f'Comma-separated list of tasks to evaluate (default: all). '
                                f'Available: {", ".join(QUESTION_ORDER)}')
    arg_parser.add_argument('--models',
                           help='Comma-separated list of models to include (default: all). '
                                'Use model_label format, e.g., "gemini-2.5-pro-8192,o3-mini-HIGH"')
    arg_parser.add_argument('--exclude-models',
                           help='Comma-separated list of models to exclude. '
                                'Use model_label format, e.g., "gpt-4o,gemini-2.0-flash"')
    arg_parser.add_argument('--no-radar', action='store_true',
                           help='Skip radar plots in individual model charts (useful when few tasks)')
    arg_parser.add_argument('--save-processed',
                           help='Path to save processed DataFrame (optional)')
    arg_parser.add_argument('--save-results',
                           help='Base path for saving evaluation results CSVs (e.g., figures/results). '
                                'Creates: *_summary.csv, *_by_model.csv, *_by_task.csv')
    
    args = arg_parser.parse_args()
    
    # Parse tasks filter
    task_filter = None
    if args.tasks:
        task_filter = [t.strip() for t in args.tasks.split(',')]
        invalid_tasks = [t for t in task_filter if t not in QUESTION_ORDER]
        if invalid_tasks:
            print(f"Error: Invalid tasks: {invalid_tasks}")
            print(f"Available tasks: {QUESTION_ORDER}")
            return 1
        print(f"Filtering to tasks: {task_filter}")
    
    # Parse models filter
    include_models = None
    if args.models:
        include_models = [m.strip() for m in args.models.split(',')]
        print(f"Including only models: {include_models}")
    
    exclude_models = None
    if args.exclude_models:
        exclude_models = [m.strip() for m in args.exclude_models.split(',')]
        print(f"Excluding models: {exclude_models}")
    
    # Load and process data
    print(f"Scanning {args.csv_dir} for CSV files...")
    df = find_and_combine_csvs(args.csv_dir)
    df = process_results(df, args.questions)
    df = create_model_label(df)
    
    # Apply task filter if specified
    if task_filter:
        before_count = len(df)
        df = df[df['question_category'].isin(task_filter)]
        print(f"Task filter: {before_count} → {len(df)} rows ({len(df['question_category'].unique())} tasks)")
    
    # Apply model filters if specified
    if include_models:
        before_count = len(df)
        available_models = df['model_label'].unique()
        invalid_models = [m for m in include_models if m not in available_models]
        if invalid_models:
            print(f"Warning: Models not found in data: {invalid_models}")
            print(f"Available models: {sorted(available_models)}")
        df = df[df['model_label'].isin(include_models)]
        print(f"Model include filter: {before_count} → {len(df)} rows ({df['model_label'].nunique()} models)")
    
    if exclude_models:
        before_count = len(df)
        df = df[~df['model_label'].isin(exclude_models)]
        print(f"Model exclude filter: {before_count} → {len(df)} rows ({df['model_label'].nunique()} models)")
    
    if args.save_processed:
        print(f"\nSaving processed DataFrame to {args.save_processed}...")
        Path(args.save_processed).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_processed, index=False)
    
    # Generate plots
    print("\nGenerating plots...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine individual plots directory
    individual_dir = args.individual_plots_dir
    if individual_dir:
        print(f"  Individual plots will be saved to {individual_dir}/")
    
    print("  1. Combined radar/bar grid...")
    plot_combined_radar_bar_grid(df,
                                 output_path=str(Path(args.output_dir) / "combined_radar_bar_grid_updated.png"),
                                 individual_plots_dir=individual_dir,
                                 no_radar=args.no_radar)
    plot_combined_radar_bar_grid(df,
                                 output_path=str(Path(args.output_dir) / "combined_radar_bar_grid_updated.pdf"),
                                 individual_plots_dir=None,
                                 no_radar=args.no_radar)  # Only save individual plots once
    
    print(f"\nAll plots saved to {args.output_dir}/")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal responses: {len(df)}")
    print(f"Unique models: {df['model_label'].nunique()}")
    print(f"Unique questions: {df['uuid'].nunique()}")
    print(f"\nOverall success rate: {df['is_correct'].mean():.1%}")
    
    print("\nSuccess rate by model:")
    for model in sorted(df['model_label'].unique()):
        model_df = df[df['model_label'] == model]
        print(f"  {model:30s} {model_df['is_correct'].mean():6.1%} (n={len(model_df)})")
    
    print("\nSuccess rate by task category:")
    for task in QUESTION_ORDER:
        task_df = df[df['question_category'] == task]
        if len(task_df) > 0:
            print(f"  {QUESTION_LABELS.get(task, task):30s} {task_df['is_correct'].mean():6.1%} (n={len(task_df)})")
    
    print("\nSuccess rate by model and task:")
    for model in sorted(df['model_label'].unique()):
        model_df = df[df['model_label'] == model]
        print(f"\n  {model}:")
        for task in QUESTION_ORDER:
            task_df = model_df[model_df['question_category'] == task]
            if len(task_df) > 0:
                print(f"    {QUESTION_LABELS.get(task, task):30s} {task_df['is_correct'].mean():6.1%} (n={len(task_df)})")
    
    # Save results to CSV if requested
    if args.save_results:
        base_path = Path(args.save_results)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary CSV: overall statistics
        summary_data = {
            'metric': ['total_responses', 'unique_models', 'unique_questions', 'overall_success_rate_pct'],
            'value': [len(df), df['model_label'].nunique(), df['uuid'].nunique(), round(df['is_correct'].mean() * 100, 2)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{base_path}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")
        
        # 2. By-model CSV: success rate per model
        model_rows = []
        for model in sorted(df['model_label'].unique()):
            model_df = df[df['model_label'] == model]
            model_rows.append({
                'model': model,
                'success_rate_pct': round(model_df['is_correct'].mean() * 100, 2),
                'n_responses': len(model_df),
                'n_correct': int(model_df['is_correct'].sum())
            })
        model_results_df = pd.DataFrame(model_rows)
        model_path = f"{base_path}_by_model.csv"
        model_results_df.to_csv(model_path, index=False)
        print(f"Saved model results to {model_path}")
        
        # 3. By-task CSV: success rate per task (and model x task breakdown)
        # First, task-level aggregates
        task_rows = []
        for task in QUESTION_ORDER:
            task_df = df[df['question_category'] == task]
            if len(task_df) > 0:
                task_rows.append({
                    'task': task,
                    'task_label': QUESTION_LABELS.get(task, task).replace('\n', ' '),
                    'success_rate_pct': round(task_df['is_correct'].mean() * 100, 2),
                    'n_responses': len(task_df),
                    'n_correct': int(task_df['is_correct'].sum())
                })
        task_results_df = pd.DataFrame(task_rows)
        task_path = f"{base_path}_by_task.csv"
        task_results_df.to_csv(task_path, index=False)
        print(f"Saved task results to {task_path}")
        
        # 4. Model x Task matrix CSV: wide format with models as rows and tasks as columns
        # Only include tasks that are present in the data (respects --tasks filter)
        active_tasks = [t for t in QUESTION_ORDER if t in df['question_category'].unique()]
        model_task_rows = []
        for model in sorted(df['model_label'].unique()):
            model_df = df[df['model_label'] == model]
            row = {'model': model}
            for task in active_tasks:
                task_df = model_df[model_df['question_category'] == task]
                if len(task_df) > 0:
                    row[task] = round(task_df['is_correct'].mean() * 100, 2)
                else:
                    row[task] = None
            # Add overall success rate (same as bar chart: total correct / total questions)
            row['Overall'] = round(model_df['is_correct'].mean() * 100, 2)
            model_task_rows.append(row)
        model_task_df = pd.DataFrame(model_task_rows)
        model_task_path = f"{base_path}_model_task_matrix.csv"
        model_task_df.to_csv(model_task_path, index=False)
        print(f"Saved model-task matrix to {model_task_path}")
        
        # 5. Generate heatmap for model-task matrix
        heatmap_data = model_task_df.set_index('model')
        # Sort by overall success rate (already calculated as total correct / total questions)
        heatmap_data = heatmap_data.sort_values('Overall', ascending=True)
        # Drop Overall column when there's only one task (redundant)
        if len(active_tasks) == 1:
            heatmap_data = heatmap_data.drop(columns=['Overall'])
        # Rename task columns to readable labels (but keep 'Overall' as is)
        new_columns = [QUESTION_LABELS.get(c, c).replace('\n', ' ') for c in heatmap_data.columns]
        heatmap_data.columns = new_columns
        
        # Create figure with appropriate size
        n_models = len(heatmap_data)
        fig_height = max(8, 0.4 * n_models)
        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=150)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data.astype(float),
            annot=True,
            fmt='.1f',
            cmap='Blues',
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Success Rate (%)'},
            ax=ax
        )
        
        ax.set_xlabel('Task', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        ax.set_title('Model Performance Heatmap (Success Rate %)', fontsize=13, pad=15)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = f"{base_path}_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        heatmap_pdf_path = f"{base_path}_heatmap.pdf"
        plt.savefig(heatmap_pdf_path, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {heatmap_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
