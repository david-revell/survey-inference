# Survey Inference

Short experiments to test *how stable* qualitative themes and labels are under resampling and different modelling choices.

## First study: theme stability under resampling
**Goal:** quantify stability of topics on gym review data with repeated random subsamples.

**Data:**
- `data/Google_12_months.xlsx` (uses column `Comment`)
- `data/Trustpilot_12_months.xlsx` (uses `Review Title` + `Review Content`)

## Scripts
- `scripts/run_resampling.py`  
  Runs LDA on repeated samples and saves per-run topic words + prevalence.
- `scripts/analyze_stability.py`  
  Computes topic overlap (Jaccard) and prevalence drift (L1) against a baseline run.

## Quick start
```powershell
# from the project root
python scripts\run_resampling.py --input data\Google_12_months.xlsx --text-col "Comment" --sample-frac 0.7 --n-runs 10 --n-topics 10
python scripts\analyze_stability.py --input results\resampling_runs.jsonl --output results\stability_summary.csv
```

Trustpilot example:
```powershell
python scripts\run_resampling.py --input data\Trustpilot_12_months.xlsx --text-col "Review Content" --title-col "Review Title" --sample-frac 0.7 --n-runs 10 --n-topics 10
python scripts\analyze_stability.py --input results\resampling_runs.jsonl --output results\stability_summary.csv
```

## Notes
- This version uses LDA (no LLMs). It is fast and cheap to run.
- A future extension can swap in local LLM labeling after topics are formed.
