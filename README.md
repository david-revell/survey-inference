# Survey Inference

**Status:** In progress (active)  
**Scope:** Statistical stability analysis only. No production deployment or automated decisioning.

Short experiments to test *how stable* qualitative themes and labels are under resampling and different modelling choices.

## First study: theme stability under resampling
**Goal:** quantify stability of topics on gym review data with repeated random subsamples.

**Data:**
- `data/Google_12_months.xlsx` (uses column `Comment`)
- `data/Trustpilot_12_months.xlsx` (uses column `Review Content`; title excluded by default)

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

Optional: add a custom neutral-terms stopword list (merged with English stopwords):
```powershell
python scripts\run_resampling.py --input data\Google_12_months.xlsx --text-col "Comment" --sample-frac 0.7 --n-runs 10 --n-topics 10 --stopwords-file config\neutral_terms.txt
```

Trustpilot example:
```powershell
python scripts\run_resampling.py --input data\Trustpilot_12_months.xlsx --text-col "Review Content" --sample-frac 0.7 --n-runs 10 --n-topics 10
python scripts\analyze_stability.py --input results\resampling_runs.jsonl --output results\stability_summary.csv
```

## Notes
- This version uses LDA (no LLMs). It is fast and cheap to run.
- A future extension can swap in local LLM labeling after topics are formed.
- For Trustpilot, we exclude the title field by default to avoid overweighting short, high-salience text and to keep preprocessing consistent across datasets. A future sensitivity check can append titles and compare stability.

## Roadmap (longer-term)
1. Batching and sampling controls (cost vs stability trade-offs)
2. Staged inference (coarse pass -> focused pass)
3. Inference probes:
   3.1 Bayesian interpretations vs token probabilities
   3.2 Theory vs behaviour once models are deployed
   3.3 Loss of nice properties at scale
   3.4 Ongoing research directions
4. Bayesian expectations vs deterministic outputs (are LLMs doing Bayesian reasoning, or just pattern-matching with probabilities?)
