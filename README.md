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

Quick POC run (fast sanity check):
```powershell
python scripts\run_resampling.py --input data\Google_12_months.xlsx --text-col "Comment" --sample-frac 0.3 --n-runs 3 --n-topics 6 --max-features 2000 --min-df 10
```

Runtime knobs (lower = faster):
- `--n-runs` (fewer repeated samples)
- `--sample-frac` (smaller fraction of rows per run)
- `--n-topics` (fewer themes to find)
- `--max-features` (smaller vocabulary)
- `--min-df` (higher threshold drops more rare words)

Trustpilot example:
```powershell
python scripts\run_resampling.py --input data\Trustpilot_12_months.xlsx --text-col "Review Content" --sample-frac 0.7 --n-runs 10 --n-topics 10
python scripts\analyze_stability.py --input results\resampling_runs.jsonl --output results\stability_summary.csv
```

## How to interpret the stability output
`scripts/analyze_stability.py` writes `results/stability_summary.csv`. The key ideas:
- **Topic overlap (Jaccard):** higher means the same themes show up again when you resample.
- **Prevalence drift (L1):** lower means the theme sizes stay similar across runs.

Quick reading guide:
- **High overlap + low drift** = stable themes.
- **Low overlap + high drift** = unstable themes.
- **Middle values** = try small adjustments and rerun.

If results look unstable, try:
- Fewer topics (`--n-topics`)
- Larger sample (`--sample-frac`)
- More runs (`--n-runs`)
- Cleaner vocabulary (`--stopwords-file` or higher `--min-df`)

## Sample output (tiny)
Example single line from `results/resampling_runs.jsonl` (structure example only; values will vary):
```json
{"run_id": 0, "seed": 42, "sample_size": 4169, "n_topics": 6, "top_words": 10, "topic_words": [["og", "er", "und", "der", "die", "man", "det", "ich", "ist", "das"], ["gym", "great", "staff", "friendly", "best", "clean", "pure", "really", "helpful", "classes"], ["good", "gym", "equipment", "clean", "great", "nice", "place", "busy", "staff", "plenty"], ["class", "really", "good", "machines", "gym", "classes", "busy", "great", "music", "workout"], ["classes", "great", "gym", "love", "fitness", "class", "24", "friendly", "fantastic", "amazing"], ["gym", "people", "equipment", "just", "machines", "like", "staff", "time", "don", "use"]], "topic_prevalence": [0.07963540417366274, 0.1772607339889662, 0.22523386903334133, 0.09450707603741905, 0.13216598704725355, 0.2911969297193572], "text_col": "Comment", "title_col": null, "sample_frac": 0.3, "max_features": 2000, "min_df": 10, "stopwords_file": null}
```

Example `results/stability_summary.csv` from a quick POC run (values will vary by run):
```csv
run_id,sample_size,avg_topic_jaccard,avg_prevalence_l1
0,4169,1.0,0.0
1,4169,0.45730487835750994,0.07539777724474293
2,4169,0.48455540560803717,0.07915567282321899
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
