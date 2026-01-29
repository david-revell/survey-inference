import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _greedy_match(ref_topics: List[List[str]], run_topics: List[List[str]]) -> Tuple[float, List[Tuple[int, int, float]]]:
    pairs = []
    used_ref = set()
    used_run = set()

    # Build all candidate scores
    scores = []
    for i, rt in enumerate(ref_topics):
        for j, tt in enumerate(run_topics):
            scores.append((i, j, _jaccard(rt, tt)))
    # Sort by score desc
    scores.sort(key=lambda x: x[2], reverse=True)

    total = 0.0
    matched = 0
    for i, j, s in scores:
        if i in used_ref or j in used_run:
            continue
        used_ref.add(i)
        used_run.add(j)
        pairs.append((i, j, s))
        total += s
        matched += 1
        if matched == min(len(ref_topics), len(run_topics)):
            break

    avg = total / matched if matched else 0.0
    return avg, pairs


def analyze(input_path: Path, output_path: Path) -> None:
    runs = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            runs.append(json.loads(line))

    if not runs:
        raise ValueError("No runs found.")

    ref = runs[0]
    ref_topics = ref["topic_words"]
    ref_prev = np.array(ref["topic_prevalence"], dtype=float)

    rows = []
    for r in runs:
        run_topics = r["topic_words"]
        run_prev = np.array(r["topic_prevalence"], dtype=float)

        avg_jaccard, pairs = _greedy_match(ref_topics, run_topics)

        # Align prevalence by matched pairs
        aligned_ref = []
        aligned_run = []
        for i, j, _ in pairs:
            aligned_ref.append(ref_prev[i])
            aligned_run.append(run_prev[j])
        if aligned_ref:
            aligned_ref = np.array(aligned_ref)
            aligned_run = np.array(aligned_run)
            l1 = float(np.abs(aligned_ref - aligned_run).mean())
        else:
            l1 = 0.0

        rows.append(
            {
                "run_id": r["run_id"],
                "sample_size": r["sample_size"],
                "avg_topic_jaccard": avg_jaccard,
                "avg_prevalence_l1": l1,
            }
        )

    df = pd.DataFrame(rows).sort_values("run_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("=== Stability Summary ===")
    print(df[["avg_topic_jaccard", "avg_prevalence_l1"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze resampling stability.")
    parser.add_argument("--input", default="results/resampling_runs.jsonl")
    parser.add_argument("--output", default="results/stability_summary.csv")
    args = parser.parse_args()

    analyze(Path(args.input), Path(args.output))
