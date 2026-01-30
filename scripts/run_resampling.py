import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _auto_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "comment",
        "review content",
        "review",
        "text",
        "response",
        "body",
    ]
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    # Fallback: longest average string length among object columns
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError("No text-like columns found.")
    avg_len = {}
    for c in obj_cols:
        series = df[c].dropna().astype(str)
        avg_len[c] = series.str.len().mean() if len(series) else 0
    return max(avg_len, key=avg_len.get)


def _build_text(df: pd.DataFrame, text_col: str, title_col: str | None) -> pd.Series:
    text = df[text_col].astype(str)
    if title_col:
        title = df[title_col].astype(str)
        text = (title.str.strip() + " - " + text.str.strip()).str.strip()
    return text


def run_resampling(
    input_path: Path,
    text_col: str | None,
    title_col: str | None,
    sample_frac: float,
    n_runs: int,
    n_topics: int,
    top_words: int,
    max_features: int,
    min_df: int,
    random_seed: int,
    stopwords_file: Path | None,
    output_path: Path,
) -> None:
    df = _load_table(input_path)
    if text_col is None:
        text_col = _auto_text_column(df)
    text = _build_text(df, text_col, title_col)
    text = text.replace("nan", "").replace("None", "").str.strip()
    text = text[text.str.len() > 0]

    stop_words = None
    if stopwords_file is not None:
        raw = stopwords_file.read_text(encoding="utf-8").splitlines()
        extra = [
            w.strip().lower()
            for w in raw
            if w.strip() and not w.strip().startswith("#")
        ]
        stop_words = sorted(set(ENGLISH_STOP_WORDS).union(extra))

    rows = []
    for i in range(n_runs):
        seed = random_seed + i
        sample = text.sample(frac=sample_frac, random_state=seed)

        vectorizer = CountVectorizer(
            stop_words=stop_words or "english",
            max_features=max_features,
            min_df=min_df,
        )
        X = vectorizer.fit_transform(sample)

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=seed,
            learning_method="batch",
        )
        doc_topic = lda.fit_transform(X)

        vocab = np.array(vectorizer.get_feature_names_out())
        topic_words = []
        for topic_idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[::-1][:top_words]
            topic_words.append(vocab[top_idx].tolist())

        # Assign each doc to dominant topic
        dominant = doc_topic.argmax(axis=1)
        counts = np.bincount(dominant, minlength=n_topics)
        prevalence = (counts / counts.sum()).tolist() if counts.sum() else [0] * n_topics

        rows.append(
            {
                "run_id": i,
                "seed": seed,
                "sample_size": int(len(sample)),
                "n_topics": n_topics,
                "top_words": top_words,
                "topic_words": topic_words,
                "topic_prevalence": prevalence,
                "text_col": text_col,
                "title_col": title_col,
                "sample_frac": sample_frac,
                "max_features": max_features,
                "min_df": min_df,
                "stopwords_file": str(stopwords_file) if stopwords_file else None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topic resampling with LDA.")
    parser.add_argument("--input", required=True, help="Path to input CSV/XLSX")
    parser.add_argument("--text-col", default=None, help="Text column name")
    parser.add_argument("--title-col", default=None, help="Optional title column name")
    parser.add_argument("--sample-frac", type=float, default=0.7)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--top-words", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stopwords-file",
        default=None,
        help="Optional path to newline-delimited stopwords (merged with English stopwords).",
    )
    parser.add_argument("--output", default="results/resampling_runs.jsonl")
    args = parser.parse_args()

    run_resampling(
        input_path=Path(args.input),
        text_col=args.text_col,
        title_col=args.title_col,
        sample_frac=args.sample_frac,
        n_runs=args.n_runs,
        n_topics=args.n_topics,
        top_words=args.top_words,
        max_features=args.max_features,
        min_df=args.min_df,
        random_seed=args.seed,
        stopwords_file=Path(args.stopwords_file) if args.stopwords_file else None,
        output_path=Path(args.output),
    )
