import pandas as pd


def quick_profile(csv_path, max_cols: int = 60) -> str:
    """Return a short text summary of a CSV that an LLM can read.

    We send this instead of the raw data: it is small, cheap, and does not
    leak the actual rows.
    """
    df = pd.read_csv(csv_path)

    lines = [
        f"Rows: {len(df)}, Columns: {df.shape[1]}",
        "",
        "Columns (name | dtype | missing values | unique values | example):",
    ]
    for col in df.columns[:max_cols]:
        s = df[col]
        example = s.dropna().iloc[0] if s.notna().any() else None
        lines.append(f"- {col} | {s.dtype} | {s.isna().sum()} | {s.nunique()} | {example}")

    if df.shape[1] > max_cols:
        lines.append(f"... and {df.shape[1] - max_cols} more columns not shown")

    return "\n".join(lines)
