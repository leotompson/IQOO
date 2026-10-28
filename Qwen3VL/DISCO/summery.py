import re
from pathlib import Path

import pandas as pd


DATASET_ORDER = [
    "ImageNet-R",
    "ArxivQA",
    "VizWiz",
    "IconQA",
    "CLEVR-Math",
    "Flickr30k",
]
TASKS = [f"task{i}" for i in range(1, 7)]


def parse_result_text(result_file: Path):
    """Read accuracy value from Result.text.

    Returns float (e.g. 90.7) when found, otherwise None.
    """
    if not result_file.exists():
        return None

    text = result_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None

    # Match classification format: Accuracy: 90.70%
    m = re.search(r"Accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if m:
        return float(m.group(1))

    # Match captioning format (e.g. Flickr30k): Average: 55.93
    m = re.search(r"Average\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    return None


def build_summary(base_dir: Path) -> pd.DataFrame:
    # rows: task1~task6, cols: dataset order user requested
    df = pd.DataFrame(index=TASKS, columns=DATASET_ORDER)

    for dataset in DATASET_ORDER:
        for i, task in enumerate(TASKS, start=1):
            result_file = base_dir / dataset / f"DISCO-task{i}" / "Result.text"
            acc = parse_result_text(result_file)
            if acc is not None:
                df.loc[task, dataset] = acc

    return df


def main():
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / "results" / "UCIT" / "each_dataset"

    summary_df = build_summary(base_dir)

    output_csv = base_dir / "DISCO_UCIT_summary.csv"

    # Keep empty cells empty in excel/csv
    summary_df.to_csv(output_csv, index=True, index_label="Task")

    print(f"Saved: {output_csv}")

def compute():
    """Compute continual-learning metrics from DISCO_UCIT_summary.csv.

    Metrics:
    - MFT: Mean Finetune Accuracy (mean diagonal)
    - MFN: Mean Final Accuracy (mean of last row over all datasets)
    - MAA: Mean Average Accuracy (mean over each step's average seen-task accuracy)
    - BWT: Backward Transfer (mean(final - diagonal) over old tasks)
    """
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / "results" / "UCIT" / "each_dataset"
    summary_csv = base_dir / "DISCO_UCIT_summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_csv}")

    # index: task1~task6, columns: DATASET_ORDER
    df = pd.read_csv(summary_csv, index_col=0)

    # Ensure numeric values (empty cells -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    num_tasks = min(len(TASKS), len(DATASET_ORDER), len(df.index), len(df.columns))

    # Build square task-dataset matrix by the predefined order
    mat = df.reindex(index=TASKS, columns=DATASET_ORDER).iloc[:num_tasks, :num_tasks]

    # Diagonal: performance right after finetuning each task
    diag_vals = [mat.iloc[i, i] for i in range(num_tasks)]
    diag_series = pd.Series(diag_vals, dtype="float64")

    # MFT: mean diagonal
    mft = diag_series.mean(skipna=True)

    # MFN: mean performance at final step (last row) on all tasks/datasets
    final_row = mat.iloc[num_tasks - 1, :]
    mfn = final_row.mean(skipna=True)

    # MAA: average over steps of the average seen-task accuracy at each step
    step_means = []
    for t in range(num_tasks):
        seen = mat.iloc[t, : t + 1]
        step_means.append(seen.mean(skipna=True))
    maa = pd.Series(step_means, dtype="float64").mean(skipna=True)

    # BWT: final performance minus immediate post-training performance (excluding last task)
    if num_tasks > 1:
        bwt_terms = []
        for j in range(num_tasks - 1):
            a_final_j = mat.iloc[num_tasks - 1, j]
            a_diag_j = mat.iloc[j, j]
            bwt_terms.append(a_final_j - a_diag_j)
        bwt = pd.Series(bwt_terms, dtype="float64").mean(skipna=True)
    else:
        bwt = float("nan")

    metrics = {
        "MFT": None if pd.isna(mft) else float(mft),
        "MFN": None if pd.isna(mfn) else float(mfn),
        "MAA": None if pd.isna(maa) else float(maa),
        "BWT": None if pd.isna(bwt) else float(bwt),
    }

    print("Computed metrics from:", summary_csv)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    # main()
    compute()
