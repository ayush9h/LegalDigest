import json
from pathlib import Path

import pandas as pd

from legal_digest.config import load_config
from legal_digest.utils.logger import setup_logger

logger = setup_logger(
    name="preprocess",
    log_dir="logs",
)

EXPECTED_COLUMNS = {
    "IPC_Section",
    "Description",
    "Offense",
    "Punishment",
    "Cognizable",
    "Bailable",
    "Court",
}


def clean(text) -> str:
    return " ".join(str(text).strip().split())


def _validate_df(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in IPC.csv: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def run_preprocess() -> None:
    cfg = load_config("flan_t5")

    csv_path = Path(cfg.data.ipc_csv)
    out_path = Path(cfg.data.processed_train)

    logger.info("Starting preprocessing")
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Output path: {out_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at path: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        logger.info("Dropped 'Unnamed: 0' column")

    _validate_df(df)

    records = []

    for _, row in df.iterrows():
        instruction = clean(row["IPC_Section"])

        parts = []
        if pd.notna(row["Offense"]):
            parts.append(f"Offense: {clean(row['Offense'])}")
        if pd.notna(row["Punishment"]):
            parts.append(f"Punishment: {clean(row['Punishment'])}")
        if pd.notna(row["Cognizable"]):
            parts.append(f"Cognizable: {clean(row['Cognizable'])}")
        if pd.notna(row["Bailable"]):
            parts.append(f"Bailable: {clean(row['Bailable'])}")
        if pd.notna(row["Court"]):
            parts.append(f"Court: {clean(row['Court'])}")

        description = clean(row["Description"])
        output_text = (description + " " + " ".join(parts)).strip()

        record = {
            "prompt": instruction,
            "target": output_text,
        }
        records.append(record)

    if not records:
        logger.error("No records generated from the IPC.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Preprocessing complete")
    logger.info(f"Samples written: {len(records)}")


def main() -> None:
    try:
        run_preprocess()
    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
