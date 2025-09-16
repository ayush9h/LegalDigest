import json

import pandas as pd

df = pd.read_csv("./IPC.csv")

if "Unnamed: 0" in df.columns:
    df.drop(
        columns=["Unnamed: 0"],
        axis=1,
        inplace=True,
    )


records = []
for _, row in df.iterrows():

    instruction = f"Explain {row["IPC_Section"]}"
    parts = []
    if pd.notna(row["Offense"]):
        parts.append(f"Offense: {row['Offense']}")
    if pd.notna(row["Punishment"]):
        parts.append(f"Punishment: {row['Punishment']}")
    if pd.notna(row["Cognizable"]):
        parts.append(f"Cognizable: {row['Cognizable']}")
    if pd.notna(row["Bailable"]):
        parts.append(f"Bailable: {row['Bailable']}")
    if pd.notna(row["Court"]):
        parts.append(f"Court: {row['Court']}")

    output = str(row["Description"]).strip() + " ".join(parts)

    record = {
        "instruction": instruction,
        "input": "",
        "output": output,
    }
    records.append(record)

with open("ipc.jsonl", "w", encoding="utf-8") as jsonl_file:
    for record in records:
        jsonl_file.write(json.dumps(record) + "\n")

print("Data converted to jsonl file")
