import json
from pathlib import Path


class LegalDataset:
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.path}\n"
                f"Run preprocessing first."
            )

        self.samples = self._load()

    def _load(self):
        data = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
