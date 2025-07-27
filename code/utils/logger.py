import csv, datetime
from pathlib import Path

class CSVLogger:
    """Simple CSV appender. Creates header if file does not exist."""
    def __init__(self, csv_path: str | Path, fieldnames: list[str]):
        self.path = Path(csv_path)
        self.fieldnames = fieldnames + ["timestamp"]
        self._first = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **row):
        # auto‑extend header with new keys
        for k in row:
            if k not in self.fieldnames:
                self.fieldnames.insert(-1, k)
        row["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self._first:
                w.writeheader(); self._first = False
            w.writerow(row)