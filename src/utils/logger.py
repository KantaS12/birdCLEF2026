"""CSV-based experiment logger."""
import csv
from pathlib import Path
from typing import Any

class CSVLogger:
    """Appends metric dicts as rows to a CSV file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fieldnames: list[str] | None = None
        self._file = None
        self._writer = None

    def log(self, metrics: dict[str, Any]) -> None:
        """Append a row of metrics."""
        if self._writer is None:
            self._fieldnames = list(metrics.keys())
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
