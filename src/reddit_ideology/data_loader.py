import json
import pandas as pd
import os
from datetime import datetime


class DataLoader:
    def __init__(self, cons_path: str, lib_path: str):
        self.cons_path = cons_path
        self.lib_path = lib_path

    def load(self):
        cons_df = self._load_file(self.cons_path, "conservative")
        lib_df = self._load_file(self.lib_path, "liberal")
        return cons_df, lib_df

    def _load_file(self, path: str, subreddit: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        records = []
        for ts, text in data.items():
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                records.append(
                    {"timestamp": dt, "text": text, "subreddit": subreddit}
                )
            except ValueError:
                continue
        return pd.DataFrame(records)
