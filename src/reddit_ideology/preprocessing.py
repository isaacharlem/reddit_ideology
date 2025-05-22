import re
import pandas as pd


class Preprocessor:
    def __init__(self):
        # regex to remove URLs
        self.url_pattern = re.compile(r"http\S+")
        # regex for markdown links
        self.markdown_link = re.compile(r"\[.*?\]\(.*?\)")
        # regex to remove non-word characters
        self.punct_pattern = re.compile(r"[^\w\s']+")

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self.url_pattern.sub("", text)
        text = self.markdown_link.sub("", text)
        text = self.punct_pattern.sub("", text)
        return text.strip()

    def apply(self, df) -> pd.DataFrame:
        df["clean_text"] = df["text"].apply(self.clean)
        return df
