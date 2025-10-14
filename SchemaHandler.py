import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class SchemaHandler:
    """
    Manages info_schema.json for tracking processed/chunked/embedded PDFs.
    Provides load, update, diff, and status marking utilities.
    """

    def __init__(self, schema_path: str = "info_schema.json"):
        self.schema_path = Path(schema_path)
        self.df = self._load_or_create()

    # ---------- Core ----------
    def _load_or_create(self) -> pd.DataFrame:
        if self.schema_path.exists():
            df = pd.read_json(self.schema_path)

            # Normalize column types to prevent dtype conflicts
            dtype_map = {
                "filename": str,
                "path": str,
                "title": str,
                "author": str,
                "year": str,           # <— force to string
                "type": str,
                "num_chunks": int,     # keep numeric
                "last_modified": str,
                "status": str,
            }

            for col, dtype in dtype_map.items():
                if col not in df.columns:
                    df[col] = pd.Series(dtype=dtype)
                else:
                    try:
                        if dtype == int:
                            df[col] = df[col].fillna(0).astype(int)
                        else:
                            df[col] = df[col].astype(str)
                    except Exception:
                        # fallback in case of mixed data
                        df[col] = df[col].astype(object).astype(str)

            return df
        else:
            print(f"[SchemaManager] No schema found. Creating new: {self.schema_path}")
            return pd.DataFrame(columns=[
                "filename", "path", "title", "author", "year",
                "type", "num_chunks", "last_modified", "status"
            ])


    def save(self):
        self.df.to_json(self.schema_path, indent=4, orient="records")
        print(f"[SchemaManager] Saved schema to {self.schema_path}")

    # ---------- Add or Update ----------
    def update_entry(self, info: dict, status: str = "processed"):
        """
        Add or update an entry from a processed file dict.
        """
        filename = info.get("new_filename") or info.get("filename")
        existing = self.df[self.df["filename"] == filename]

        # Normalize all values to string where appropriate
        entry = {
            "filename": str(filename),
            "path": str(info.get("orig_path", "")),
            "title": str(info.get("title", "")),
            "author": str(info.get("author", "")),
            "year": str(info.get("year", "")),      # <— convert explicitly to str
            "type": str(info.get("type", "")),
            "num_chunks": int(info.get("num_chunks", 0)),  # keep numeric
            "last_modified": datetime.now().isoformat(),
            "status": str(status),
        }

        if not existing.empty:
            # Ensure correct dtype before assignment
            for k, v in entry.items():
                if k in self.df.columns:
                    if self.df[k].dtype == "int64" and isinstance(v, str) and v.isdigit():
                        self.df.loc[self.df["filename"] == filename, k] = int(v)
                    else:
                        # force convert to string if column dtype is object or string
                        if self.df[k].dtype == "object" or self.df[k].dtype == "string":
                            v = str(v)
                        self.df.loc[self.df["filename"] == filename, k] = v
        else:
            self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)


    # ---------- Diffs ----------
    def get_new_or_modified(self, old_schema_path: str):
        """
        Compare this schema with an older version and return new or updated rows.
        """
        if not Path(old_schema_path).exists():
            print("[SchemaManager] Old schema not found; returning all entries.")
            return self.df
        old_df = pd.read_json(old_schema_path)
        merged = self.df.merge(old_df, on="filename", how="left", suffixes=("", "_old"))
        changed = merged[
            merged["num_chunks"] != merged["num_chunks_old"]
        ].dropna(subset=["filename"])
        return changed[self.df.columns]

    # ---------- Status helpers ----------
    def mark_embedded(self, filename: str):
        self.df.loc[self.df["filename"] == filename, "status"] = "embedded"
        self.df.loc[self.df["filename"] == filename, "last_modified"] = datetime.now().isoformat()

    def get_pending_embeddings(self) -> pd.DataFrame:
        return self.df[self.df["status"] != "embedded"]

