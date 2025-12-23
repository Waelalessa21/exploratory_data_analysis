import pandas as pd
from pandas import DataFrame
import re

_ws = re.compile(r"\s+")

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return(
        df.isna()
        .sum()
        .rename("n_missing")
        .to_frame()
        .assign(p_missing=lambda t: t["n_missing"] / len(df))
        .sort_values("p_missing", ascending=False)
    )


def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}__isna"] = out[c].isna()
    return out



def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        order_id=df["order_id"].astype("string"),
        user_id=df["user_id"].astype("string"),
        amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64"),
        quantity=pd.to_numeric(df["quantity"], errors="coerce").astype("Int64"),
    )


def normalize_text(s: pd.Series) -> pd.Series:
    return(
        s.astype("string")
        .str.lower()
        .str.replace(_ws, " ", regex=True)
        .str.strip()
        .str.casefold()
    )


def apply_mapping(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return s.map(lambda x: mapping.get(x, x))



def remove_duplicates(df: pd.DataFrame, key_cols: list[str], ts_col: str) -> pd.DataFrame:
    return(
        df.sort_values(ts_col)
        .drop_duplicates(subset=key_cols, keep="last")
        .reset_index(drop=True)
    )