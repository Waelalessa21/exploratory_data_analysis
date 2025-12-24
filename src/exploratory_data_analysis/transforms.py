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

def parse_datetime(df: pd.DataFrame, col: str, *, utc: bool = True) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], utc=utc)
    return df.assign(**{col: dt})


def add_time_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ts = df[col]
    return df.assign(
        date=ts.dt.date,
        year=ts.dt.year,
        month=ts.dt.to_period("M").astype("string"),
        dow=ts.dt.day_name(),
        hour=ts.dt.hour,
    )




def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    x = s.dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)

def add_outlier_flag(df: pd.DataFrame, col: str, *, k: float = 1.5) -> pd.DataFrame:
    lo, hi = iqr_bounds(df[col], k=k)
    return df.assign(**{f"{col}__is_outlier": (df[col] < lo) | (df[col] > hi)})


def winsorize(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=a, upper=b)