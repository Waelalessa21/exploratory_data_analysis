from pathlib import Path
import pandas as pd
from pandas import DataFrame

NA = ["", "NA", "N/A", "null", "None", "not_a_number"]

def read_orders_csv(path: Path) -> DataFrame:
    return pd.read_csv(
        path,
        dtype={"order_id": "string", "user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )

def read_users_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )


def write_parquet(df: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> DataFrame:
    ...


