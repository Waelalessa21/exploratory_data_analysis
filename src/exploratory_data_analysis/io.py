from pathlib import Path
import pandas as pd
from pandas import DataFrame

NA = ["", "NA", "N/A", "null", "None", "not_a_number"]

def read_orders_csv(path: str = "../../data/orders.csv") -> DataFrame:
    return pd.read_csv(
        path,
        dtype={"order_id": "string", "user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )

def read_users_csv(path: Path) -> DataFrame:
    ...

def write_parquet(path: Path, df: DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print("printeddd")

def read_parquet(path: Path) -> DataFrame:
    ...

if __name__ == "__main__":
    orders_df = read_orders_csv()

    write_parquet(Path("../../data/orders.parquet"), orders_df)

