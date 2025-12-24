import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exploratory_data_analysis.config import make_paths
from src.exploratory_data_analysis.quality import (
    check_required_col,
    assert_non_empty,
    assert_unique_key,
)
from src.exploratory_data_analysis.transforms import (
    parse_datetime,
    add_time_parts,
    winsorize,
    add_outlier_flag,
)
from src.exploratory_data_analysis.join import safe_left_join

def main() -> None:
    p = make_paths(ROOT)

    orders = pd.read_parquet(p.processed / "orders_clean.parquet")
    users  = pd.read_parquet(p.processed / "users.parquet")

    check_required_col(orders, ["order_id","user_id","amount","quantity","created_at","status_clean"])
    check_required_col(users, ["user_id","country","signup_date"])
    assert_non_empty(orders, "orders_clean")
    assert_non_empty(users, "users")
    assert_unique_key(users, "user_id")

    orders_t = orders.pipe(parse_datetime, col="created_at", utc=True).pipe(add_time_parts, col="created_at")

    n_missing_ts = int(orders_t["created_at"].isna().sum())
    print("missing created_at after parse:", n_missing_ts, "/", len(orders_t))

    joined = safe_left_join(
        orders_t,
        users,
        on="user_id",
        validate="many_to_one",
        suffixes=("", "_user"),
    )

    assert len(joined) == len(orders_t), "Row count changed، is join explosion?"

    match_rate = 1.0 - float(joined["country"].isna().mean())
    print("rows:", len(joined))
    print("country match rate:", round(match_rate, 3))

    joined = joined.assign(amount_winsor=winsorize(joined["amount"]))
    joined = add_outlier_flag(joined, "amount", k=1.5)

    out_path = p.processed / "analytics_table.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_path, index=False)
    print("wrote:", out_path)

if __name__ == "__main__":
    main()
