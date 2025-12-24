import sys
import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exploratory_data_analysis.config import make_paths
from src.exploratory_data_analysis.io import read_orders_csv, read_users_csv, write_parquet
from src.exploratory_data_analysis.transforms import (
    missingness_report,
    add_missing_flags,
    normalize_text,
    apply_mapping,
    remove_duplicates,
    enforce_schema,
)
from src.exploratory_data_analysis.quality import (
    check_required_col,
    assert_non_empty,
    assert_unique_key,
    assert_in_range,
)

log = logging.getLogger(__name__)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = make_paths(ROOT)

    log.info("Here is loading the raw inputs")
    orders_raw = read_orders_csv(p.raw / "orders.csv")
    users_raw = read_users_csv(p.raw / "users.csv")
    log.info("Rows: orders_raw=%s", len(orders_raw))
    log.info("Rows: users_raw=%s", len(users_raw))

    check_required_col(orders_raw, ["order_id","user_id","amount","quantity","created_at","status"])
    check_required_col(users_raw, ["user_id","country","signup_date"])
    assert_non_empty(orders_raw, "orders_raw")
    assert_non_empty(users_raw, "users_raw")

    orders_raw = remove_duplicates(orders_raw, key_cols=["order_id"], ts_col="created_at")
    users_raw = remove_duplicates(users_raw, key_cols=["user_id"], ts_col="signup_date")
    assert_unique_key(orders_raw, "order_id")
    assert_unique_key(users_raw, "user_id")

    orders = enforce_schema(orders_raw)
    users = users_raw.copy()
    rep = missingness_report(orders)
    rep_users = missingness_report(users)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rep_path = reports_dir / "missingness_orders.csv"
    rep.to_csv(rep_path, index=True)
    log.info("Wrote missingness report: %s", rep_path)

    status_norm = normalize_text(orders["status"])
    mapping = {"paid": "paid", "refund": "refund", "refunded": "refund"}
    status_clean = apply_mapping(status_norm, mapping)
    orders_clean = (
        orders.assign(status_clean=status_clean)
              .pipe(add_missing_flags, cols=["amount", "quantity"])
    )

    assert_in_range(orders_clean["amount"], 0, 1000000, "amount")
    assert_in_range(orders_clean["quantity"], 1, 1000, "quantity")

    write_parquet(orders_clean, p.processed / "orders_clean.parquet")
    write_parquet(users, p.processed / "users.parquet")
    log.info("Wrote processed outputs: %s", p.processed)

if __name__ == "__main__":
    main()
