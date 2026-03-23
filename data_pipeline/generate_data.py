"""generate_data.py — synthetic transaction data generator.

Supports both:
1) single file generation with tunable anomaly rates, and
2) scenario suite generation for broader pipeline testing.
"""

import argparse
import logging
import random
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

fake = Faker()
random.seed(42)
Faker.seed(42)

CURRENCIES = ["USD", "EUR", "INR", "GBP"]
INVALID_CURRENCIES = ["USDX", "BTC", "123", "??"]
ACCOUNTS = [f"ACC_{i:03d}" for i in range(1, 51)]  # ACC_001 … ACC_050

# ── Dirty timestamp pool ──────────────────────────────────────────────────────
_CLEAN_WITH_TZ = [
    "2024-01-15T10:23:45+05:30",
    "2024-03-22T08:00:00-04:00",
    "2024-06-01T00:00:00+00:00",
]
_CLEAN_PLAIN = [
    "2024-02-10 14:35:22",
    "2024-04-18 09:12:00",
    "2024-07-04 17:45:59",
]
_MALFORMED = [
    "15-01-2024",          # DD-MM-YYYY
    "not-a-date",
    "2024/13/01 25:61:00", # impossible month/time
    "",
    "NULL",
]


def _dirty_timestamp(
    malformed_rate: float = 0.10,
    tz_rate: float = 0.60,
) -> str:
    """Return one timestamp string from the dirty pool.

    Roughly 60 % clean-with-tz, 30 % plain string, 10 % malformed.
    """
    malformed_rate = max(0.0, min(1.0, malformed_rate))
    tz_rate = max(0.0, min(1.0 - malformed_rate, tz_rate))

    roll = random.random()
    if roll < tz_rate:
        return random.choice(_CLEAN_WITH_TZ)
    if roll < (1.0 - malformed_rate):
        return random.choice(_CLEAN_PLAIN)
    return random.choice(_MALFORMED)


def _nullable(value: object, null_rate: float) -> object:
    """Return *value* or ``None`` based on *null_rate* probability.

    Args:
        value: The original value.
        null_rate: Fraction of rows that should be null (0–1).

    Returns:
        ``None`` with probability *null_rate*, otherwise *value*.
    """
    return None if random.random() < null_rate else value


def build_dataframe(
    n_rows: int,
    *,
    null_amount_rate: float = 0.03,
    null_currency_rate: float = 0.02,
    null_narration_rate: float = 0.02,
    malformed_ts_rate: float = 0.10,
    duplicate_txn_rate: float = 0.00,
    invalid_currency_rate: float = 0.00,
) -> pd.DataFrame:
    """Build a DataFrame of *n_rows* synthetic transaction records.

    Args:
        n_rows: Number of rows to generate.

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``txn_id``, ``account_id``, ``txn_ts``, ``amount``,
        ``currency``, ``narration``.
    """
    log.info("Generating %d rows …", n_rows)

    txn_ids = [str(uuid.uuid4()) for _ in range(n_rows)]
    # Duplicate some txn_ids to stress idempotency + dedup logic.
    duplicate_count = int(max(0.0, min(1.0, duplicate_txn_rate)) * n_rows)
    for _ in range(duplicate_count):
        src_idx = random.randrange(0, n_rows)
        dst_idx = random.randrange(0, n_rows)
        txn_ids[dst_idx] = txn_ids[src_idx]

    account_ids = [random.choice(ACCOUNTS) for _ in range(n_rows)]
    txn_ts = [
        _dirty_timestamp(malformed_rate=malformed_ts_rate) for _ in range(n_rows)
    ]
    amounts = [
        _nullable(round(random.uniform(-500.0, 10_000.0), 2), null_rate=null_amount_rate)
        for _ in range(n_rows)
    ]
    currencies: list[Any] = []
    for _ in range(n_rows):
        base = random.choice(CURRENCIES)
        if random.random() < max(0.0, min(1.0, invalid_currency_rate)):
            base = random.choice(INVALID_CURRENCIES)
        currencies.append(_nullable(base, null_rate=null_currency_rate))

    narrations = [
        _nullable(
            fake.sentence(nb_words=random.randint(4, 10)).rstrip("."),
            null_rate=null_narration_rate,
        )
        for _ in range(n_rows)
    ]

    df = pd.DataFrame(
        {
            "txn_id": txn_ids,
            "account_id": account_ids,
            "txn_ts": txn_ts,
            "amount": amounts,
            "currency": currencies,
            "narration": narrations,
        }
    )
    log.info("DataFrame built: %d rows × %d columns", *df.shape)
    return df


def save_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Write *df* to an Excel file at *output_path*.

    Creates parent directories if they do not exist.

    Args:
        df: DataFrame to persist.
        output_path: Destination ``.xlsx`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")
    log.info("Saved → %s  (%d KB)", output_path, output_path.stat().st_size // 1024)


def generate_scenario_suite(base_output_dir: Path, rows: int) -> None:
    """Generate multiple datasets covering common happy/edge/stress paths."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        (
            "sample_transactions_happy.xlsx",
            dict(
                malformed_ts_rate=0.00,
                null_amount_rate=0.00,
                null_currency_rate=0.00,
                null_narration_rate=0.00,
                duplicate_txn_rate=0.00,
                invalid_currency_rate=0.00,
            ),
        ),
        (
            "sample_transactions_balanced.xlsx",
            dict(
                malformed_ts_rate=0.10,
                null_amount_rate=0.03,
                null_currency_rate=0.02,
                null_narration_rate=0.02,
                duplicate_txn_rate=0.01,
                invalid_currency_rate=0.01,
            ),
        ),
        (
            "sample_transactions_noisy.xlsx",
            dict(
                malformed_ts_rate=0.25,
                null_amount_rate=0.10,
                null_currency_rate=0.08,
                null_narration_rate=0.08,
                duplicate_txn_rate=0.05,
                invalid_currency_rate=0.07,
            ),
        ),
        (
            "sample_transactions_idempotency.xlsx",
            dict(
                malformed_ts_rate=0.05,
                null_amount_rate=0.02,
                null_currency_rate=0.01,
                null_narration_rate=0.01,
                duplicate_txn_rate=0.20,
                invalid_currency_rate=0.00,
            ),
        ),
    ]

    for filename, kwargs in scenarios:
        df = build_dataframe(rows, **kwargs)
        save_excel(df, base_output_dir / filename)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with ``rows`` and ``output``.
    """
    default_output = (
        Path(__file__).parent / "data" / "sample_transactions.xlsx"
    )
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data.")
    parser.add_argument(
        "--rows",
        type=int,
        default=5_000,
        help="Number of rows to generate (default: 5000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output .xlsx path (default: {default_output}).",
    )
    parser.add_argument(
        "--scenario",
        choices=["single", "suite"],
        default="single",
        help="Generate one file or a full scenario suite (default: single).",
    )
    parser.add_argument(
        "--null-amount-rate",
        type=float,
        default=0.03,
        help="Null rate for amount column in single mode (default: 0.03).",
    )
    parser.add_argument(
        "--null-currency-rate",
        type=float,
        default=0.02,
        help="Null rate for currency column in single mode (default: 0.02).",
    )
    parser.add_argument(
        "--null-narration-rate",
        type=float,
        default=0.02,
        help="Null rate for narration column in single mode (default: 0.02).",
    )
    parser.add_argument(
        "--malformed-ts-rate",
        type=float,
        default=0.10,
        help="Malformed timestamp rate in single mode (default: 0.10).",
    )
    parser.add_argument(
        "--duplicate-txn-rate",
        type=float,
        default=0.00,
        help="Duplicate txn_id rate in single mode (default: 0.00).",
    )
    parser.add_argument(
        "--invalid-currency-rate",
        type=float,
        default=0.00,
        help="Invalid currency token rate in single mode (default: 0.00).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, generate data, save to Excel."""
    args = parse_args()

    if not (5_000 <= args.rows <= 10_000):
        log.warning(
            "--rows %d is outside the recommended 5000–10000 range.", args.rows
        )

    if args.scenario == "suite":
        output_dir = args.output if args.output.is_dir() else args.output.parent
        log.info("Generating scenario suite in: %s", output_dir)
        generate_scenario_suite(output_dir, args.rows)
    else:
        df = build_dataframe(
            args.rows,
            null_amount_rate=args.null_amount_rate,
            null_currency_rate=args.null_currency_rate,
            null_narration_rate=args.null_narration_rate,
            malformed_ts_rate=args.malformed_ts_rate,
            duplicate_txn_rate=args.duplicate_txn_rate,
            invalid_currency_rate=args.invalid_currency_rate,
        )
        save_excel(df, args.output)
    log.info("Done.")


if __name__ == "__main__":
    main()
