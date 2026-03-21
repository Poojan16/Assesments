"""generate_data.py — synthetic transaction data generator."""

import argparse
import logging
import random
import uuid
from pathlib import Path

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


def _dirty_timestamp() -> str:
    """Return one timestamp string from the dirty pool.

    Roughly 60 % clean-with-tz, 30 % plain string, 10 % malformed.
    """
    roll = random.random()
    if roll < 0.60:
        return random.choice(_CLEAN_WITH_TZ)
    if roll < 0.90:
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


def build_dataframe(n_rows: int) -> pd.DataFrame:
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
    account_ids = [random.choice(ACCOUNTS) for _ in range(n_rows)]
    txn_ts = [_dirty_timestamp() for _ in range(n_rows)]
    amounts = [
        _nullable(round(random.uniform(-500.0, 10_000.0), 2), null_rate=0.03)
        for _ in range(n_rows)
    ]
    currencies = [
        _nullable(random.choice(CURRENCIES), null_rate=0.02)
        for _ in range(n_rows)
    ]
    narrations = [
        _nullable(
            fake.sentence(nb_words=random.randint(4, 10)).rstrip("."),
            null_rate=0.02,
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
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, generate data, save to Excel."""
    args = parse_args()

    if not (5_000 <= args.rows <= 10_000):
        log.warning(
            "--rows %d is outside the recommended 5000–10000 range.", args.rows
        )

    df = build_dataframe(args.rows)
    save_excel(df, args.output)
    log.info("Done.")


if __name__ == "__main__":
    main()
