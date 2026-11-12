"""Build the Task 4 data: UCI Spambase, deduplicated and shuffled.

Source: UCI Machine Learning Repository, "Spambase" (dataset 94), files ``spambase.data`` and
``spambase.names`` inside
    https://archive.ics.uci.edu/static/public/94/spambase.zip
Licence: CC BY 4.0 (stated on the dataset page). Citation: Hopkins, M., Reeber, E., Forman, G.,
& Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X

Transformations, in order:
    1. Parse ``spambase.data`` (no header, comma separated): 4,601 rows, 57 numeric features
       and a last 0/1 column. The 57 feature names are read from ``spambase.names``
       (the lines of the form ``name:  continuous.``); the last column becomes ``spam``
       (1 = spam, 0 = not spam).
    2. Rename the six ``char_freq_`` columns whose names are symbols:
           char_freq_;  -> char_freq_semicolon      char_freq_(  -> char_freq_paren
           char_freq_[  -> char_freq_bracket        char_freq_!  -> char_freq_exclam
           char_freq_$  -> char_freq_dollar         char_freq_#  -> char_freq_hash
       The other 51 names are kept as in the source.
    3. Drop exact duplicate rows (all 58 values equal), keeping the first copy. A copy of a row
       on both sides of a split would give a free hit to every memorising model.
    4. Drop every row whose 57 feature values occur with BOTH labels (the same e-mail
       described once as spam and once as not spam). No model can be right on both copies.
    5. Shuffle the rows once with the fixed seed ``SHUFFLE_SEED`` (the raw file is sorted:
       all spam rows come first).
    6. Nothing else is changed: values are not scaled, clipped or rounded; ``capital_run_length_*``
       keep their meaning (the values are the raw UCI values; any transform is applied later,
       in the notebook).

The counts of every step are printed; none of them is assumed.

Output (next to this script):
    data/spambase.csv     57 features + spam (header, no index)   -> students

The file is one labelled table, not a train/test pair: students make their own split.

Run:
    python make_dataset.py                      # downloads the UCI archive
    python make_dataset.py --zip spambase.zip   # use a local copy of the archive
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
DATA_MEMBER = "spambase.data"
NAMES_MEMBER = "spambase.names"

TARGET = "spam"
SHUFFLE_SEED = 3231  # course code MOP 3231
N_FEATURES = 57

RENAMES = {
    "char_freq_;": "char_freq_semicolon",
    "char_freq_(": "char_freq_paren",
    "char_freq_[": "char_freq_bracket",
    "char_freq_!": "char_freq_exclam",
    "char_freq_$": "char_freq_dollar",
    "char_freq_#": "char_freq_hash",
}

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# A feature line of spambase.names: "word_freq_make:         continuous."
_NAME_LINE = re.compile(r"^(?P<name>\S+):\s+continuous\.\s*$")


def read_archive(url: str = UCI_URL, local_zip: Path | None = None) -> tuple[bytes, bytes]:
    """Get ``spambase.data`` and ``spambase.names`` from the UCI zip archive.

    Parameters
    ----------
    url : str
        Location of the UCI Spambase zip archive. Ignored when ``local_zip`` is given.
    local_zip : pathlib.Path or None
        A previously downloaded copy of the archive.

    Returns
    -------
    tuple of (bytes, bytes)
        The content of ``spambase.data`` and of ``spambase.names``.
    """
    if local_zip is not None:
        payload = local_zip.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(DATA_MEMBER), archive.read(NAMES_MEMBER)


def parse_names(names_bytes: bytes) -> list[str]:
    """Read the 57 feature names, in order, from the text of ``spambase.names``.

    Parameters
    ----------
    names_bytes : bytes
        Content of ``spambase.names``.

    Returns
    -------
    list of str
        The feature names as written in the source (``char_freq_;`` and friends included).

    Raises
    ------
    ValueError
        If the file does not list exactly ``N_FEATURES`` names.
    """
    names = []
    for line in names_bytes.decode("utf-8", errors="strict").splitlines():
        match = _NAME_LINE.match(line)
        if match:
            names.append(match.group("name"))
    if len(names) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} feature names in {NAMES_MEMBER}, found {len(names)}")
    return names


def build_frame(data_bytes: bytes, source_names: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse, rename, deduplicate, drop label conflicts and shuffle.

    Parameters
    ----------
    data_bytes : bytes
        Content of ``spambase.data``.
    source_names : list of str
        Output of :func:`parse_names`.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The final table (features in source order, then ``spam``) and the counts of every
        step: ``n_raw``, ``n_duplicates_removed``, ``n_conflicting_removed``, ``n_rows``.
    """
    raw = pd.read_csv(io.BytesIO(data_bytes), header=None)
    if raw.shape[1] != N_FEATURES + 1:
        raise ValueError(f"expected {N_FEATURES + 1} columns, found {raw.shape[1]}")
    names = [RENAMES.get(name, name) for name in source_names]
    raw.columns = names + [TARGET]
    unrenamed = [name for name in RENAMES if name not in source_names]
    if unrenamed:
        raise ValueError(f"names to rename not found in {NAMES_MEMBER}: {unrenamed}")

    deduplicated = raw.drop_duplicates().reset_index(drop=True)
    labels_per_row = deduplicated.groupby(names)[TARGET].transform("nunique")
    clean = deduplicated.loc[labels_per_row == 1]
    shuffled = clean.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)
    counts = {
        "n_raw": len(raw),
        "n_duplicates_removed": len(raw) - len(deduplicated),
        "n_conflicting_removed": len(deduplicated) - len(clean),
        "n_rows": len(shuffled),
    }
    return shuffled, counts


def check_frame(frame: pd.DataFrame) -> None:
    """Assert the invariants the handout relies on.

    Raises
    ------
    AssertionError
        If values are missing, the target is not 0/1, the column order is wrong, a row is
        duplicated, or one feature vector carries both labels.
    """
    features = [c for c in frame.columns if c != TARGET]
    assert len(features) == N_FEATURES and list(frame.columns) == features + [TARGET]
    assert not frame.isna().any().any()
    assert set(frame[TARGET].unique()) == {0, 1}
    assert not any(any(ch in name for ch in ";([!$# ") for name in features)
    assert int(frame.duplicated().sum()) == 0
    assert int(frame.duplicated(subset=features).sum()) == 0  # no feature vector twice, so none with both labels


def main() -> None:
    """Fetch the UCI source, build and verify the table, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=None, help="local copy of spambase.zip (skips the download)")
    args = parser.parse_args()

    data_bytes, names_bytes = read_archive(local_zip=args.zip)
    frame, counts = build_frame(data_bytes, parse_names(names_bytes))
    check_frame(frame)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / "spambase.csv", index=False)

    print(f"raw rows: {counts['n_raw']}")
    print(f"exact duplicate rows removed: {counts['n_duplicates_removed']}")
    print(f"rows with a feature vector seen under both labels removed: {counts['n_conflicting_removed']}")
    print(f"final rows: {counts['n_rows']}  features: {frame.shape[1] - 1}")
    print(f"spam share: {frame[TARGET].mean():.4f} ({int(frame[TARGET].sum())} spam, "
          f"{int((frame[TARGET] == 0).sum())} not spam)")
    print(f"shuffle seed: {SHUFFLE_SEED}")
    print(f"wrote {DATA_DIR / 'spambase.csv'}")


if __name__ == "__main__":
    main()
