# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "quadrant_metadata.csv"
)

MANUAL_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
    / "manual_segment_audit_colored.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "binary_clean_metadata.csv"
)


def main():

    metadata = pd.read_csv(METADATA_PATH)
    audit = pd.read_excel(MANUAL_AUDIT_PATH)

    # We cleaned only the training set manually.
    train = metadata[
        metadata["split"] == "train"
    ].copy()

    # Keep only the manual information we need.
    audit = audit[
        ["segment_id", "identifiability"]
    ].copy()

    # Attach manual audit to original metadata.
    train = train.merge(
        audit,
        on="segment_id",
        how="left"
    )

    print("Original train segments:", len(train))

    print("\nManual audit:")
    print(
        train["identifiability"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------
    # Remove manually identified ambiguous segments
    # --------------------------------------------------

    clean_train = train[
        train["identifiability"] != "Ambiguous"
    ].copy()

    dropped = train[
        train["identifiability"] == "Ambiguous"
    ].copy()

    print("\nDropped ambiguous segments:", len(dropped))

    print(
        dropped[
            ["segment_id", "label"]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # Convert quadrant labels to binary labels
    #
    # Q1, Q2 -> Upper
    # Q3, Q4 -> Lower
    # --------------------------------------------------

    label_mapping = {
        "Q1": "Upper",
        "Q2": "Upper",
        "Q3": "Lower",
        "Q4": "Lower",
    }

    clean_train["binary_label"] = (
        clean_train["label"]
        .map(label_mapping)
    )

    print("\nBefore cleaning — binary distribution:")

    before_binary = (
        train["label"]
        .map(label_mapping)
        .value_counts()
    )

    print(before_binary)

    print("\nAfter cleaning — binary distribution:")

    after_binary = (
        clean_train["binary_label"]
        .value_counts()
    )

    print(after_binary)

    print(
        "\nRemaining train segments:",
        len(clean_train)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    clean_train.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()