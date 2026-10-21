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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
    / "segment_quality_summary.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_quality"
    / "manual_segment_audit.csv"
)


def main():

    summary = pd.read_csv(SUMMARY_PATH)

    audit = summary[
        [
            "segment_id",
            "label",
            "num_frames",
        ]
    ].copy()

    # Manual audit fields
    audit["identifiability"] = ""
    audit["temporal_consistency"] = ""
    audit["visibility_issue"] = ""
    audit["anatomical_context"] = ""
    audit["notes"] = ""

    audit = audit.sort_values(
        "segment_id"
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Rows created: {len(audit)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()