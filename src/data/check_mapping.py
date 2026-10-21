# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[1]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

from pathlib import Path
from functools import cmp_to_key
import ctypes

ROOT = Path("data/raw/Vident-real")


# ============================================================
# Windows Explorer-style sorting
# ============================================================

StrCmpLogicalW = ctypes.windll.shlwapi.StrCmpLogicalW
StrCmpLogicalW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
StrCmpLogicalW.restype = ctypes.c_int


def windows_compare(a, b):
    return StrCmpLogicalW(a.name, b.name)


def get_folders(split):
    folders = [
        p
        for p in (ROOT / split).iterdir()
        if p.is_dir()
    ]

    folders = sorted(
        folders,
        key=cmp_to_key(windows_compare)
    )

    return folders


# ============================================================
# Expected frame counts from professor annotation
# ============================================================

EXPECTED = {
    "train": [
        888, 174, 394, 868, 285, 343, 632, 486, 2475, 497,
        176, 151, 1068, 1589, 84, 778, 253, 1376, 430, 3762,
        1690, 386, 1171, 130, 315, 257, 180, 730, 1119, 2091,
        482, 116, 48, 700, 461, 890, 1458, 12, 325, 1053,
        61, 23, 2041, 676, 683, 362, 253, 1118, 1716, 1127,
        585, 144, 495, 1338, 201, 542, 811, 378, 1113, 561,
        763, 945, 1488, 263, 828
    ],

    "val": [
        158, 719, 167, 236, 1250,
        240, 331, 501, 632, 28
    ],

    "test": [
        548, 374, 228, 302, 342,
        624, 1131, 1225, 610, 427,
        797, 328, 824, 897, 313,
        184, 428, 345, 1535, 2083,
        888, 42, 382, 131, 1929
    ],
}


# ============================================================
# Check all splits
# ============================================================

all_ok = True

for split, expected_frames in EXPECTED.items():

    folders = get_folders(split)

    print()
    print("=" * 90)
    print(split.upper())
    print("=" * 90)

    print(
        f"Professor videos: {len(expected_frames)}"
    )

    print(
        f"Raw folders:      {len(folders)}"
    )

    print()

    if len(folders) != len(expected_frames):
        print("WARNING: number of folders does not match professor annotation!")
        all_ok = False

    matches = 0

    for i, (folder, expected) in enumerate(
        zip(folders, expected_frames),
        start=1
    ):

        input_dir = folder / "input"

        actual = len(
            list(input_dir.glob("*.jpg"))
        )

        ok = actual == expected

        if ok:
            matches += 1
        else:
            all_ok = False

        status = "OK" if ok else "MISMATCH"

        print(
            f"{i:2d} | "
            f"prof={expected:4d} | "
            f"raw={actual:4d} | "
            f"{status:8s} | "
            f"{folder.name}"
        )

    print()

    print(
        f"MATCHED: {matches}/{len(expected_frames)}"
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 90)
print("FINAL RESULT")
print("=" * 90)

if all_ok:
    print("ALL SPLITS MATCH.")
    print("The professor video numbering matches Windows Explorer natural folder order.")
else:
    print("SOME SPLITS DO NOT MATCH.")
    print("Do not create final metadata yet.")