"""Run every test, then prove that no research file was created, changed or deleted by the dashboard code.

    python tests/run_all.py        (from the ui/ folder)

The fingerprint covers every file in the project folder outside ui/: SHA-256 for files up to 8 MB,
size and modification time for the larger ones.
"""
import hashlib
import sys
import unittest
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
ROOT = UI.parent
SKIP_DIRS = {"ui", ".claude", "backups", "__pycache__", ".git"}
SKIP_FILES = {"run_dashboard.py", "run_dashboard.bat", "PROGRESS.md", "desktop.ini"}


def fingerprint():
    out = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or rel.name in SKIP_FILES:
            continue
        st = p.stat()
        out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest() if st.st_size <= 8_000_000 else (st.st_size, st.st_mtime_ns)
    return out


def main():
    before = fingerprint()
    suite = unittest.defaultTestLoader.discover(str(UI / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = fingerprint()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    print(f"\nResearch files fingerprinted: {len(before)}. Created, changed or deleted while the tests ran: {len(changed)}")
    for k in changed:
        print("  CHANGED:", k)
    sys.exit(0 if result.wasSuccessful() and not changed else 1)


if __name__ == "__main__":
    main()
