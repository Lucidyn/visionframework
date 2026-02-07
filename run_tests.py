"""Test runner — run core tests without pytest."""

import sys
import io
import traceback
import importlib
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

TEST_MODULES = [
    "test.core.test_tracker_utils",
    "test.core.test_input_validation",
    "test.core.test_basic_api",
]


def run_module(module_name: str):
    print(f"\n{'=' * 60}")
    print(f"  {module_name}")
    print("=" * 60)
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"  [IMPORT ERROR] {e}")
        traceback.print_exc()
        return 0, 1

    passed = failed = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        try:
            fn()
            print(f"  {name} ... [PASSED]")
            passed += 1
        except Exception as e:
            print(f"  {name} ... [FAILED] {e}")
            failed += 1
    return passed, failed


def main():
    total_p = total_f = 0
    for m in TEST_MODULES:
        p, f = run_module(m)
        total_p += p
        total_f += f

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {total_p} passed, {total_f} failed")
    print("=" * 60)
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
