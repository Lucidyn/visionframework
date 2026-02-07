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
    "test.core.test_processors",
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

    # Collect skip exception types
    _skip_types = []
    try:
        from _pytest.outcomes import Skipped
        _skip_types.append(Skipped)
    except ImportError:
        pass
    # Also detect test-level _Skip markers
    if hasattr(mod, "_Skip"):
        _skip_types.append(mod._Skip)
    _skip_types = tuple(_skip_types) if _skip_types else None

    passed = failed = skipped = 0
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
        except BaseException as e:
            if _skip_types and isinstance(e, _skip_types):
                print(f"  {name} ... [SKIPPED]")
                skipped += 1
            else:
                print(f"  {name} ... [FAILED] {e}")
                failed += 1
    if skipped:
        print(f"\n  ({skipped} tests skipped due to unavailable dependencies)")
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
