"""Lance tous les tests crystal en mode 'python -m'."""

import sys
import traceback


MODULES = [
    "tests.crystal.test_crypto",
    "tests.crystal.test_witness",
    "tests.crystal.test_equation",
    "tests.crystal.test_msg_processor",
]


def main() -> int:
    failed = 0
    for mod_name in MODULES:
        print(f"\n=== {mod_name} ===")
        try:
            mod = __import__(mod_name, fromlist=["*"])
            for name in dir(mod):
                if name.startswith("test_"):
                    fn = getattr(mod, name)
                    print(f"  - {name}", end="... ")
                    fn()
                    print("ok")
        except Exception:  # pragma: no cover
            traceback.print_exc()
            failed += 1

    if failed:
        print(f"\n{failed} module(s) en echec")
        return 1
    print("\nTous les tests crystal OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
