"""Lance tous les tests ciphermark en mode 'python -m'."""

import sys
import traceback


MODULES = [
    "tests.ciphermark.test_crypto",
    "tests.ciphermark.test_witness",
    "tests.ciphermark.test_equation",
    "tests.ciphermark.test_stable_subspace",
    "tests.ciphermark.test_msg_processor",
    "tests.ciphermark.test_registry",
    "tests.ciphermark.test_parity",
    "tests.ciphermark.test_pipeline",
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
    print("\nTous les tests ciphermark OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
