"""Run the complete offline correctness suite, including tiny RoBERTa integration."""
import unittest
import contextlib
import io


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(logs.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
