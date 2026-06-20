"""
Pytest configuration and custom hooks for the test suite.
"""

from collections import Counter


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a per-file test summary at the end of the test run."""
    reports = terminalreporter.stats
    passed = reports.get("passed", [])
    failed = reports.get("failed", [])

    def get_file(report):
        return report.nodeid.split("::")[0]

    passed_files = Counter(get_file(rep) for rep in passed)
    failed_files = Counter(get_file(rep) for rep in failed)

    all_files = set(passed_files) | set(failed_files)
    print("\nTest Results by File:")
    for file in sorted(all_files):
        print(
            f"  {file}: {passed_files.get(file, 0)} passed, {failed_files.get(file, 0)} failed"
        )

    print(f"\nFinal Summary: {len(passed)} passed, {len(failed)} failed\n")
