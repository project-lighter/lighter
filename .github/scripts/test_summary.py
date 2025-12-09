#!/usr/bin/env python3
"""Generate GitHub Actions test summary from JUnit XML."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    """Parse JUnit XML and write summary to GitHub Actions step summary."""
    xml_path = Path("test-results.xml")

    if not xml_path.exists():
        print("⚠️  No test results found", file=sys.stderr)
        return 1

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        tests = int(root.get("tests", 0))
        failures = int(root.get("failures", 0))
        errors = int(root.get("errors", 0))
        skipped = int(root.get("skipped", 0))
        passed = tests - failures - errors - skipped

        # Determine status emoji
        if failures + errors > 0:
            status = "❌"
        elif skipped == tests:
            status = "⏭️"
        else:
            status = "✅"

        # Print summary lines
        print(f"{status} **Test Results Summary**")
        print(f"- ✅ Passed: {passed}")
        print(f"- ❌ Failed: {failures}")
        print(f"- ⚠️  Errors: {errors}")
        print(f"- ⏭️  Skipped: {skipped}")
        print(f"- **Total: {tests}**")

        # Exit with error if tests failed
        return 1 if (failures + errors > 0) else 0

    except ET.ParseError as e:
        print(f"❌ Failed to parse XML: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
