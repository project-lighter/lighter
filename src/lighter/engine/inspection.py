"""Read-only operations shared by human and agent command-line users."""

import argparse
import json
from typing import Any

import yaml

from lighter.engine.records import describe, diff_records, list_records, read_record


def inspection_cli(argv: list[str]) -> bool:
    """Handle inspect/runs commands; return false for native execution stages."""
    if not argv or argv[0] not in {"inspect", "runs"}:
        return False
    if argv[0] == "inspect":
        from lighter.engine.runner import ConfigLoader

        parser = argparse.ArgumentParser(prog="lighter inspect", description="Compose source without imports or construction")
        parser.add_argument("inputs", nargs="+", help="Configuration files and overrides")
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
        args = parser.parse_intermixed_args(argv[1:])
        source = describe(ConfigLoader.load(args.inputs).get())
        print(
            json.dumps(source, indent=2, sort_keys=True, allow_nan=False)
            if args.json
            else yaml.safe_dump(source, sort_keys=False, allow_unicode=True),
            end="\n" if args.json else "",
        )
        return True

    parser = argparse.ArgumentParser(
        prog="lighter runs", description="Read local experiment records without executing recipes"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List published attempts; status does not establish process liveness")
    listing.add_argument("root", help="Record root, normally DEFAULT_ROOT_DIR/lighter_runs")
    listing.add_argument("--json", action="store_true", help="Emit complete records as JSON")
    show = commands.add_parser("show", help="Read one record as JSON")
    show.add_argument("path", help="Attempt directory or record.json")
    compare = commands.add_parser("diff", help="Compare requested and observed settings as JSON")
    compare.add_argument("left")
    compare.add_argument("right")
    args = parser.parse_args(argv[1:])
    result: Any
    if args.command == "list":
        result = list_records(args.root)
        if not args.json:
            print("attempt_id\tstage\tstatus\tname")
            for record in result:
                print("\t".join(str(record.get(key) or "") for key in ("attempt_id", "stage", "status", "name")))
            return True
    elif args.command == "show":
        result = read_record(args.path)
    else:
        result = diff_records(args.left, args.right)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return True
