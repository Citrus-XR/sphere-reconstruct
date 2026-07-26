"""管理 CLI."""

from __future__ import annotations

import argparse
import json

from .diagnostics import diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description="sphere-reconstruct 環境診断")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = diagnose()
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"環境診断: {'準備完了' if report['ready'] else '不足あり'}")
        for name, check in report["checks"].items():
            marker = "✓" if check["ok"] else "○" if check.get("optional") else "✗"
            print(f"  {marker} {name}: {check['message']}")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
