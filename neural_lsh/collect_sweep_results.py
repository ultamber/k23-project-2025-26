#!/usr/bin/env python3
"""
collect_sweep_results.py
Scan a `sweep_experiments` folder, read `config.json` and the last lines
of `search_results.txt` for each experiment, and produce a board
in CSV, Markdown and JSON formats.

Outputs (by default placed into the base folder):
 - board.csv    : table with flattened config keys + metrics
 - board.md     : same table in Markdown
 - board.json   : list of entries (config + metrics)

Usage:
  python collect_sweep_results.py --base sweep_experiments

Optional flags: --out-csv, --out-md, --out-json, --preview N
"""

import json
import csv
import re
from pathlib import Path
from collections import deque
import argparse
from typing import Dict, Any


METRIC_KEYS = [
    "Average AF",
    "Recall@5",
    "QPS",
    "tApproximateAverage",
    "tTrueAverage",
]


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested config dicts into single-level dict with dot keys."""
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def tail_lines(path: Path, n: int = 50):
    """Return the last n lines of a text file as a list."""
    dq = deque(maxlen=n)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                dq.append(line.rstrip("\n"))
    except FileNotFoundError:
        return []
    return list(dq)


def parse_metrics(lines):
    """Extract the required five metrics from file lines (search_results.txt).

    Returns dict with keys from METRIC_KEYS (values as strings) or None if not found.
    """
    metrics = {k: None for k in METRIC_KEYS}

    # Search from the end for the metric lines (they're often near EOF)
    for line in reversed(lines):
        for key in METRIC_KEYS:
            if line.startswith(key + ":") or line.startswith(key + " "):
                # Split at first colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    metrics[key] = parts[1].strip()
                else:
                    # fallback: grab trailing token
                    metrics[key] = line[len(key):].strip()

    return metrics


def collect(base: Path):
    base = Path(base)
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Base folder does not exist: {base}")

    entries = []

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        config_path = child / "config.json"
        results_path = child / "search_results.txt"

        if not config_path.exists() or not results_path.exists():
            # skip folders without both files
            continue

        try:
            cfg = json.load(open(config_path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {config_path}: {e}")
            continue

        lines = tail_lines(results_path, n=200)
        metrics = parse_metrics(lines)

        flat_cfg = flatten(cfg)

        entry = {"folder": child.name}
        # add config entries (flattened)
        entry.update({k: v for k, v in flat_cfg.items()})
        # add metrics
        entry.update({k: metrics.get(k) for k in METRIC_KEYS})

        entries.append(entry)

    return entries


def write_csv(entries, out_csv: Path):
    if not entries:
        print("No entries to write to CSV")
        return

    # determine all columns
    columns = []
    # always put folder first
    columns.append("folder")
    # collect remaining keys from all entries
    keys = set()
    for e in entries:
        keys.update(e.keys())
    # remove folder from keys
    keys.discard("folder")
    # order: sort keys, but place metrics at end in desired order
    metric_keys = METRIC_KEYS
    other_keys = sorted(k for k in keys if k not in metric_keys)
    columns.extend(other_keys)
    columns.extend(metric_keys)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for e in entries:
            # ensure all columns present
            row = {c: e.get(c, "") for c in columns}
            writer.writerow(row)

    print(f"Wrote CSV: {out_csv}")


def write_markdown(entries, out_md: Path):
    if not entries:
        print("No entries to write to Markdown")
        return

    # reuse CSV columns logic
    columns = ["folder"]
    keys = set()
    for e in entries:
        keys.update(e.keys())
    keys.discard("folder")
    metric_keys = METRIC_KEYS
    other_keys = sorted(k for k in keys if k not in metric_keys)
    columns.extend(other_keys)
    columns.extend(metric_keys)

    lines = []
    # header
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(sep)

    for e in entries:
        row = [str(e.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(row) + " |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Markdown: {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Collect sweep_experiments configs and results into a board")
    parser.add_argument("--base", default="sweep_experiments", help="Base folder containing experiment subfolders")
    parser.add_argument("--out-csv", default=None, help="Output CSV path (default: <base>/board.csv)")
    parser.add_argument("--out-md", default=None, help="Output Markdown path (default: <base>/board.md)")
    parser.add_argument("--out-json", default=None, help="Output JSON path (default: <base>/board.json)")
    parser.add_argument("--preview", type=int, default=0, help="Show a short preview of results (N rows)")
    parser.add_argument("--sort-by", default=None, help="Sort table by metric or config key (e.g. 'Recall@5' or 'Average AF')")
    parser.add_argument("--desc", action="store_true", help="Sort descending (largest first)")
    parser.add_argument("--top", type=int, default=0, help="Keep only top K rows after sorting (0 = keep all)")
    args = parser.parse_args()

    base = Path(args.base)
    out_csv = Path(args.out_csv) if args.out_csv else base / "board.csv"
    out_md = Path(args.out_md) if args.out_md else base / "board.md"
    out_json = Path(args.out_json) if args.out_json else base / "board.json"

    entries = collect(base)

    # Helper: try to parse numeric value from string (handles percentages and plain numbers)
    def parse_number(s):
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        # strip percent sign and commas
        try:
            t = str(s).strip()
            # Replace comma thousands separators
            t = t.replace(",", "")
            # If percentage like '12.34%' handle
            if t.endswith("%"):
                return float(t[:-1])
            # Extract first numeric token using regex
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", t)
            if m:
                return float(m.group(0))
        except Exception:
            return None
        return None

    # Sorting / top-k
    if args.sort_by:
        # Normalize sort key: allow common shortcuts
        sort_key = args.sort_by
        # If user provided a metric alias, map
        aliases = {
            "recall": "Recall@5",
            "recall@5": "Recall@5",
            "af": "Average AF",
            "average af": "Average AF",
            "qps": "QPS",
            "tapprox": "tApproximateAverage",
            "tapproximateaverage": "tApproximateAverage",
            "ttrue": "tTrueAverage",
            "ttrueaverage": "tTrueAverage",
        }
        key_lower = sort_key.replace(" ", "").lower()
        if key_lower in aliases:
            sort_key = aliases[key_lower]

        # If sort_key not present in any entry, warn and skip sorting
        if not any(sort_key in e for e in entries):
            print(f"Warning: sort key '{sort_key}' not found in entries; skipping sort")
        else:
            def sort_tuple(e):
                v = e.get(sort_key)
                num = parse_number(v)
                # Place None values after numeric ones
                return (num is None, num if num is not None else 0.0)

            entries = sorted(entries, key=sort_tuple, reverse=args.desc)

            if args.top and args.top > 0:
                entries = entries[: args.top]

    # write outputs
    write_csv(entries, out_csv)
    write_markdown(entries, out_md)
    out_json.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {out_json}")

    if args.preview and entries:
        print("\nPreview:")
        for e in entries[: args.preview]:
            print(e)

    print(f"\nDone. Collected {len(entries)} experiment entries from {base}")


if __name__ == "__main__":
    main()
