#!/usr/bin/env python3
"""Check experiment labels and saved configurations without loading raw inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT
from g3_confirmatory.configuration import load_yaml, study_paths, experiment_config_from_mapping
from g3_confirmatory.run_checks import scan_run_settings
from g3_confirmatory.serialization import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "confirmatory.yaml")
    parser.add_argument("--output-root", type=Path, help="Override only the folder to inspect.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output = args.output_root or study_paths(cfg)["output_root"]
    config = experiment_config_from_mapping(cfg.get("experiment"))
    records = scan_run_settings(output, expected_core=config)
    if not records:
        raise SystemExit(f"No saved seed manifests found under {output}")
    bad = [r for r in records if not r["passed"]]
    summary = {"n_manifests": len(records),
               "n_experiments": len({r["experiment_id"] for r in records}),
               "n_failed": len(bad), "passed": not bad, "records": records}
    if args.report:
        write_json(args.report, summary)
    print(f"Checked {summary['n_manifests']} manifests in {summary['n_experiments']} experiments; {len(bad)} failed.")
    for record in bad:
        print(record["manifest"])
        for issue in record["issues"]:
            print("  " + issue)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
