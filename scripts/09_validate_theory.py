#!/usr/bin/env python3
"""Check the mechanisms and counterexamples used in the paper."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, load_yaml, study_paths, write_json
from g3_confirmatory.theory_checks import run_all_theory_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    paths = study_paths(cfg)
    output = paths["output_root"] / "theory_checks"
    output.mkdir(parents=True, exist_ok=True)
    checks = run_all_theory_checks()
    checks.to_csv(output / "theory_checks.csv", index=False)
    records = checks.to_dict(orient="records")
    write_json(output / "theory_checks.json", records)
    if "passed" in checks and not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"].astype(bool)].to_dict(orient="records")
        raise SystemExit(f"Theory check failure: {failed}")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
