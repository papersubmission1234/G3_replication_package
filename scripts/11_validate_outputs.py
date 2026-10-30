#!/usr/bin/env python3
"""Check output coverage and numerical consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, load_context, read_seed_table, write_json


def _record(records, name: str, passed: bool, detail: str, severity: str = "error") -> None:
    records.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def _optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "confirmatory.yaml"))
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    cfg, paths, data, experiment, _, _ = load_context(args.config)
    core = paths["output_root"] / "core"
    returns = read_seed_table(core, "returns.csv")
    posterior = read_seed_table(core, "posterior.csv")
    scenario_hashes = read_seed_table(core, "scenario_hashes.csv")
    optimizer = read_seed_table(core, "optimizer_diagnostics.csv")
    records: list[dict[str, object]] = []

    calendar_cfg = dict(cfg.get("calendar", {}) or {})
    expected_outcomes = int(calendar_cfg.get("expected_outcomes", 0))
    if expected_outcomes <= 0:
        expected_outcomes = len(data.calendar)

    date_counts = returns.groupby(["master_seed", "strategy"])["outcome_date"].nunique()
    _record(
        records,
        f"{expected_outcomes} unique outcomes per core seed/strategy",
        bool((date_counts == expected_outcomes).all()),
        f"expected={expected_outcomes}, min={date_counts.min()}, max={date_counts.max()}",
    )
    signatures = (
        returns.sort_values("outcome_date")
        .groupby(["master_seed", "strategy"])["outcome_date"]
        .apply(lambda x: tuple(pd.to_datetime(x).dt.strftime("%Y-%m-%d")))
    )
    per_seed_unique = signatures.groupby(level=0).nunique()
    _record(
        records,
        "return dates align across compared strategies",
        bool((per_seed_unique == 1).all()),
        f"max date-signatures within seed={per_seed_unique.max()}",
    )

    finite = np.isfinite(
        returns[["net_simple_return", "net_excess_return", "wealth"]].to_numpy(dtype=float)
    ).all()
    _record(records, "finite realized returns and wealth", bool(finite), str(finite))

    duplicate_returns = returns.duplicated(
        ["master_seed", "strategy", "outcome_date"], keep=False
    ).any()
    _record(
        records,
        "one return row per seed/strategy/outcome",
        not bool(duplicate_returns),
        f"duplicate key present={bool(duplicate_returns)}",
    )

    sums = posterior.groupby(["master_seed", "formation_date", "score"])["weight"].sum()
    max_error = float((sums - 1.0).abs().max())
    _record(
        records,
        "posterior weights sum to one",
        max_error < 1e-10,
        f"max absolute error={max_error:.3e}",
    )

    duplicate_hashes = scenario_hashes.duplicated(
        ["master_seed", "formation_date", "arm"], keep=False
    ).any()
    _record(
        records,
        "one frozen scenario matrix per seed/date/arm in core",
        not bool(duplicate_hashes),
        f"duplicate key present={bool(duplicate_hashes)}",
    )

    failed = optimizer[
        pd.to_numeric(
            optimizer.get("solver_status", pd.Series(index=optimizer.index, dtype=float)),
            errors="coerce",
        ).fillna(0) != 0
    ]
    fallback_missing = False
    if not failed.empty and "fallback" in failed:
        fallback_missing = (~failed["fallback"].astype("boolean").fillna(False)).any()
    _record(
        records,
        "failed optimizations have explicit fallback records",
        failed.empty or not bool(fallback_missing),
        f"nonzero solver-status rows={len(failed)}",
    )

    tail_count = (
        (1.0 - float(experiment.optimizer.cvar_alpha)) * int(experiment.n_scenarios)
    )
    _record(
        records,
        "primary CVaR has at least 20 effective tail scenarios",
        tail_count >= 20.0,
        (
            f"alpha={experiment.optimizer.cvar_alpha}, "
            f"scenarios={experiment.n_scenarios}, effective_tail_count={tail_count:.3f}"
        ),
    )
    if "cvar_effective_tail_count" in optimizer:
        reported = pd.to_numeric(optimizer["cvar_effective_tail_count"], errors="coerce")
        max_tail_error = float((reported - tail_count).abs().max())
        _record(
            records,
            "reported optimizer tail count matches configured horizon",
            max_tail_error < 1e-10,
            f"max absolute error={max_tail_error:.3e}",
        )

    eligibility_cfg = dict(cfg.get("eligibility", {}) or {})
    mode = str(eligibility_cfg.get("delisting_mode", ""))
    event_path = paths.get("delisting_events_csv")
    primary_no_lookahead = mode == "current_only" or (
        mode == "point_in_time" and event_path is not None and event_path.exists()
    )
    _record(
        records,
        "primary eligibility has no future-activity delisting oracle",
        primary_no_lookahead,
        f"delisting_mode={mode}, point_in_time_event_file={event_path}",
    )

    source_meta = dict(data.source_metadata or {})
    missing_meta = dict(source_meta.get("missing_outcome_return_policy", {}) or {})
    unresolved = int(missing_meta.get("n_unresolved", 0))
    _record(
        records,
        "missing eligible outcome returns are resolved explicitly",
        unresolved == 0,
        (
            f"detected={missing_meta.get('n_missing_eligible_outcomes_detected', 0)}, "
            f"filled={missing_meta.get('n_filled', 0)}, unresolved={unresolved}"
        ),
    )

    state_audit = paths["output_root"] / "extensions" / "state" / "corrected_regime_audit.csv"
    if state_audit.exists():
        state = pd.read_csv(state_audit)
        same = (state["formation_regime"] == state["update_regime"]).all()
        exact_columns = {"vix", "hy_oas_bps", "source_date"}.issubset(state.columns)
        _record(
            records,
            "state feedback is credited to formation state",
            bool(same and exact_columns),
            f"same_state={same}, audit_columns={exact_columns}",
        )

    one_arm_root = paths["output_root"] / "extensions" / "arm_pools" / "hier_only"
    if one_arm_root.exists() and list(one_arm_root.glob("seed_*/returns.csv")):
        one = read_seed_table(one_arm_root, "returns.csv")
        panel = one[one["strategy"].isin([
            "standalone__M_HierMacro", "post_cost_utility__mix", "fixed_equal__mix"
        ])].pivot_table(
            index=["master_seed", "outcome_date"],
            columns="strategy",
            values="net_excess_return",
        ).dropna()
        error = float(
            panel.sub(panel["standalone__M_HierMacro"], axis=0).abs().to_numpy().max()
        )
        _record(
            records,
            "one-arm mixtures reproduce standalone path",
            error < 1e-10,
            f"max absolute return difference={error:.3e}",
        )

    riskmap = _optional_csv(
        paths["output_root"] / "decompositions" / "full_risk_map" / "summary"
        / "action_channel_decomposition_by_seed.csv"
    )
    if not riskmap.empty and "reconstruction_gap" in riskmap:
        gap = float(pd.to_numeric(riskmap["reconstruction_gap"], errors="coerce").abs().max())
        _record(
            records,
            "risk-map channel decomposition reconstructs total action effect",
            gap < 1e-10,
            f"max absolute reconstruction gap={gap:.3e}",
        )

    table_status = paths["output_root"] / "paper_tables" / "table_build_status.json"
    if table_status.exists():
        payload = json.loads(table_status.read_text(encoding="utf-8"))
        _record(
            records,
            "all required manuscript tables are available",
            bool(payload.get("paper_ready_tables", False)),
            f"required_missing={payload.get('required_missing', [])}",
            severity="warning",
        )

    output = paths["output_root"] / "output_validation"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "output_validation_checks.csv", index=False)
    write_json(output / "output_validation_checks.json", records)
    failed_checks = frame[(~frame["passed"].astype(bool)) & (frame["severity"] == "error")]
    print(frame.to_string(index=False))
    if not failed_checks.empty and not args.no_fail:
        raise SystemExit("Output validation failed; see output_validation_checks.csv")


if __name__ == "__main__":
    main()
