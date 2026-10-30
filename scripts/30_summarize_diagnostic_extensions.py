#!/usr/bin/env python3
"""Summarize diagnostic extensions and collect their result archive."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import t

from _common import ROOT, load_context
from g3_confirmatory.metrics import compute_performance_metrics
from g3_confirmatory.serialization import write_json


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = float(x.mean()) if len(x) else np.nan
    if len(x) < 2:
        return m, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    q = float(t.ppf(.975, len(x)-1))
    return m, m-q*se, m+q*se


ALLOWED_FAMILIES = {"small_pool", "frozen_shrinkage", "covariance_robustness"}


def collect(root: Path):
    rows = []
    for f in sorted(root.glob("*/*/seed_*/returns.csv")):
        family = f.parents[2].name
        if family not in ALLOWED_FAMILIES:
            continue
        experiment = f.parents[1].name
        d = pd.read_csv(f)
        seed = int(f.parent.name.split("seed_")[-1])
        manifest = json.loads((f.parent / "run_manifest.json").read_text(encoding="utf-8"))
        gamma = float(manifest.get("experiment_config", {}).get("utility_gamma", 1.0))
        for strategy, g in d.groupby("strategy"):
            rows.append({
                "family": family,
                "experiment": experiment,
                "master_seed": seed,
                "strategy": strategy,
                **compute_performance_metrics(g, utility_gamma=gamma),
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, group_cols, metrics):
    out=[]
    for key,g in frame.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base=dict(zip(group_cols,key))
        for metric in metrics:
            m,lo,hi=ci(g[metric])
            out.append({**base,"metric":base.get("metric", metric),"n_seeds":g[metric].notna().sum(),"mean":m,"ci_low":lo,"ci_high":hi})
    return pd.DataFrame(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="work/config/generated_study_config.yaml")
    args=parser.parse_args()
    os.chdir(ROOT)
    cfg, paths, data, base, _, seeds = load_context(args.config)
    ext_root = paths["output_root"] / "diagnostics/extensions"
    out = paths["output_root"] / "diagnostics/summary/extensions"
    out.mkdir(parents=True, exist_ok=True)
    frame=collect(ext_root)
    if frame.empty:
        raise FileNotFoundError(f"No extension returns under {ext_root}")
    frame.to_csv(out/"extension_metrics_by_seed.csv",index=False)
    summary=summarize(frame,["family","experiment","strategy"],["excess_sharpe","annualized_excess_mean","certainty_equivalent_annualized","maximum_drawdown","mean_one_way_turnover_ex_initial"])
    summary.to_csv(out/"extension_metric_summary.csv",index=False)

    effects=[]
    for (family,experiment,seed),g in frame.groupby(["family","experiment","master_seed"]):
        w=g.set_index("strategy")
        if {"post_cost_utility__mix","post_cost_utility__hard"}.issubset(w.index):
            for metric in ("excess_sharpe","annualized_excess_mean","certainty_equivalent_annualized"):
                effects.append({"family":family,"experiment":experiment,"master_seed":seed,"metric":metric,
                    "mix_minus_hard":float(w.loc["post_cost_utility__mix",metric]-w.loc["post_cost_utility__hard",metric])})
    effects=pd.DataFrame(effects)
    if not effects.empty:
        effects.to_csv(out/"mix_minus_hard_by_seed.csv",index=False)
        summarize(effects,["family","experiment","metric"],["mix_minus_hard"]).to_csv(out/"mix_minus_hard_summary.csv",index=False)

    fs=frame[frame.family=="frozen_shrinkage"].copy()
    if not fs.empty:
        parsed=fs.experiment.str.extract(r"shrink_(?P<fraction>[0-9p]+)_(?P<risk_map>endogenous|common_full)")
        fs["fraction"]=parsed.fraction.str.replace("p",".",regex=False).astype(float)
        fs["risk_map"]=parsed.risk_map
        shr=[]
        metrics=("excess_sharpe","annualized_excess_mean","certainty_equivalent_annualized","maximum_drawdown")
        for (seed,risk_map,strategy),g in fs.groupby(["master_seed","risk_map","strategy"]):
            base0=g[g.fraction==0]
            if len(base0)!=1: continue
            b=base0.iloc[0]
            for _,r in g.iterrows():
                for metric in metrics:
                    shr.append({"master_seed":seed,"risk_map":risk_map,"strategy":strategy,"fraction":r.fraction,
                        "metric":metric,"difference_from_fraction_zero":float(r[metric]-b[metric])})
        shr=pd.DataFrame(shr)
        shr.to_csv(out/"frozen_shrinkage_effects_by_seed.csv",index=False)
        summarize(shr,["risk_map","strategy","fraction","metric"],["difference_from_fraction_zero"]).to_csv(out/"frozen_shrinkage_effects_summary.csv",index=False)

    cv=effects[effects.family=="covariance_robustness"].copy() if not effects.empty else pd.DataFrame()
    if not cv.empty:
        parts=cv.experiment.str.extract(r"(?P<estimator>sample|mp|ledoit_wolf|ewma_ledoit_wolf)__(?P<risk_map>endogenous|common_budget|common_full)")
        cv=pd.concat([cv.reset_index(drop=True),parts.reset_index(drop=True)],axis=1)
        cv.to_csv(out/"covariance_action_effects_by_seed.csv",index=False)
        summarize(cv,["estimator","risk_map","metric"],["mix_minus_hard"]).to_csv(out/"covariance_action_effects_summary.csv",index=False)
        piv=cv.pivot_table(index=["master_seed","estimator","metric"],columns="risk_map",values="mix_minus_hard").reset_index()
        if {"endogenous","common_full"}.issubset(piv.columns):
            piv["risk_map_attenuation"]=piv["endogenous"]-piv["common_full"]
            piv.to_csv(out/"covariance_riskmap_attenuation_by_seed.csv",index=False)
            summarize(piv,["estimator","metric"],["risk_map_attenuation"]).to_csv(out/"covariance_riskmap_attenuation_summary.csv",index=False)

    status={
        "configured_seeds":list(map(int,seeds)),
        "families":sorted(frame.family.unique().tolist()),
        "seed_counts":frame.groupby(["family","experiment"])["master_seed"].nunique().reset_index().to_dict(orient="records"),
    }
    write_json(out/"summary_status.json",status)
    bundle=paths["output_root"] / "diagnostics/extension_bundle.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for family in sorted(ALLOWED_FAMILIES):
            base_dir = ext_root / family
            if base_dir.exists():
                for p in sorted(base_dir.rglob("*")):
                    if p.is_file() and p.suffix in {".csv",".json",".png",".md"}:
                        z.write(p,p.relative_to(paths["output_root"] / "diagnostics").as_posix())
        if out.exists():
            for p in sorted(out.rglob("*")):
                if p.is_file() and p.suffix in {".csv",".json",".png",".md"}:
                    z.write(p,p.relative_to(paths["output_root"] / "diagnostics").as_posix())
        for p in [
            ROOT/"scripts/24_run_diagnostic_extensions.py",
            ROOT/"scripts/28_run_frozen_posterior_shrinkage.py",
            ROOT/"scripts/29_run_covariance_regularization.py",
            ROOT/"scripts/30_summarize_diagnostic_extensions.py",
            ROOT/"src/g3_confirmatory/covariance_estimators.py",
        ]:
            if p.exists():
                z.write(p,"reproduction/"+p.relative_to(ROOT).as_posix())
    print(f"Summary: {out}\nReturn this file: {bundle}")


if __name__=="__main__":
    main()
