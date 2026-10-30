#!/usr/bin/env python3
"""Export manuscript summaries from saved CSV results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]

def resolve(p: Path) -> Path:
    p=p.expanduser()
    return p.resolve() if p.is_absolute() or p.exists() else (ROOT/p).resolve()

def summarize(values: pd.Series) -> pd.Series:
    a=values.to_numpy(float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Incomplete/nonfinite seed metric')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return pd.Series({'n_seeds':len(a),'mean':mean,'ci_low':mean-half,'ci_high':mean+half,'seed_sd':sd})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'work/config/generated_study_config.yaml')
    parser.add_argument('--output-root',type=Path,help='Root containing inference/; overrides config')
    parser.add_argument('--tables-dir',type=Path)
    parser.add_argument('--expected-seeds',type=int,default=30)
    args=parser.parse_args()
    if args.output_root:root=resolve(args.output_root)
    else:
        import yaml
        cfg=yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
        root=Path(cfg['paths']['output_root']).expanduser()
        if not root.is_absolute():root=ROOT/root
    out=resolve(args.tables_dir) if args.tables_dir else root/'paper_tables/manuscript_support'
    out.mkdir(parents=True,exist_ok=True)
    sources={
        'metrics':root/'inference/all_trial_metrics_by_seed.csv',
        'dsr':root/'inference/all_trials_deflated_sharpe_approx.csv',
        'rw':root/'inference/all_trials_romano_wolf_by_seed.csv',
    }
    for path in sources.values():
        if not path.is_file():raise FileNotFoundError(path)
    required=['master_seed','strategy','period','n_months','excess_sharpe']
    data=pd.read_csv(sources['metrics'],usecols=required)
    data=data.loc[data['period'].eq('full')].copy()
    if not data['n_months'].eq(83).all():raise ValueError('Non-primary outcome count in full-period metrics')
    if data.duplicated(['master_seed','strategy']).any():raise ValueError('Duplicate seed/strategy key')
    def aggregate(sub):
        rows=[]
        for strategy,g in sub.groupby('strategy',sort=True):
            if g.master_seed.nunique()!=args.expected_seeds:raise ValueError(f'Incomplete seeds: {strategy}')
            rows.append({'strategy':strategy,**summarize(g.excess_sharpe).to_dict()})
        return pd.DataFrame(rows)
    mean=aggregate(data.loc[data.strategy.str.startswith('mean_ablation/')])
    mean[['experiment','portfolio_strategy']]=mean.strategy.str.split('::',n=1,expand=True)
    mean.to_csv(out/'table_09_mean_ablation_all_strategies.csv',index=False)
    filt=data.strategy.str.startswith('core::') | data.strategy.str.match(r'^robustness/(?:cost_|position_cap_)')
    aggregate(data.loc[filt]).to_csv(out/'table_EC16_cost_cap_sharpe_summary.csv',index=False)
    dsr=pd.read_csv(sources['dsr']);rw=pd.read_csv(sources['rw'])
    rows=[]
    for strategy in ['standalone__M_HierMacro','fixed_equal__mix','post_cost_utility__mix','post_cost_utility__hard','standalone__M_FF5MOM']:
        name='core::'+strategy
        d=dsr.loc[dsr.strategy.eq(name)];r=rw.loc[rw.strategy.eq(name)]
        if d.master_seed.nunique()!=args.expected_seeds or r.master_seed.nunique()!=args.expected_seeds:
            raise ValueError(f'Incomplete multiplicity coverage: {strategy}')
        rows.append({'strategy':strategy,'n_seeds':args.expected_seeds,
                     'approx_dsr_mean':d.dsr_probability.mean(),
                     'stepdown_p_mean':r.romano_wolf_p_value_one_sided.mean(),
                     'stepdown_p_median':r.romano_wolf_p_value_one_sided.median()})
    pd.DataFrame(rows).to_csv(out/'table_EC20_multiplicity_diagnostics.csv',index=False)
    eff=dsr.groupby('master_seed')[['effective_trials','expected_max_monthly_sharpe']].first()
    eff.to_csv(out/'effective_trial_approximation_by_seed.csv')
    report={'sources':{k:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in sources.items()},
            'effective_trials_mean':float(eff.effective_trials.mean()),
            'expected_max_monthly_sharpe_mean':float(eff.expected_max_monthly_sharpe.mean()),
            'tables_dir':str(out.resolve()),
            'note':'DSR approximation is reported as a diagnostic, not reliable selection-adjusted evidence.'}
    (out/'support_tables_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
