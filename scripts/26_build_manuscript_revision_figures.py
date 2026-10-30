#!/usr/bin/env python3
"""Draw three manuscript figures from saved diagnostic tables.

Run from the repository root or scripts/. Use --validate-only to check input
paths, columns, and coverage without drawing figures.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'cross_risk': 'figure_07_cross_risk_feasibility',
    'association': 'figure_08_displacement_outcomes',
    'calibration': 'figure_EC02_tail_calibration',
}

def resolve_cli(path: Path) -> Path:
    """Existing relative paths prefer cwd; otherwise resolve against repository."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return path.resolve() if path.exists() else (ROOT / path).resolve()

def select_one(frame: pd.DataFrame, **keys: Any) -> pd.Series:
    selected = frame
    for key, value in keys.items():
        if key not in frame:
            raise ValueError(f'Missing column: {key}')
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f'Expected one row for {keys}, found {len(selected)}')
    return selected.iloc[0]

def checked_read(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing result table: {path}')
    frame = pd.read_csv(path)
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    return frame

def load_inputs(diagnostics_root: Path, tag: str, expected_seeds: int, expected_months: int):
    sources = {
        'cross_risk': diagnostics_root/'summary'/tag/'cross_risk_feasibility_summary.csv',
        'association': diagnostics_root/'summary'/tag/'monthly_displacement_outcomes.csv',
        'calibration': diagnostics_root/'audit'/'core_tail_calibration_excess_corrected.csv',
    }
    cross = checked_read(sources['cross_risk'], {
        'score','action','source_map','evaluation_map','metric','mean','ci_low','ci_high','n_seeds'})
    rows = []
    for action in ('hard','mix'):
        for source, dest in (('endogenous','hier_full_risk'), ('hier_full_risk','endogenous')):
            row = select_one(cross, score='post_cost_utility', action=action,
                             source_map=source, evaluation_map=dest, metric='infeasible')
            if int(row['n_seeds']) != expected_seeds:
                raise ValueError('Cross-risk coverage differs from --expected-seeds')
            values = row[['mean','ci_low','ci_high']].to_numpy(float)
            if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
                raise ValueError('Invalid cross-risk mean or confidence interval')
            rows.append(row)
    cross_plot = pd.DataFrame(rows).reset_index(drop=True)
    monthly = checked_read(sources['association'], {
        'score','action','left_map','right_map','scope','outcome_date',
        'allocation_displacement','net_excess_return_difference'})
    monthly = monthly.loc[
        monthly['score'].eq('post_cost_utility') & monthly['action'].eq('mix')
        & monthly['left_map'].eq('endogenous') & monthly['right_map'].eq('hier_full_risk')
        & monthly['scope'].eq('one_step_convex_parent')].copy()
    monthly['outcome_date'] = pd.to_datetime(monthly['outcome_date'], errors='raise')
    monthly = monthly.sort_values('outcome_date')
    if len(monthly) != expected_months or monthly['outcome_date'].duplicated().any():
        raise ValueError('Expected one seed-averaged row per historical month')
    if not np.isfinite(monthly[['allocation_displacement','net_excess_return_difference']].to_numpy(float)).all():
        raise ValueError('Nonfinite monthly association observations')
    tail = checked_read(sources['calibration'], {
        'strategy','n_seeds','n_valid','gross_excess_var_exceedance_rate',
        'mean_predicted_excess_cvar','actual_cvar_binding_rate'})
    strategy_order = [
        ('standalone__M_HierMacro','HierMacro expert'),
        ('post_cost_utility__mix','Utility weights + linear pool'),
        ('fixed_equal__mix','Fixed pool'),
        ('post_cost_utility__hard','Utility weights + model sampling'),
        ('predictive_log__mix','Log weights + linear pool'),
        ('predictive_log__hard','Log weights + model sampling'),
        ('standalone__M_FF5MOM','FF5MOM expert'),
    ]
    tr = []
    for strategy,label in strategy_order:
        row = select_one(tail, strategy=strategy).copy()
        if int(row['n_seeds']) != expected_seeds or int(row['n_valid']) != expected_seeds*expected_months:
            raise ValueError(f'Unexpected tail-diagnostic coverage for {strategy}')
        value = float(row['gross_excess_var_exceedance_rate'])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid tail exceedance frequency')
        row['label'] = label; tr.append(row)
    return sources, cross_plot, monthly, pd.DataFrame(tr)

def draw_figures(cross: pd.DataFrame, monthly: pd.DataFrame, tail: pd.DataFrame,
                 output_dir: Path, dpi: int, formats: list[str]) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
    try:
        font_manager.findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
        family = 'Times New Roman'
    except ValueError:
        family = 'Liberation Serif'
    params = {
        'font.family': family, 'font.size': 10, 'font.weight': 'normal',
        'axes.labelweight': 'normal', 'axes.titleweight': 'normal',
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    }
    written = []
    output_dir.mkdir(parents=True, exist_ok=True)
    def save(fig, key):
        for ext in formats:
            path = output_dir/f'{NAMES[key]}.{ext}'
            tmp = path.with_name(path.name+'.tmp')
            fig.savefig(tmp, format=ext, dpi=dpi, bbox_inches='tight', facecolor='white')
            os.replace(tmp, path); written.append(path)
        plt.close(fig)
    def despine(ax):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_linewidth(.6)
    with plt.rc_context(params):
        fig, ax = plt.subplots(figsize=(6.8,3.8))
        labels = [
            'Model sampling: endogenous → common risk',
            'Model sampling: common risk → endogenous',
            'Linear pool: endogenous → common risk',
            'Linear pool: common risk → endogenous',
        ]
        colors = ['#8C8C8C', '#8C8C8C', '#4D4D4D', '#4D4D4D']
        y = np.arange(4); means = cross['mean'].to_numpy(float)*100
        lows = cross['ci_low'].to_numpy(float)*100; highs = cross['ci_high'].to_numpy(float)*100
        ax.barh(y, means, height=.52, alpha=1, color=colors)
        ax.errorbar(means,y,xerr=np.vstack([means-lows,highs-means]),fmt='none', color='black',
                    capsize=3,elinewidth=.8,capthick=.8)
        for yi,mean,hi in zip(y,means,highs):
            ax.text(hi+1,yi,f'{mean:.1f}%',va='center',fontsize=9)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,max(highs)*1.20)
        ax.set_xlabel('Portfolios violating the alternative CVaR constraint (%)')
        ax.tick_params(axis='y',length=0,pad=6)
        ax.grid(axis='x',linewidth=.4,alpha=.3);ax.set_axisbelow(True)
        despine(ax)
        fig.subplots_adjust(left=.36,right=.98,top=.96,bottom=.18)
        save(fig,'cross_risk')
        # plot 83 calendar means; seed-months are not independent observations
        fig, ax = plt.subplots(figsize=(6.5,4.0))
        ax.scatter(monthly['allocation_displacement']*100,
                   monthly['net_excess_return_difference']*100,s=18,alpha=.5, color='black')
        ax.axhline(0,linewidth=.65, color='black')
        ax.set_xlabel('Allocation displacement (% of portfolio mass)')
        ax.set_ylabel('Next-month excess-return difference (pp)')
        ax.xaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(linewidth=.4,alpha=.25);ax.set_axisbelow(True)
        despine(ax)
        fig.subplots_adjust(left=.16,right=.98,top=.97,bottom=.20)
        save(fig,'association')
        fig, ax = plt.subplots(figsize=(6.5,3.6))
        values = tail['gross_excess_var_exceedance_rate'].to_numpy(float)*100
        y=np.arange(len(tail));ax.barh(y,values,height=.58,alpha=.80, color='#8C8C8C', edgecolor='black', linewidth=.6)
        ax.axvline(5,linestyle='--',linewidth=.9,color='black')
        for yi,val in zip(y,values):ax.text(val+.5,yi,f'{val:.1f}%',va='center',fontsize=9)
        ax.set_yticks(y,tail['label']);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.17)
        ax.set_xlabel('Realized loss beyond the predicted 95% VaR (% of months)')
        ax.tick_params(axis='y',length=0,pad=6);ax.legend(frameon=False,loc='lower right',fontsize=9)
        ax.grid(axis='x',linewidth=.4,alpha=.3);ax.set_axisbelow(True)
        despine(ax)
        fig.subplots_adjust(left=.24,right=.98,top=.97,bottom=.20)
        save(fig,'calibration')
    return written

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=ROOT/'work/config/generated_study_config.yaml')
    parser.add_argument('--diagnostics-root',type=Path,help='Override root containing audit/ and summary/<tag>/')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--tag',default='primary')
    parser.add_argument('--expected-seeds',type=int,default=30)
    parser.add_argument('--expected-months',type=int,default=83)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--formats',default='png,pdf')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.expected_seeds<2 or args.expected_months<1:parser.error('Invalid coverage requirements')
    if Path(args.tag).name != args.tag:parser.error('--tag must be a simple directory name')
    formats=[x.strip().lower() for x in args.formats.split(',') if x.strip()]
    if not formats or any(x not in {'png','pdf','svg'} for x in formats):parser.error('Formats must be png,pdf,svg')
    if args.diagnostics_root:
        diagnostics_root=resolve_cli(args.diagnostics_root)
        output_dir=resolve_cli(args.output_dir) if args.output_dir else ROOT/'work/outputs/paper_figures'
    else:
        import yaml
        config=resolve_cli(args.config)
        if not config.is_file():raise FileNotFoundError(f'Missing config: {config}')
        cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
        output_root=Path(cfg['paths']['output_root']).expanduser()
        if not output_root.is_absolute():output_root=ROOT/output_root
        diagnostics_root=output_root/'diagnostics'
        output_dir=resolve_cli(args.output_dir) if args.output_dir else output_root/'paper_figures'
    sources,cross,monthly,tail=load_inputs(diagnostics_root,args.tag,args.expected_seeds,args.expected_months)
    report={
        'tag':args.tag,'n_seeds':args.expected_seeds,'n_historical_months':len(monthly),
        'sources':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in sources.items()},
        'cross_risk_values':cross[['action','source_map','evaluation_map','mean','ci_low','ci_high']].to_dict('records'),
        'tail_frequencies':tail[['strategy','gross_excess_var_exceedance_rate']].to_dict('records'),
        'outputs':[],
    }
    if args.validate_only:
        print(json.dumps(report,indent=2));print('Validation passed. No plots were drawn or files overwritten.');return
    report['outputs']=[str(p) for p in draw_figures(cross,monthly,tail,output_dir,args.dpi,formats)]
    (output_dir/'manuscript_revision_figures_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
