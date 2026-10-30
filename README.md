# Reproducing the study: run guide

Run all commands from the project root using `config/confirmatory.yaml`. The primary study uses 2,000 scenarios per expert and month, 30 master seeds, and 83 monthly outcomes. The paths in the configuration must point to the supplied or separately obtained original inputs.

## Folder layout

```text
scripts/                            Scripts
src/g3_confirmatory/                Engine
config/confirmatory.yaml            Original experiment; paths, cache mode, multiprocessing settings
input_data/                         Place the five data files
docs/                               Provenance
requirements*.txt                   Dependencies and recorded historical package versions
```

## Important config notes
- Multiprocessing can be used for the experiments. The default setting is 1 worker.
- Using cache options like 'readwrite' may speed up the future steps of the process, but the whole pipeline (with robustness and other non-core tests), being already computationally heavy, would take up ~280 GB, which is why using cache is not recommended. The default settings are 'off'.

## 1. Set up the inputs

Create a virtual environment:

```bash
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux), then install dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` supplies the dependencies for every command in this guide.
Check the presence of these five source files in `input_data/`:
- `Meta.csv`
- `Portfolio Macro Data.xlsx`
- `adjusted_close_prices.csv`
- `Volume.csv`
- `ff5_plus_momentum_monthly_2016_2025.csv`

For another input location, update both the builder's `--input-dir` argument and the corresponding paths in `config/confirmatory.yaml`. Prepare the study data and run the initial checks:

```bash
python scripts/00_build_from_five_files.py --input-dir input_data --work-root work
python scripts/01_validate_inputs.py --config config/confirmatory.yaml
python scripts/09_validate_theory.py --config config/confirmatory.yaml
python scripts/10_archive_audit.py --config config/confirmatory.yaml
```

The builder creates prepared data in `work/data/`. Use a clean `work/` directory for a fresh reproduction; do not use `--overwrite` unless intentionally rebuilding existing results.

## 2. Run the core and primary robustness experiments

The frozen main inference family consists of the core, five mean treatments, the original robustness cells, the risk-map decomposition, and the Shapley experiment. Its 37 experiments are declared in `docs/expected_37_experiments.csv`.

```bash
python scripts/02_run_core.py --config config/confirmatory.yaml
python scripts/03_run_mean_ablation.py --config config/confirmatory.yaml
python scripts/05_run_robustness.py --config config/confirmatory.yaml --families scenario_count,transaction_cost,position_cap,cvar_alpha,cvar_tail_resolution,adjustment_penalty,eligibility_rule,missing_outcome_return
python scripts/18_run_full_riskmap_decomposition.py --config config/confirmatory.yaml --no-weights
python scripts/17_run_decision_shapley.py --config config/confirmatory.yaml --family mean --mean-caps 0.10 --no-weights
```

Summarizing the decompositions, running the declared main-family inference, and calculating the core risk diagnostics:

```bash
python scripts/19_summarize_full_riskmap_decomposition.py --config config/confirmatory.yaml
python scripts/18_summarize_decision_shapley.py --config config/confirmatory.yaml
python scripts/06_run_inference.py --config config/confirmatory.yaml
python scripts/20_build_risk_diagnostics.py --config config/confirmatory.yaml
```

## 3. Run the additional scenario-resolution checks for Table 10

Table 10 compares 250, 500, 1,000 and 2,000 scenarios under endogenous and common-risk constraints. Four additional configurations, each with 30 seeds, supply the 500-scenario endogenous cell and the 250-, 500- and 1,000-scenario common-risk cells. The other scenario cells come from the core, primary robustness and risk-map experiments:

```bash
python scripts/05_run_robustness.py --config config/confirmatory.yaml --families scenario_resolution_additions,scenario_resolution_common_risk
```

The two additional families are defined in `src/g3_confirmatory/robustness.py`. The 97.5%-CVaR/1,000-scenario experiment is included in the `cvar_tail_resolution` family run in section 2.

## 4. Run feasible-set diagnostics and supplementary extensions

Feasible-set comparisons and cross-risk displacement:

```bash
python scripts/22_audit_diagnostics.py --config config/confirmatory.yaml
python scripts/23_run_feasible_set_displacement.py --config config/confirmatory.yaml --allow-refit --tag primary
python scripts/25_summarize_displacement.py --config config/confirmatory.yaml --tag primary
```

Additional expert pools, one-seed mean shrinkage, and alternative covariance experts:

```bash
python scripts/24_run_diagnostic_extensions.py --config config/confirmatory.yaml --family small_pool
python scripts/28_run_frozen_posterior_shrinkage.py --config config/confirmatory.yaml --seed 20260322 --allow-refit
python scripts/29_run_covariance_regularization.py --config config/confirmatory.yaml
python scripts/30_summarize_diagnostic_extensions.py --config config/confirmatory.yaml
```

Script 30 summarizes the saved extension returns, preserving the performance-metric labels and leaving confidence limits blank when only one finite seed value is available. It also creates `work/outputs/diagnostics/extension_bundle.zip`.

If the extension experiments have already finished, apply the script 30 reporting correction by rerunning only:

```bash
python scripts/30_summarize_diagnostic_extensions.py --config config/confirmatory.yaml
```

This refreshes the extension summary CSVs and their archive from the existing seed results. The correction does not require rerunning scripts 24, 28 or 29. Rebuild the final paper bundle in section 6 after updating the reports.

## 5. Generate paper tables and figures

Build the standard tables, standard figures, and supplementary manuscript tables:

```bash
python scripts/07_build_tables.py --config config/confirmatory.yaml
python scripts/08_build_figures.py --config config/confirmatory.yaml
python scripts/27_export_manuscript_support_tables.py --config config/confirmatory.yaml
python scripts/32_export_extension_manuscript_tables.py --config config/confirmatory.yaml
```

Build Tables 2 and 10 with the additional exporters, then the four manuscript-specific figures (main Fig. 6; supplementary Figs. S1, S2 and S5):

```bash
python scripts/33_export_manuscript_tables.py --config config/confirmatory.yaml --tables 2,10
python scripts/34_build_manuscript_figures.py --config config/confirmatory.yaml --validate-only
python scripts/34_build_manuscript_figures.py --config config/confirmatory.yaml
```

Produce the remaining displacement and extension figures (including main Figs. 7–8 and supplementary Figs. S3–S4):

```bash
python scripts/26_build_manuscript_revision_figures.py --config config/confirmatory.yaml --tag primary
python scripts/31_build_extension_manuscript_figure.py --config config/confirmatory.yaml
```

The added reporting scripts do not run automatically from scripts 07–08.

## 6. Validate and package the results

```bash
python scripts/12_audit_run_settings.py --config config/confirmatory.yaml
python scripts/11_validate_outputs.py --config config/confirmatory.yaml
python scripts/21_build_paper_bundle.py --config config/confirmatory.yaml --destination work/paper_bundle
```

Build the bundle after scripts 27 and 31–34 so it includes their files under `paper_tables/` and `paper_figures/`. With the destination above, script 21 creates `work/paper_bundle/` and `work/paper_bundle.zip`.

The diagnostic archives remain separate:

| Archive | Created by | Contents |
|---|---|---|
| `work/paper_bundle.zip` | Script 21 | Paper tables and figures, inference, selected diagnostics, pooled seed panels and manifests |
| `work/outputs/diagnostics/displacement_bundle_primary.zip` | Script 25 | Displacement results, summaries and reproduction support |
| `work/outputs/diagnostics/extension_bundle.zip` | Script 30 | Extension results, summaries and reproduction support |

Scripts 25 and 30 do not collect the later files written by scripts 31–34 under `paper_tables/` and `paper_figures/`. Script 21 includes those files when it runs last, but it does not nest the two diagnostic ZIPs inside `paper_bundle.zip`. The diagnostic archives are, once again, separate.

The paper bundle is a results archive. Supply the source package and the five input files separately for a full reproduction; the bundle intentionally omits original market data, full holdings and caches.

## Where to find the results

| Content | Directory under `work/outputs/` |
|---|---|
| Main portfolios and individual seed returns | `core/` |
| Original and additional robustness experiments | `robustness/` |
| Mean treatments | `mean_ablation/` |
| Risk-map and Shapley decompositions | `decompositions/` |
| Main inference, SPA and multiplicity tests | `inference/` |
| Daily risk tests | `risk_diagnostics/` |
| Feasible-set tests and additional experts | `diagnostics/` |
| Paper tables and supplementary exports | `paper_tables/` |
| Tables 2 and 10 | `paper_tables/manuscript_additions/` |
| Paper figures | `paper_figures/` |