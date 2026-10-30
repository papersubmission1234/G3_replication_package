from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

DEFAULT_CLASS_STYLE_FAMILIES = {
    "equity": ("equity",),
    "bond": ("bond",),
    "mixed": ("equity", "bond"),
    "real_estate": ("equity",),
    "commodities": (),
    "crypto": ("equity",),
    "metals": (),
    "cash_fx": ("bond",),
}
import numpy as np
import pandas as pd

EPS = 1e-10
SUPPORTED_ASSET_CLASSES = ("equity", "bond", "mixed", "real_estate", "commodities", "crypto", "metals", "cash_fx")
DEFAULT_MACRO_COLUMNS = (
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
)
DEFAULT_EQUITY_STYLE_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
DEFAULT_BOND_STYLE_COLUMNS = ("duration",)
DEFAULT_MACRO_DIFF_COLUMNS = (
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
)
FactorToggleMap = dict[str, bool]


@dataclass
class HierarchicalBayesianModelConfig:
    min_obs: int = 24
    n_draws: int = 1000
    n_iter: int = 6
    random_state: int | None = 42
    forecast_mode: str = "mean_zero"
    alpha_prior_var: float = 1e-4
    macro_prior_var: float = 0.20
    style_prior_var: float = 0.25
    alpha_group_mean_var: float = 1e-4
    macro_group_mean_var: float = 0.20
    style_group_mean_var: float = 0.25
    coef_prior_spec: dict[str, Any] = field(default_factory=lambda: {"family": "gaussian"})
    residual_var_floor: float = 1e-6
    macro_columns: tuple[str, ...] | None = DEFAULT_MACRO_COLUMNS
    equity_style_columns: tuple[str, ...] = DEFAULT_EQUITY_STYLE_COLUMNS
    bond_style_columns: tuple[str, ...] = DEFAULT_BOND_STYLE_COLUMNS
    macro_diff_columns: tuple[str, ...] = DEFAULT_MACRO_DIFF_COLUMNS
    factor_toggles: FactorToggleMap | None = None
    factor_toggle_default: bool = True
    class_style_families: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_STYLE_FAMILIES)
    )


def _to_month_end_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = df.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _coerce_asset_class_series(
    asset_class_series: pd.Series,
    target_assets: pd.Index,
) -> pd.Series:
    out = asset_class_series.copy()
    out.index = out.index.astype(str)
    out = out.reindex(target_assets)
    out = out.where(out.isin(SUPPORTED_ASSET_CLASSES))
    return out


def _factor_is_enabled(
    factor_family: str,
    factor_name: str,
    factor_toggles: FactorToggleMap | None,
    default_enabled: bool = True,
) -> bool:
    if not factor_toggles:
        return bool(default_enabled)

    enabled = bool(default_enabled)
    for key in (
        factor_family,
        f"{factor_family}::*",
        factor_name,
        f"{factor_family}::{factor_name}",
    ):
        if key in factor_toggles:
            enabled = bool(factor_toggles[key])
    return enabled


def _resolve_factor_columns(
    requested_columns: tuple[str, ...] | list[str] | None,
    available_columns: pd.Index | list[str] | tuple[str, ...],
    factor_family: str,
    factor_toggles: FactorToggleMap | None,
    factor_toggle_default: bool = True,
    strict: bool = True,
) -> list[str]:
    available = list(available_columns)
    selected = available if requested_columns is None else list(requested_columns)

    if strict:
        available_set = set(available)
        missing = [col for col in selected if col not in available_set]
        if missing:
            raise KeyError(f"Missing requested {factor_family} factor columns: {missing}")

    return [
        col
        for col in selected
        if _factor_is_enabled(
            factor_family=factor_family,
            factor_name=col,
            factor_toggles=factor_toggles,
            default_enabled=factor_toggle_default,
        )
    ]


def _transform_macro_block(
    macro_factors_df: pd.DataFrame,
    config: HierarchicalBayesianModelConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    macro_factors_df = _to_month_end_index(macro_factors_df).sort_index()
    macro_columns = _resolve_factor_columns(
        requested_columns=config.macro_columns,
        available_columns=macro_factors_df.columns,
        factor_family="macro",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
        strict=False,
    )
    macro_block = macro_factors_df.reindex(columns=macro_columns).copy()

    transform_map: dict[str, str] = {}
    for col in macro_block.columns:
        if col == "VIX" and col in set(config.macro_diff_columns):
            macro_block[col] = np.log(macro_block[col].where(macro_block[col] > 0.0)).diff()
            transform_map[col] = "log_diff"
        elif col in set(config.macro_diff_columns):
            macro_block[col] = macro_block[col].diff()
            transform_map[col] = "diff"
        else:
            transform_map[col] = "level"

    return macro_block, transform_map


def _standardize_block(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean_ = train_df.mean(axis=0)
    std_ = train_df.std(axis=0, ddof=0).replace(0.0, np.nan)
    z_df = (train_df - mean_) / std_
    valid_cols = z_df.columns[z_df.notna().any(axis=0)]
    return z_df.loc[:, valid_cols], mean_.loc[valid_cols], std_.loc[valid_cols]


def _sample_factor_draws(
    factor_history_z: pd.DataFrame,
    n_draws: int,
    forecast_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if factor_history_z.empty:
        return np.zeros((n_draws, 0), dtype=float)

    x = factor_history_z.to_numpy(dtype=float)
    p = x.shape[1]

    if forecast_mode == "mean_zero":
        return rng.normal(loc=0.0, scale=1.0, size=(n_draws, p))

    if forecast_mode != "ar1":
        raise ValueError("forecast_mode must be 'mean_zero' or 'ar1'.")

    draws = np.zeros((n_draws, p), dtype=float)
    for j in range(p):
        series = factor_history_z.iloc[:, j].dropna().astype(float)
        if len(series) < 3:
            draws[:, j] = rng.normal(loc=0.0, scale=1.0, size=n_draws)
            continue

        x_prev = series.iloc[:-1].to_numpy(dtype=float)
        x_next = series.iloc[1:].to_numpy(dtype=float)
        denom = float(np.dot(x_prev, x_prev))
        phi = 0.0 if denom <= EPS else float(np.dot(x_prev, x_next) / denom)
        phi = float(np.clip(phi, -0.99, 0.99))
        resid = x_next - phi * x_prev
        shock_std = float(np.std(resid, ddof=0))
        shock_std = max(shock_std, 0.25)
        mean_forecast = phi * float(series.iloc[-1])
        draws[:, j] = rng.normal(loc=mean_forecast, scale=shock_std, size=n_draws)

    return draws


def _prior_variance_vector(
    macro_columns: list[str],
    style_columns: list[str],
    config: HierarchicalBayesianModelConfig,
) -> np.ndarray:
    return np.array(
        [config.alpha_prior_var]
        + [config.macro_prior_var] * len(macro_columns)
        + [config.style_prior_var] * len(style_columns),
        dtype=float,
    )


def _group_prior_variance_vector(
    macro_columns: list[str],
    style_columns: list[str],
    config: HierarchicalBayesianModelConfig,
) -> np.ndarray:
    return np.array(
        [config.alpha_group_mean_var]
        + [config.macro_group_mean_var] * len(macro_columns)
        + [config.style_group_mean_var] * len(style_columns),
        dtype=float,
    )


def _normalize_coef_prior_spec(prior_spec: dict[str, Any] | None) -> dict[str, Any]:
    spec = dict({"family": "gaussian"} if prior_spec is None else prior_spec)
    family = str(spec.get("family", "gaussian")).lower()

    if family == "gaussian":
        pass
    elif family == "laplace":
        spec.setdefault("max_iter", 1000)
        spec.setdefault("tol", 1e-6)
        spec.setdefault("penalty_floor", 1e-4)
    elif family == "horseshoe_approx":
        spec.setdefault("global_scale", 1.0)
        spec.setdefault("local_scale", 1.0)
        spec.setdefault("max_iter", 50)
        spec.setdefault("tol", 1e-6)
        spec.setdefault("max_penalty", 1e6)
        spec.setdefault("init_ridge", 1e-4)
    elif family == "none":
        pass
    else:
        raise ValueError(f"Unsupported coefficient prior family: {family}")

    spec["family"] = family
    return spec


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def _fit_centered_gaussian_prior(
    X_aug: np.ndarray,
    y_centered: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    p_aug = X_aug.shape[1]
    xtx = X_aug.T @ X_aug
    xty = X_aug.T @ y_centered
    prior_prec_diag = 1.0 / np.maximum(prior_var, EPS)
    ridge = sigma2 * prior_prec_diag
    jitter = 1e-8 * max(float(np.trace(xtx) / max(p_aug, 1)), 1.0)
    theta = np.linalg.solve(xtx + np.diag(ridge + jitter), xty)

    return theta, prior_prec_diag, {
        "iterations": 1,
        "condition_number": float(np.linalg.cond(xtx)) if p_aug > 0 else 1.0,
    }


def _fit_centered_laplace_prior(
    X_aug: np.ndarray,
    y_centered: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    p_aug = X_aug.shape[1]
    b = np.sqrt(np.maximum(prior_var, EPS) / 2.0)
    alpha_vec = sigma2 / np.maximum(b, EPS)
    max_iter = int(spec.get("max_iter", 1000))
    tol = float(spec.get("tol", 1e-6))
    penalty_floor = float(spec.get("penalty_floor", 1e-4))

    theta = np.zeros(p_aug, dtype=float)
    col_norm_sq = np.sum(X_aug * X_aug, axis=0)
    pred = X_aug @ theta

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        theta_old = theta.copy()

        for j in range(p_aug):
            if col_norm_sq[j] <= EPS:
                theta[j] = 0.0
                continue

            pred_without_j = pred - X_aug[:, j] * theta[j]
            rho_j = X_aug[:, j] @ (y_centered - pred_without_j)
            theta_new_j = _soft_threshold(rho_j, alpha_vec[j]) / col_norm_sq[j]

            pred = pred_without_j + X_aug[:, j] * theta_new_j
            theta[j] = theta_new_j

        if np.max(np.abs(theta - theta_old)) < tol:
            break

    abs_floor = np.maximum(np.sqrt(np.maximum(prior_var, EPS)) * penalty_floor, penalty_floor)
    prior_prec_diag = 1.0 / np.maximum(b * np.maximum(np.abs(theta), abs_floor), EPS)
    xtx = X_aug.T @ X_aug

    return theta, prior_prec_diag, {
        "iterations": n_iter,
        "condition_number": float(np.linalg.cond(xtx)) if p_aug > 0 else 1.0,
    }


def _fit_centered_horseshoe_prior(
    X_aug: np.ndarray,
    y_centered: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    p_aug = X_aug.shape[1]
    xtx = X_aug.T @ X_aug
    xty = X_aug.T @ y_centered

    global_scale = float(spec.get("global_scale", 1.0))
    local_scale = float(spec.get("local_scale", 1.0))
    max_iter = int(spec.get("max_iter", 50))
    tol = float(spec.get("tol", 1e-6))
    max_penalty = float(spec.get("max_penalty", 1e6))
    init_ridge = float(spec.get("init_ridge", 1e-4))

    jitter = 1e-8 * max(float(np.trace(xtx) / max(p_aug, 1)), 1.0)
    theta = np.linalg.solve(xtx + np.diag(np.full(p_aug, init_ridge + jitter)), xty)
    prior_scale = np.maximum(prior_var, EPS) * max(global_scale ** 2, EPS)

    n_iter = 0
    prior_prec_diag = np.full(p_aug, 1.0 / np.maximum(prior_var, EPS), dtype=float)
    for n_iter in range(1, max_iter + 1):
        theta_old = theta.copy()

        local_var = theta_old ** 2 + max(local_scale ** 2, EPS)
        prior_prec_diag = 1.0 / np.maximum(prior_scale * local_var, EPS)
        prior_prec_diag = np.clip(prior_prec_diag, 0.0, max_penalty)

        ridge = sigma2 * prior_prec_diag
        theta = np.linalg.solve(xtx + np.diag(ridge + jitter), xty)

        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta, prior_prec_diag, {
        "global_scale": global_scale,
        "local_scale": local_scale,
        "iterations": n_iter,
        "condition_number": float(np.linalg.cond(xtx)) if p_aug > 0 else 1.0,
    }


def _fit_centered_prior_map(
    X_aug: np.ndarray,
    y_vec: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    sigma2: float,
    prior_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_centered = y_vec - (X_aug @ prior_mean)
    family = prior_spec["family"]

    if family == "gaussian":
        return _fit_centered_gaussian_prior(
            X_aug=X_aug,
            y_centered=y_centered,
            sigma2=sigma2,
            prior_var=prior_var,
        )
    if family == "laplace":
        return _fit_centered_laplace_prior(
            X_aug=X_aug,
            y_centered=y_centered,
            sigma2=sigma2,
            prior_var=prior_var,
            spec=prior_spec,
        )
    if family == "horseshoe_approx":
        return _fit_centered_horseshoe_prior(
            X_aug=X_aug,
            y_centered=y_centered,
            sigma2=sigma2,
            prior_var=prior_var,
            spec=prior_spec,
        )
    if family == "none":
        theta = np.linalg.pinv(X_aug) @ y_centered
        prior_prec_diag = np.zeros_like(prior_var, dtype=float)
        return theta, prior_prec_diag, {"iterations": 1}

    raise ValueError(f"Unsupported coefficient prior family: {family}")


def _fit_asset_posterior(
    y: pd.Series,
    X: pd.DataFrame,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    residual_var_floor: float,
    coef_prior_spec: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prior_spec = _normalize_coef_prior_spec(coef_prior_spec)
    joined = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    if joined.shape[0] < 2:
        return None

    y_vec = joined["y"].to_numpy(dtype=float)
    X_mat = joined[X.columns].to_numpy(dtype=float)
    X_aug = np.column_stack([np.ones(len(joined)), X_mat])

    n_obs = X_aug.shape[0]
    p_aug = X_aug.shape[1]
    dof = max(n_obs - p_aug, 1)

    sigma2 = max(float(np.var(y_vec, ddof=1)), residual_var_floor)
    for _ in range(2):
        theta, prior_prec_diag, fit_meta = _fit_centered_prior_map(
            X_aug=X_aug,
            y_vec=y_vec,
            prior_mean=prior_mean,
            prior_var=prior_var,
            sigma2=sigma2,
            prior_spec=prior_spec,
        )
        posterior_mean = prior_mean + theta
        posterior_prec = (X_aug.T @ X_aug) / sigma2 + np.diag(prior_prec_diag)
        posterior_prec = posterior_prec + 1e-8 * np.eye(p_aug)
        posterior_cov = np.linalg.pinv(posterior_prec)

        resid = y_vec - X_aug @ posterior_mean
        sigma2 = max(float(np.dot(resid, resid) / dof), residual_var_floor)

    return {
        "posterior_mean": posterior_mean,
        "posterior_cov": posterior_cov,
        "residual_var": sigma2,
        "fitted": pd.Series(X_aug @ posterior_mean, index=joined.index, dtype=float),
        "residuals": pd.Series(y_vec - X_aug @ posterior_mean, index=joined.index, dtype=float),
        "n_obs": int(n_obs),
        "train_index": joined.index,
        "prior_family": prior_spec["family"],
        "fit_meta": fit_meta,
    }


def fit_hierarchical_bayesian_etf_model(
    monthly_excess_returns: pd.DataFrame,
    asset_class_series: pd.Series,
    macro_factors_df: pd.DataFrame,
    equity_factors_df: pd.DataFrame,
    bond_factors_df: pd.DataFrame,
    config: HierarchicalBayesianModelConfig | None = None,
) -> dict[str, Any]:
    config = HierarchicalBayesianModelConfig() if config is None else config
    coef_prior_spec = _normalize_coef_prior_spec(config.coef_prior_spec)
    rng = np.random.default_rng(config.random_state)

    monthly_excess_returns = _to_month_end_index(monthly_excess_returns).sort_index()
    equity_factors_df = _to_month_end_index(equity_factors_df).sort_index()
    bond_factors_df = _to_month_end_index(bond_factors_df).sort_index()
    macro_block_raw, macro_transform_map = _transform_macro_block(macro_factors_df, config)

    common_index = (
        monthly_excess_returns.index
        .intersection(macro_block_raw.index)
        .intersection(equity_factors_df.index)
        .intersection(bond_factors_df.index)
        .sort_values()
    )
    if len(common_index) < config.min_obs:
        raise ValueError("Not enough common monthly observations to fit the hierarchical model.")

    y = monthly_excess_returns.loc[common_index].copy()
    asset_class = _coerce_asset_class_series(asset_class_series, y.columns)
    supported_assets = asset_class.dropna().index
    y = y.loc[:, supported_assets]
    asset_class = asset_class.loc[supported_assets]

    macro_block_raw = macro_block_raw.loc[common_index]
    equity_style_columns = _resolve_factor_columns(
        requested_columns=config.equity_style_columns,
        available_columns=equity_factors_df.columns,
        factor_family="equity",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
    )
    bond_style_columns = _resolve_factor_columns(
        requested_columns=config.bond_style_columns,
        available_columns=bond_factors_df.columns,
        factor_family="bond",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
    )
    equity_style_raw = equity_factors_df.loc[common_index, equity_style_columns].copy()
    bond_style_raw = bond_factors_df.loc[common_index, bond_style_columns].copy()

    macro_z, macro_mean, macro_std = _standardize_block(macro_block_raw)
    equity_z, equity_mean, equity_std = _standardize_block(equity_style_raw)
    bond_z, bond_mean, bond_std = _standardize_block(bond_style_raw)

    common_index = (
        y.index
        .intersection(macro_z.index)
        .intersection(equity_z.index)
        .intersection(bond_z.index)
        .sort_values()
    )
    y = y.loc[common_index]
    macro_z = macro_z.loc[common_index]
    equity_z = equity_z.loc[common_index]
    bond_z = bond_z.loc[common_index]

    style_blocks = {
        "equity": equity_z.add_prefix("equity::"),
        "bond": bond_z.add_prefix("bond::"),
    }
    class_specs: dict[str, dict[str, Any]] = {}
    observed_classes = [
        cls for cls in SUPPORTED_ASSET_CLASSES if (asset_class == cls).any()
    ]
    for class_name in observed_classes:
        families = tuple(config.class_style_families.get(class_name, ()))
        unknown_families = [family for family in families if family not in style_blocks]
        if unknown_families:
            raise ValueError(
                f"Unknown style-factor families for {class_name}: {unknown_families}"
            )
        pieces = [style_blocks[family] for family in families]
        style_z = (
            pd.concat(pieces, axis=1)
            if pieces
            else pd.DataFrame(index=common_index)
        )
        class_specs[class_name] = {
            "style_z": style_z,
            "style_columns": style_z.columns.tolist(),
            "style_families": families,
        }

    asset_posterior_records: dict[str, dict[str, Any]] = {}
    group_mean_by_class: dict[str, pd.Series] = {}
    factor_draws_by_class: dict[str, np.ndarray] = {}
    factor_history_by_class: dict[str, pd.DataFrame] = {}
    factor_scaling = {
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "equity_style_mean": equity_mean,
        "equity_style_std": equity_std,
        "bond_style_mean": bond_mean,
        "bond_style_std": bond_std,
    }

    macro_z = macro_z.add_prefix("macro::")
    macro_columns = macro_z.columns.tolist()

    for asset_group, spec in class_specs.items():
        class_assets = asset_class[asset_class == asset_group].index.tolist()
        if not class_assets:
            continue

        X_class = pd.concat([macro_z, spec["style_z"]], axis=1)
        factor_history_by_class[asset_group] = X_class.copy()
        style_columns = spec["style_columns"]

        eligible_assets = [
            asset
            for asset in class_assets
            if y[asset].dropna().shape[0] >= config.min_obs
        ]
        if not eligible_assets:
            continue

        prior_var = _prior_variance_vector(macro_columns, style_columns, config)
        group_prior_var = _group_prior_variance_vector(macro_columns, style_columns, config)
        group_mean = np.zeros(1 + len(macro_columns) + len(style_columns), dtype=float)

        for _ in range(max(config.n_iter, 1)):
            fitted_assets: list[np.ndarray] = []
            for asset in eligible_assets:
                fit = _fit_asset_posterior(
                    y=y[asset],
                    X=X_class,
                    prior_mean=group_mean,
                    prior_var=prior_var,
                    residual_var_floor=config.residual_var_floor,
                    coef_prior_spec=coef_prior_spec,
                )
                if fit is None:
                    continue
                asset_posterior_records[asset] = {
                    **fit,
                    "asset_class": asset_group,
                    "feature_columns": ["alpha", *macro_columns, *style_columns],
                }
                fitted_assets.append(fit["posterior_mean"])

            if not fitted_assets:
                break

            fitted_mat = np.vstack(fitted_assets)
            post_prec = 1.0 / np.maximum(group_prior_var, EPS) + len(fitted_assets) / np.maximum(prior_var, EPS)
            post_rhs = fitted_mat.sum(axis=0) / np.maximum(prior_var, EPS)
            group_mean = post_rhs / post_prec

        group_mean_by_class[asset_group] = pd.Series(
            group_mean,
            index=["alpha", *macro_columns, *style_columns],
            dtype=float,
            name=asset_group,
        )
        factor_draws_by_class[asset_group] = _sample_factor_draws(
            factor_history_z=factor_history_by_class[asset_group],
            n_draws=config.n_draws,
            forecast_mode=config.forecast_mode,
            rng=rng,
        )

    if not asset_posterior_records:
        raise ValueError("No ETFs met the minimum observation threshold for model fitting.")

    asset_order = pd.Index(sorted(asset_posterior_records))
    posterior_draws = np.full((config.n_draws, len(asset_order)), np.nan, dtype=float)
    posterior_predictive_draws = np.full((config.n_draws, len(asset_order)), np.nan, dtype=float)

    for asset_idx, asset in enumerate(asset_order):
        fit = asset_posterior_records[asset]
        class_name = fit["asset_class"]
        feature_columns = fit["feature_columns"]
        factor_draws = factor_draws_by_class[class_name]
        coef_draws = rng.multivariate_normal(
            mean=fit["posterior_mean"],
            cov=fit["posterior_cov"] + 1e-8 * np.eye(len(feature_columns)),
            size=config.n_draws,
        )
        x_draws = np.column_stack([np.ones(config.n_draws), factor_draws])
        expected_return_draws = np.sum(x_draws * coef_draws, axis=1)
        posterior_draws[:, asset_idx] = expected_return_draws

        residual_std = float(np.sqrt(max(fit["residual_var"], config.residual_var_floor)))
        residual_shocks = rng.normal(loc=0.0, scale=residual_std, size=config.n_draws)
        posterior_predictive_draws[:, asset_idx] = expected_return_draws + residual_shocks

    posterior_draws_df = pd.DataFrame(
        posterior_draws,
        index=pd.RangeIndex(start=0, stop=config.n_draws, step=1, name="draw"),
        columns=asset_order,
    )
    posterior_predictive_draws_df = pd.DataFrame(
        posterior_predictive_draws,
        index=pd.RangeIndex(start=0, stop=config.n_draws, step=1, name="draw"),
        columns=asset_order,
    )
    posterior_mean = posterior_draws_df.mean(axis=0).rename("posterior_mean")
    posterior_interval = pd.DataFrame(
        {
            "lower_5": posterior_draws_df.quantile(0.05, axis=0),
            "median": posterior_draws_df.quantile(0.50, axis=0),
            "upper_95": posterior_draws_df.quantile(0.95, axis=0),
        }
    )
    posterior_predictive_interval = pd.DataFrame(
        {
            "lower_5": posterior_predictive_draws_df.quantile(0.05, axis=0),
            "median": posterior_predictive_draws_df.quantile(0.50, axis=0),
            "upper_95": posterior_predictive_draws_df.quantile(0.95, axis=0),
        }
    )

    loading_rows = []
    residual_var = {}
    observation_count = {}
    residual_history = pd.DataFrame(index=common_index, columns=asset_order, dtype=float)
    for asset in asset_order:
        fit = asset_posterior_records[asset]
        loading_rows.append(pd.Series(fit["posterior_mean"], index=fit["feature_columns"], name=asset))
        residual_var[asset] = float(fit["residual_var"])
        observation_count[asset] = int(fit["n_obs"])
        residual_history.loc[fit["residuals"].index, asset] = fit["residuals"].values

    factor_loadings = pd.DataFrame(loading_rows).sort_index()
    residual_variance = pd.Series(residual_var, name="residual_variance").sort_index()
    observation_count = pd.Series(observation_count, name="n_obs").sort_index()
    factor_history_z = pd.concat(
        [
            macro_z,
            equity_z.add_prefix("equity::"),
            bond_z.add_prefix("bond::"),
        ],
        axis=1,
    ).sort_index()

    asset_summary = (
        pd.DataFrame({"asset_class": asset_class.reindex(asset_order)})
        .join(observation_count, how="left")
        .join(residual_variance, how="left")
    )

    diagnostics = {
        "config": asdict(config),
        "train_start": common_index.min(),
        "train_end": common_index.max(),
        "n_months": int(len(common_index)),
        "n_assets_total": int(len(monthly_excess_returns.columns)),
        "n_assets_supported": int(len(supported_assets)),
        "n_assets_modeled": int(len(asset_order)),
        "asset_count_by_class": asset_class.reindex(asset_order).value_counts().to_dict(),
        "macro_transform_map": macro_transform_map,
        "macro_columns_used": macro_columns,
        "equity_style_columns_used": equity_z.columns.tolist(),
        "bond_style_columns_used": bond_z.columns.tolist(),
        "class_style_families": {k: list(v) for k, v in config.class_style_families.items()},
        "coef_prior_spec": coef_prior_spec,
        "draw_semantics": {
            "posterior_draws": "expected_return_draws",
            "posterior_predictive_draws": "full_predictive_return_draws",
        },
    }

    return {
        "posterior_draws": posterior_draws_df,
        "posterior_predictive_draws": posterior_predictive_draws_df,
        "posterior_mean": posterior_mean,
        "posterior_interval": posterior_interval,
        "posterior_predictive_interval": posterior_predictive_interval,
        "factor_loadings": factor_loadings,
        "residual_variance": residual_variance,
        "asset_summary": asset_summary,
        "group_mean_by_class": pd.DataFrame(group_mean_by_class).T.sort_index(),
        "factor_draws_by_class": factor_draws_by_class,
        "factor_history_z_by_class": factor_history_by_class,
        "factor_history_z": factor_history_z,
        "factor_scaling": factor_scaling,
        "residual_history": residual_history.sort_index(),
        "diagnostics": diagnostics,
    }


def diagnose_factor_relevance(
    result: dict[str, Any],
    loading_threshold: float = 0.05,
    min_active_share: float = 0.10,
    corr_threshold: float = 0.90,
) -> pd.DataFrame:
    factor_loadings_by_date = result.get("factor_loadings_by_date")
    if not factor_loadings_by_date:
        factor_loadings = result.get("factor_loadings")
        if isinstance(factor_loadings, pd.DataFrame) and not factor_loadings.empty:
            factor_loadings_by_date = {pd.NaT: factor_loadings}
        else:
            raise ValueError("Result does not contain factor loadings.")

    records: list[dict[str, Any]] = []
    for rebalance_date, loadings in factor_loadings_by_date.items():
        if not isinstance(loadings, pd.DataFrame) or loadings.empty:
            continue

        factor_matrix = loadings.drop(columns=["alpha"], errors="ignore").copy().astype(float)
        if factor_matrix.empty:
            continue

        abs_factor_matrix = factor_matrix.abs()
        loading_corr = factor_matrix.corr().abs()

        for factor_name in factor_matrix.columns:
            corr_peers = pd.Series(dtype=float)
            if factor_name in loading_corr.index:
                corr_peers = loading_corr.loc[factor_name].drop(index=factor_name, errors="ignore").dropna()

            records.append(
                {
                    "date": rebalance_date,
                    "factor": factor_name,
                    "factor_family": factor_name.split("::", 1)[0] if "::" in factor_name else "other",
                    "n_assets": int(abs_factor_matrix[factor_name].notna().sum()),
                    "mean_abs_loading": float(abs_factor_matrix[factor_name].mean()),
                    "median_abs_loading": float(abs_factor_matrix[factor_name].median()),
                    "max_abs_loading": float(abs_factor_matrix[factor_name].max()),
                    "share_assets_above_threshold": float((abs_factor_matrix[factor_name] >= loading_threshold).mean()),
                    "avg_abs_loading_corr_with_other_factors": float(corr_peers.mean()) if not corr_peers.empty else np.nan,
                    "max_abs_loading_corr_with_other_factors": float(corr_peers.max()) if not corr_peers.empty else np.nan,
                }
            )

    if not records:
        raise ValueError("No factor records were available for diagnostics.")

    snapshot_table = pd.DataFrame(records)
    summary = (
        snapshot_table.groupby("factor", dropna=False)
        .agg(
            factor_family=("factor_family", "last"),
            n_snapshots=("date", "size"),
            n_assets=("n_assets", "max"),
            mean_abs_loading=("mean_abs_loading", "mean"),
            median_abs_loading=("median_abs_loading", "mean"),
            max_abs_loading=("max_abs_loading", "max"),
            share_assets_above_threshold=("share_assets_above_threshold", "mean"),
            avg_abs_loading_corr_with_other_factors=("avg_abs_loading_corr_with_other_factors", "mean"),
            max_abs_loading_corr_with_other_factors=("max_abs_loading_corr_with_other_factors", "max"),
        )
        .sort_index()
    )

    summary["weak_loading_flag"] = summary["share_assets_above_threshold"] < float(min_active_share)
    summary["redundant_flag"] = summary["max_abs_loading_corr_with_other_factors"] >= float(corr_threshold)
    flag_count = summary["weak_loading_flag"].astype(int) + summary["redundant_flag"].astype(int)
    summary["recommendation"] = np.where(flag_count >= 2, "drop_candidate", np.where(flag_count == 1, "review", "keep"))
    summary["reason"] = summary.apply(
        lambda row: ", ".join(
            reason
            for reason, active in [
                ("low cross-sectional loading", bool(row["weak_loading_flag"])),
                ("high overlap with other factors", bool(row["redundant_flag"])),
            ]
            if active
        )
        or "no warning flags",
        axis=1,
    )
    summary = summary.sort_values(
        by=["recommendation", "share_assets_above_threshold", "mean_abs_loading"],
        ascending=[True, True, True],
    )
    return summary


def build_factor_pruned_config(
    config: HierarchicalBayesianModelConfig,
    macro_factors_df: pd.DataFrame,
    factors_to_drop: list[str] | tuple[str, ...] | pd.Index,
) -> HierarchicalBayesianModelConfig:
    drop_set = {str(name) for name in factors_to_drop}
    macro_columns = _resolve_factor_columns(
        requested_columns=config.macro_columns,
        available_columns=macro_factors_df.columns,
        factor_family="macro",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
        strict=False,
    )
    equity_columns = _resolve_factor_columns(
        requested_columns=config.equity_style_columns,
        available_columns=DEFAULT_EQUITY_STYLE_COLUMNS,
        factor_family="equity",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
        strict=False,
    )
    bond_columns = _resolve_factor_columns(
        requested_columns=config.bond_style_columns,
        available_columns=DEFAULT_BOND_STYLE_COLUMNS,
        factor_family="bond",
        factor_toggles=config.factor_toggles,
        factor_toggle_default=config.factor_toggle_default,
        strict=False,
    )

    kept_macro_columns = [col for col in macro_columns if f"macro::{col}" not in drop_set]
    kept_equity_columns = [col for col in equity_columns if f"equity::{col}" not in drop_set]
    kept_bond_columns = [col for col in bond_columns if f"bond::{col}" not in drop_set]

    return replace(
        config,
        macro_columns=tuple(kept_macro_columns),
        equity_style_columns=tuple(kept_equity_columns),
        bond_style_columns=tuple(kept_bond_columns),
    )
