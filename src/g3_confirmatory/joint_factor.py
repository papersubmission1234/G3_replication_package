"""Joint posterior for factor loadings B and residual variances Psi.

For each asset, r = B f + epsilon with diagonal residual covariance Psi.
Independent inverse-gamma priors on psi_i and conditional normal priors
B_i | psi_i ~ N(0, psi_i * Lambda_B**-1) give paired matrix-normal/IG draws.
Each draw implies mu = B f_t and Sigma = B Cov(f) B.T + Psi.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import invgamma


class JointFactorPosterior:
    def __init__(self, n_assets: int, n_factors: int,
                 a0: float = 2.0, b0: float = 0.005,
                 lambda_B: float = 1.0):
        self.n = n_assets
        self.k = n_factors
        self.a0, self.b0 = a0, b0
        self.Lambda_B_prior = lambda_B * np.eye(n_factors)
        self.B_post = np.zeros((n_assets, n_factors))
        self.Lambda_B_post = self.Lambda_B_prior.copy()
        self.a_post = np.full(n_assets, a0)
        self.b_post = np.full(n_assets, b0)
        self.f_mean = np.zeros(n_factors)
        self.f_cov = np.eye(n_factors)
        self._L_inv_chol = None
        self._is_fitted = False

    def fit(self, F: np.ndarray, R: np.ndarray):
        """Fit aligned factors F (T x k) and returns R (T x n).
        
        F must be standardized and complete. Missing returns are skipped per asset.
        """
        F = np.asarray(F, dtype=float)
        R = np.asarray(R, dtype=float)
        T, k = F.shape
        assert k == self.k
        n = R.shape[1]
        assert n == self.n

        B_post = np.zeros((n, k))
        a_post = np.empty(n)
        b_post = np.empty(n)
        for i in range(n):
            mask = ~np.isnan(R[:, i])
            T_i = mask.sum()
            if T_i < self.k + 1:
                B_post[i] = 0.0
                a_post[i] = self.a0
                b_post[i] = self.b0
                continue
            F_i = F[mask]
            r_i = R[mask, i]
            FtF = F_i.T @ F_i
            Lambda_i = self.Lambda_B_prior + FtF
            b_hat = np.linalg.solve(Lambda_i, F_i.T @ r_i)
            B_post[i] = b_hat
            resid = r_i - F_i @ b_hat
            rss = float((resid ** 2).sum())
            a_post[i] = self.a0 + T_i / 2.0
            prior_quad = float(b_hat @ self.Lambda_B_prior @ b_hat)
            b_post[i] = self.b0 + 0.5 * rss + 0.5 * prior_quad

        self.B_post = B_post
        # shared draw covariance approximates asset-specific missing-data posteriors
        dense_mask = ~np.all(np.isnan(R), axis=1)
        F_dense = F[dense_mask]
        self.Lambda_B_post = self.Lambda_B_prior + F_dense.T @ F_dense
        self.a_post = a_post
        self.b_post = b_post

        self.f_mean = F.mean(axis=0)
        self.f_cov = np.cov(F, rowvar=False, ddof=1)
        if self.f_cov.ndim == 0:
            self.f_cov = np.array([[float(self.f_cov)]])

        L_inv = np.linalg.inv(self.Lambda_B_post)
        L_inv = 0.5 * (L_inv + L_inv.T) + 1e-10 * np.eye(self.k)
        self._L_inv_chol = np.linalg.cholesky(L_inv)
        self._is_fitted = True

    def sample(self, x_factors: np.ndarray, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray]:
        """Draw paired mean and covariance given the current factor vector."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before sample().")

        psi_diag = invgamma.rvs(a=self.a_post, scale=self.b_post, random_state=rng)
        psi_diag = np.clip(psi_diag, 1e-8, None)

        Z = rng.standard_normal(size=(self.n, self.k))
        B_sample = self.B_post + np.sqrt(psi_diag)[:, None] * (Z @ self._L_inv_chol.T)

        x_factors = np.asarray(x_factors, dtype=float).ravel()
        mu = B_sample @ x_factors
        Sigma = B_sample @ self.f_cov @ B_sample.T + np.diag(psi_diag)
        Sigma = 0.5 * (Sigma + Sigma.T)
        return mu, Sigma

    def posterior_mean(self, x_factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Point estimate (no sampling): B_post f_t and B_post Cov(f) B_post' + diag(b_post/(a_post-1))."""
        x_factors = np.asarray(x_factors, dtype=float).ravel()
        mu = self.B_post @ x_factors
        a_safe = np.maximum(self.a_post, 1.01)
        psi_mean = self.b_post / (a_safe - 1.0)
        Sigma = self.B_post @ self.f_cov @ self.B_post.T + np.diag(psi_mean)
        Sigma = 0.5 * (Sigma + Sigma.T)
        return mu, Sigma
