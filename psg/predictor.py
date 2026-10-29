"""Forward model ('cerebellum'): appearance + grasp -> predicted contact signature with honest variance.

Bootstrap ensemble of ridge regressors on random Fourier features. No torch needed, trains in
milliseconds on a few hundred grasps, and gives a variance that grows away from the training data,
which is what the gate and the residual's Kalman gain need. Swap for a heteroscedastic MLP later
if the data grows; keep the predict() signature.

Trained SUPERVISED on logged control-point labels. Never updated online (that is the residual's job).
"""
from __future__ import annotations

import pickle

import numpy as np

from .types import N_OUT


class ForwardModel:
    def __init__(self, n_features=512, n_members=8, lengthscale=None, ridge=1e-2, seed=0):
        self.n_features, self.n_members = n_features, n_members
        self.lengthscale, self.ridge = lengthscale, ridge
        self.rng = np.random.default_rng(seed)
        self.fitted = False

    # ---- features -------------------------------------------------------
    def _phi(self, X):
        Xs = (X - self.mu_x) / self.sd_x
        return np.sqrt(2.0 / self.n_features) * np.cos(Xs @ self.W + self.b)

    # ---- training -------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (n, d) percept vectors. Y: (n, N_OUT) log-space labels (NaN = missing)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.mu_x, self.sd_x = X.mean(0), X.std(0) + 1e-6
        d = X.shape[1]
        ls = self.lengthscale or np.sqrt(d)
        self.W = self.rng.normal(0, 1.0 / ls, (d, self.n_features))
        self.b = self.rng.uniform(0, 2 * np.pi, self.n_features)
        P = self._phi(X)
        n = len(X)
        self.coefs = np.zeros((self.n_members, N_OUT, self.n_features + 1))
        resid = [[] for _ in range(N_OUT)]
        for m in range(self.n_members):
            idx = self.rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), idx)
            for j in range(N_OUT):
                ok = idx[~np.isnan(Y[idx, j])]
                A = np.hstack([P[ok], np.ones((len(ok), 1))])
                w = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ Y[ok, j])
                self.coefs[m, j] = w
                o = oob[~np.isnan(Y[oob, j])]
                if len(o):
                    resid[j].extend(Y[o, j] - np.hstack([P[o], np.ones((len(o), 1))]) @ w)
        # aleatoric floor from out-of-bag residuals
        self.noise_sd = np.array([np.std(r) if len(r) > 3 else 0.1 for r in resid])
        self.fitted = True
        return self

    # ---- inference -------------------------------------------------------
    def predict(self, X: np.ndarray):
        """Returns mean (n, N_OUT) and std (n, N_OUT) in log space."""
        X = np.atleast_2d(np.asarray(X, float))
        A = np.hstack([self._phi(X), np.ones((len(X), 1))])
        preds = np.einsum("nf,mjf->mnj", A, self.coefs)        # (members, n, N_OUT)
        mean = preds.mean(0)
        std = np.sqrt(preds.var(0) + self.noise_sd ** 2)
        return mean, std

    # ---- persistence -----------------------------------------------------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "ForwardModel":
        m = cls.__new__(cls)
        with open(path, "rb") as f:
            m.__dict__.update(pickle.load(f))
        return m
