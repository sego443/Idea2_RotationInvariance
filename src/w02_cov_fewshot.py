"""Lagged-covariance tangent-space classifiers for W02 strokes (recipes chosen on the W02 development wears).

Features: 0.5-12 Hz third-order zero-phase band-pass, temporal demeaning, covariance of the four channels stacked
at lags 0, 5, ..., 30 samples (28 x 28), mapped to the tangent space at a reference covariance (affine-invariant
log map, upper triangle with sqrt(2) off-diagonal weights).

Recipes (a "recording" is one 20-stroke trial recording; its first 8 strokes are the labeled prefix):
  rec_nc        support = the recording's prefix (2 per class); tangent at the support mean; nearest centroid after
                z-scoring with support statistics. Training-free.
  rec1_borrowed support = the first prefix stroke of each class; tangent at the prefix mean; Mahalanobis distance
                under a within-class covariance learned on training wears (PCA 30 + Ledoit-Wolf), class means
                refined twice with soft assignments of the recording's body strokes.
  wear_logreg   support = all prefixes of the wear; tangent at the support mean; logistic regression (C=0.1).
  zero_logreg   no labels from the test wear; ring channels cyclically shifted by -round(theta/90); tangent at the
                mean of all of the wear's strokes (unlabeled, transductive); logistic regression trained on the
                training wears, each re-centered at its own mean.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

FS = 250
BAND = (0.5, 12.0)
LAGS = (0, 5, 10, 15, 20, 25, 30)
RECIPES = ("rec_nc", "rec1_borrowed", "wear_logreg", "zero_logreg")


def covariances(windows: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, BAND, btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, windows, axis=1)
    x = x - x.mean(axis=1, keepdims=True)
    z = np.concatenate([np.roll(x, lag, axis=1) for lag in LAGS], axis=2)[:, max(LAGS):]
    covs = np.einsum("nti,ntj->nij", z, z) / z.shape[1]
    return covs + 1e-6 * np.trace(covs, axis1=1, axis2=2)[:, None, None] * np.eye(covs.shape[1])


def tangent(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    w, v = eigh(reference)
    isqrt = v @ np.diag(w ** -0.5) @ v.T
    iu = np.triu_indices(covs.shape[1])
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    out = []
    for c in covs:
        w2, v2 = eigh(isqrt @ c @ isqrt)
        out.append((v2 @ np.diag(np.log(np.maximum(w2, 1e-12))) @ v2.T)[iu] * weight)
    return np.array(out)


def cyclic_canonical(windows: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    shifts = np.floor(-np.asarray(theta_deg, dtype=float) / 90 + 0.5).astype(int) % 4  # round half up
    return np.stack([np.roll(w, -s, axis=1) for w, s in zip(windows, shifts)])


def nearest_centroid(train_x, train_y, test_x):
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    a, b = (train_x - mean) / std, (test_x - mean) / std
    centroids = np.stack([a[train_y == c].mean(0) for c in range(4)])
    return ((b[:, None, :] - centroids[None]) ** 2).sum(2).argmin(1)


def logistic(train_x, train_y, test_x):
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    model = LogisticRegression(C=0.1, max_iter=3000).fit((train_x - mean) / std, train_y)
    return model.predict((test_x - mean) / std)


def _recording_tangent_at_prefix(covs, data):
    out = np.empty((len(covs), covs.shape[1] * (covs.shape[1] + 1) // 2))
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        out[mask] = tangent(covs[mask], covs[mask & data["is_calibration"]].mean(0))
    return out


def borrowed_metric(train_covs, train):
    """PCA basis and within-class precision from labeled training recordings."""
    features = _recording_tangent_at_prefix(train_covs, train)
    pca = PCA(n_components=30).fit(features)
    z = pca.transform(features)
    residuals = [z[(train["group"] == g) & (train["y"] == c)] - z[(train["group"] == g) & (train["y"] == c)].mean(0)
                 for g in np.unique(train["group"]) for c in range(4)]
    return pca, LedoitWolf().fit(np.concatenate(residuals)).precision_


def predict(recipe: str, test: dict, train: dict | None = None, permute_seed: int | None = None) -> np.ndarray:
    """Predictions for the body strokes of `test` (-1 elsewhere). `train` holds labeled training wears.

    permute_seed permutes the test support labels within every support set (matched null for calibrated recipes).
    """
    if recipe not in RECIPES:
        raise ValueError(recipe)
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    labels = test["y"].copy()
    predicted = np.full(len(labels), -1)
    if recipe == "zero_logreg":
        covs = covariances(cyclic_canonical(test["X"], test["position"]))
        train_covs = covariances(cyclic_canonical(train["X"], train["position"]))
        train_x = np.concatenate([tangent(train_covs[train["wear_group"] == w],
                                          train_covs[train["wear_group"] == w].mean(0))
                                  for w in np.unique(train["wear_group"])])
        train_y = np.concatenate([train["y"][train["wear_group"] == w] for w in np.unique(train["wear_group"])])
        for wear in np.unique(test["wear_group"]):
            mask = test["wear_group"] == wear
            body = mask & ~test["is_calibration"]
            features = tangent(covs[mask], covs[mask].mean(0))
            predicted[body] = logistic(train_x, train_y, features[~test["is_calibration"][mask]])
        return predicted
    covs = covariances(test["X"])
    if recipe == "rec1_borrowed":
        pca, precision = borrowed_metric(covariances(train["X"]), train)
        z = pca.transform(_recording_tangent_at_prefix(covs, test))
    for group in np.unique(test["group"]):
        in_group = test["group"] == group
        body = np.flatnonzero(in_group & ~test["is_calibration"])
        if recipe == "wear_logreg":
            wear = test["wear_group"][in_group][0]
            support = np.flatnonzero((test["wear_group"] == wear) & test["is_calibration"])
        else:
            prefix = np.flatnonzero(in_group & test["is_calibration"])
            shots = 1 if recipe == "rec1_borrowed" else 2
            support = np.concatenate([prefix[labels[prefix] == c][:shots] for c in range(4)])
        support_y = labels[support] if rng is None else rng.permutation(labels[support])
        if recipe == "rec1_borrowed":
            means = np.stack([z[support][support_y == c].mean(0) for c in range(4)])
            onehot = np.eye(4)[support_y]
            for _ in range(2):
                diff = z[body][:, None, :] - means[None]
                d2 = np.einsum("nci,ij,ncj->nc", diff, precision, diff)
                weights = np.exp(-(d2 - d2.min(1, keepdims=True)) / 2)
                weights /= weights.sum(1, keepdims=True)
                means = (onehot.T @ z[support] + weights.T @ z[body]) / (onehot.sum(0) + weights.sum(0))[:, None]
            diff = z[body][:, None, :] - means[None]
            predicted[body] = np.einsum("nci,ij,ncj->nc", diff, precision, diff).argmin(1)
            continue
        reference = covs[support].mean(0)
        train_x, test_x = tangent(covs[support], reference), tangent(covs[body], reference)
        rule = nearest_centroid if recipe == "rec_nc" else logistic
        predicted[body] = rule(train_x, support_y, test_x)
    return predicted
