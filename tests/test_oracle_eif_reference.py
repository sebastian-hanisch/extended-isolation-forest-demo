"""Unabhängiges Orakel für den Extended Isolation Forest: eine naive rekursive Referenz nach dem Pseudocode von Hariri, Kind und Brunner (Python-Rekursion statt Ebenen-Schleife, eigene Zufallsziehung) liefert
dieselben mittleren Pfadlängen wie die Demo, bis auf das Monte-Carlo-Rauschen (gemessen an dem Rauschen zweier Läufe der Demo untereinander); bei ℓ = 0 stimmt der Wert mit scikit-learn überein."""

import math

import numpy as np
import pytest

import eif_algorithm as eif
import eif_isolation_forest as isf


def _c(n):
    if n > 2:
        return 2.0 * (math.log(n - 1) + 0.5772156649015329) - 2.0 * (n - 1) / n
    return 1.0 if n == 2 else 0.0


def _tree(X, e, limit, ext, rng):
    n, p = X.shape
    if e >= limit or n <= 1:
        return ("leaf", n)
    lo, hi = X.min(axis=0), X.max(axis=0)
    normal = rng.standard_normal(p)
    if p - ext - 1 > 0:
        normal[rng.choice(p, size=p - ext - 1, replace=False)] = 0.0
    point = rng.uniform(lo, hi)
    left = (X - point) @ normal <= 0
    return ("node", normal, point, _tree(X[left], e + 1, limit, ext, rng), _tree(X[~left], e + 1, limit, ext, rng))


def _path(x, t, e=0):
    if t[0] == "leaf":
        return e + _c(t[1])
    _, normal, point, left, right = t
    return _path(x, left if (x - point) @ normal <= 0 else right, e + 1)


def test_mean_path_lengths_equal_the_naive_recursive_reference_within_monte_carlo_noise():
    rng = np.random.default_rng(3)
    trees = 250
    for case, ext in enumerate((0, 1, 2)):
        p = 3
        X = rng.standard_normal((40, p)) @ rng.standard_normal((p, p))
        X[:4] += 5.0
        psi = 16
        limit = math.ceil(math.log2(psi))
        ref_rng = np.random.default_rng(100 + case)
        paths = np.zeros((trees, len(X)))
        for t in range(trees):
            idx = ref_rng.choice(len(X), size=psi, replace=False)
            tree = _tree(X[idx], 0, limit, ext, ref_rng)
            paths[t] = [_path(x, tree) for x in X]
        mean_ref = paths.mean(axis=0)
        a = eif.path_lengths(eif.fit_forest(X, trees, psi, seed=case, extension=ext), X).mean(axis=0)
        b = eif.path_lengths(eif.fit_forest(X, trees, psi, seed=case + 50, extension=ext), X).mean(axis=0)
        noise = float(np.sqrt(np.mean((a - b) ** 2)))                                   # Rauschen zweier Läufe der Demo untereinander
        assert float(np.sqrt(np.mean((a - mean_ref) ** 2))) < 2.0 * noise + 0.02
        assert abs(float(np.mean(a - mean_ref))) < 2.0 * noise + 0.02


def test_level_zero_matches_scikit_learn_isolation_forest_scores():
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(6)
    for i, psi in enumerate((16, 64)):
        X = rng.standard_normal((120, 3))
        X[:6] += 5.0
        ours = eif.score(eif.fit_forest(X, 300, psi, seed=i, extension=0), X)
        theirs = -ensemble.IsolationForest(n_estimators=300, max_samples=psi, random_state=i).fit(X).score_samples(X)
        assert abs(float(np.mean(ours - theirs))) < 0.01 and float(np.sqrt(np.mean((ours - theirs) ** 2))) < 0.02
    n_values = np.array([0, 1, 2, 3, 5, 10, 256])
    iforest = pytest.importorskip("sklearn.ensemble._iforest")
    assert np.allclose(isf.c_factor(n_values), iforest._average_path_length(n_values), atol=1e-12)
