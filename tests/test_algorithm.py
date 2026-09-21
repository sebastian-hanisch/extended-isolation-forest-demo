"""Extended Isolation Forest: Baumstruktur, Hyperebenen (Handinstanzen), ℓ = 0 gegen den Isolation Forest, Erweiterungsgrad, Pfadlängen, Score gegen scikit-learn, Skalen- und Drehungsverhalten,
Determinismus; die kopierten Komponenten der Vorgänger (Isolation Forest, χ², MCD)."""

import numpy as np
import pytest
from scipy.stats import chi2, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

import eif_algorithm as eif
import eif_ee_algorithm as alg
import eif_evaluation as ev
import eif_isolation_forest as isf


def _data(n=300, p=4, n_out=15, shift=8.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.standard_normal((n - n_out, p)), shift + 0.5 * rng.standard_normal((n_out, p))])
    return X, np.arange(n) >= n - n_out


def _route(tree, X):
    node = np.zeros(len(X), dtype=int)
    for _ in range(64):
        internal = tree.split[node]
        if not internal.any():
            break
        proj = ((X - tree.point[node]) * tree.normal[node]).sum(axis=1)
        node = np.where(internal, np.where(proj <= 0, tree.left[node], tree.right[node]), node)
    return node


# --- Baum ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 3])
def test_tree_structure_sizes_normals_and_depth(level):
    X, _ = _data(n=64, n_out=4)
    tree = eif.build_tree(X, np.random.default_rng(1), isf.height_limit(64), level)
    inner = np.flatnonzero(tree.split)
    assert tree.size[0] == 64 and tree.depth.max() <= isf.height_limit(64)
    for node in inner:
        assert tree.size[node] == tree.size[tree.left[node]] + tree.size[tree.right[node]]
        assert tree.depth[tree.left[node]] == tree.depth[node] + 1
        assert 1 <= np.count_nonzero(tree.normal[node]) <= level + 1                              # höchstens ℓ + 1 Komponenten sind ungleich null
    leaves = np.flatnonzero(~tree.split)
    assert np.array_equal(np.bincount(_route(tree, X), minlength=len(tree.size))[leaves], tree.size[leaves])              # Blattgrößen = wirklich dort landende Punkte
    assert (tree.normal[leaves] == 0).all()


def test_level_zero_uses_exactly_one_feature_per_split_and_full_level_uses_all():
    X, _ = _data(n=128, p=5)
    axis = eif.build_tree(X, np.random.default_rng(2), 7, 0)
    assert all(np.count_nonzero(axis.normal[k]) == 1 for k in np.flatnonzero(axis.split))
    full = eif.build_tree(X, np.random.default_rng(2), 7, 4)
    counts = [np.count_nonzero(full.normal[k]) for k in np.flatnonzero(full.split)]
    assert min(counts) >= 3 and max(counts) == 5                                                 # bei vollem Grad nahezu alle Komponenten (Gauß'sch, nie exakt null)


def test_extension_is_capped_at_p_minus_one():
    X, _ = _data(n=64, p=3)
    tree = eif.build_tree(X, np.random.default_rng(0), 6, 99)
    assert max(np.count_nonzero(tree.normal[k]) for k in np.flatnonzero(tree.split)) <= 3
    assert eif.fit_forest(X, 3, 32, 0, 99).extension == 2 and eif.fit_forest(X, 3, 32, 0, -4).extension == 0


def test_split_side_is_the_hyperplane_side_hand_instance():
    """Zwei Gruppen im Plan: n = (1, 1) trennt entlang der Diagonalen; q liegt zwischen den Gruppen."""
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 0.9]])
    tree = eif.build_tree(X, np.random.default_rng(5), 3, 1)
    root_n, root_q = tree.normal[0], tree.point[0]
    left = ((X - root_q) * root_n).sum(axis=1) <= 0
    assert tree.split[0] and tree.size[tree.left[0]] == left.sum() and tree.size[tree.right[0]] == (~left).sum()
    assert np.array_equal(_route(tree, X) == _route(tree, X), np.ones(4, bool))
    steps, size = eif.point_path(tree, X[0])
    assert size >= 1 and len(steps) == int(tree.depth[_route(tree, X[:1])[0]])


def test_identical_points_stay_in_one_leaf_and_constant_features_are_ignored():
    tree = eif.build_tree(np.ones((10, 3)), np.random.default_rng(0), 5, 2)
    assert len(tree.size) == 1 and not tree.split[0]
    assert eif.tree_path_length(tree, np.ones((1, 3)))[0] == pytest.approx(float(isf.c_factor(10)))
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal(20), np.full(20, 5.0)])
    t = eif.build_tree(X, rng, 40, 1)
    assert (t.normal[np.flatnonzero(t.split)][:, 1] == 0).all()                                 # die konstante Spalte bekommt nie eine Komponente
    assert (t.size[~t.split] == 1).all()


def test_path_length_equals_the_explicit_walk():
    X, _ = _data(n=100)
    tree = eif.build_tree(X, np.random.default_rng(2), isf.height_limit(100), 2)
    lengths = eif.tree_path_length(tree, X)
    for i in (0, 10, 55, 99):
        steps, size = eif.point_path(tree, X[i])
        assert lengths[i] == pytest.approx(len(steps) + float(isf.c_factor(size)))
    assert lengths.min() >= 1 and lengths.max() <= isf.height_limit(100) + float(isf.c_factor(100))


# --- Wald und Score ------------------------------------------------------------------------------------------------------------------------


def test_scores_are_in_the_unit_interval_and_anomalies_score_higher():
    X, an = _data()
    F = eif.fit_forest(X, 100, 256, 0, 3)
    s = eif.score(F, X)
    assert ((s > 0) & (s < 1)).all() and s[an].min() > s[~an].mean() and ev.roc_auc(s, an) > 0.99
    assert eif.score(F, np.full((1, 4), 40.0))[0] > 0.6


def test_forest_shape_psi_cap_and_seed_behaviour():
    X, _ = _data(n=50, n_out=3)
    F = eif.fit_forest(X, 7, 256, 0, 2)
    assert F.psi == 50 and len(F.trees) == 7 and F.extension == 2 and all(len(s) == 50 for s in F.subsets) and eif.path_lengths(F, X).shape == (7, 50)
    a, b, c = (eif.score(eif.fit_forest(X, 20, 50, seed, 2), X) for seed in (3, 3, 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(eif.score(eif.fit_forest(X, 20, 50, 3, 2), X), isf.score(isf.fit_forest(X, 20, 50, 3), X))         # verschiedene Zufallsströme von IF und EIF


def test_level_zero_is_the_isolation_forest_in_distribution():
    """ℓ = 0 gegen den Isolation Forest (eigene Implementierung und scikit-learn): gleiche Skala, Rangkorrelation, gleiche AUC."""
    for seed in (0, 1, 2):
        X, an = _data(seed=seed)
        ours0 = eif.score(eif.fit_forest(X, 200, 256, 0, 0), X)
        ours_if = isf.score(isf.fit_forest(X, 200, 256, 1), X)
        theirs = -IsolationForest(n_estimators=200, max_samples=256, random_state=0).fit(X).score_samples(X)
        assert spearmanr(ours0, ours_if)[0] > 0.9 and spearmanr(ours0, theirs)[0] > 0.9
        assert abs(float(ours0.mean()) - float(ours_if.mean())) < 0.01 and abs(ev.roc_auc(ours0, an) - ev.roc_auc(theirs, an)) < 0.02
        assert abs(float(np.median(ours0[~an])) - float(np.median(ours_if[~an]))) < 0.015


def test_score_convergence_and_flag_top_are_shared_with_the_isolation_forest():
    X, an = _data()
    F = eif.fit_forest(X, 200, 256, 0, 2)
    paths = eif.path_lengths(F, X)
    conv = eif.score_convergence(paths, F.psi, [1, 10, 100, 200])
    assert np.allclose(conv[-1], eif.score_from_paths(paths, F.psi))
    err = [np.abs(conv[k] - conv[-1]).mean() for k in range(3)]
    assert err[0] > err[1] > err[2]
    assert eif.flag_top(np.array([0.1, 0.9, 0.5]), 0.34).tolist() == [False, True, False]


def test_scale_dependence_the_isolation_forest_is_invariant_the_extended_one_is_not():
    X, an = _data(p=3)
    a = np.array([1e-3, 5.0, 300.0])
    scaled = X * a
    assert np.allclose(isf.score(isf.fit_forest(X, 60, 128, 5), X), isf.score(isf.fit_forest(scaled, 60, 128, 5), scaled), atol=1e-9)
    base = eif.score(eif.fit_forest(X, 60, 128, 5, 2), X)
    other = eif.score(eif.fit_forest(scaled, 60, 128, 5, 2), scaled)
    assert np.abs(base - other).max() > 0.02
    std = ev.standardise
    assert np.allclose(std(scaled), std(X))                                                       # nach dem Standardisieren identisch: deshalb rechnet die App darauf


def test_rotation_leaves_the_extended_forest_scores_similar_and_moves_the_axis_parallel_one():
    """Zwei getrennte Gaußwolken entlang der Diagonalen: der achsenparallele Wald verändert seine Scores mit der Drehung mehr als der schräge (in Rangkorrelation gegen die ungedrehten Daten)."""
    rng = np.random.default_rng(0)
    A = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    X = np.concatenate([rng.standard_normal((150, 2)) * [3.0, 0.3] @ A + [4, 4], rng.standard_normal((150, 2)) * [3.0, 0.3] @ A - [4, 4]])
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Xr = X @ R.T
    def corr(fit, score):
        return np.mean([spearmanr(score(fit(X, s), X), score(fit(Xr, s), Xr))[0] for s in range(4)])
    c_if = corr(lambda Z, s: isf.fit_forest(Z, 100, 256, s), isf.score)
    c_eif = corr(lambda Z, s: eif.fit_forest(Z, 100, 256, s, 1), eif.score)
    assert c_eif > c_if + 0.03


# --- Zerlegung der Ebene (Darstellung) -----------------------------------------------------------------------------------------------------


def test_tree_segments_lie_in_the_bounds_and_axis_parallel_trees_give_axis_parallel_lines():
    rng = np.random.default_rng(4)
    P = rng.standard_normal((40, 2))
    bounds = (P[:, 0].min() - 1, P[:, 0].max() + 1, P[:, 1].min() - 1, P[:, 1].max() + 1)
    axis = eif.build_tree(P, rng, isf.height_limit(40), 0)
    segs = eif.tree_segments(axis, bounds)
    assert len(segs) == int(axis.split.sum())
    for x0, y0, x1, y1, d in segs:
        assert (abs(x0 - x1) < 1e-9 or abs(y0 - y1) < 1e-9) and bounds[0] - 1e-9 <= min(x0, x1) and max(x0, x1) <= bounds[1] + 1e-9 and bounds[2] - 1e-9 <= min(y0, y1) and max(y0, y1) <= bounds[3] + 1e-9
    slanted = eif.build_tree(P, rng, isf.height_limit(40), 1)
    seg2 = eif.tree_segments(slanted, bounds)
    assert len(seg2) >= int(slanted.split.sum()) - 2 and any(abs(x0 - x1) > 1e-6 and abs(y0 - y1) > 1e-6 for x0, y0, x1, y1, d in seg2)      # es gibt schräge Linien


def test_normal_angle():
    assert eif.normal_angle(np.array([1.0, 0.0])) == pytest.approx(0.0) and eif.normal_angle(np.array([1.0, 1.0])) == pytest.approx(45.0)


# --- Kopierte Komponenten der Vorgänger -------------------------------------------------------------------------------------------------------


def test_isolation_forest_copy_matches_scikit_learn_and_c_factor_is_the_definition():
    X, an = _data(seed=1)
    ours = isf.score(isf.fit_forest(X, 100, 256, 0), X)
    theirs = -IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(X).score_samples(X)
    assert spearmanr(ours, theirs)[0] > 0.9
    n = np.array([3.0, 10.0, 256.0])
    assert np.allclose(isf.c_factor(n), 2 * (np.log(n - 1) + isf.EULER_GAMMA) - 2 * (n - 1) / n) and isf.c_factor(np.array([1, 2])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("dof", [1, 5, 12])
@pytest.mark.parametrize("q", [0.5, 0.975])
def test_chi2_and_mcd_copies(dof, q):
    assert alg.chi2_ppf(q, dof) == pytest.approx(chi2.ppf(q, dof), rel=1e-7)


def test_mcd_close_to_scikit_learn():
    X, is_out = _data(n=300, p=4, n_out=45, shift=8.0, seed=2)
    ours = alg.fit_mcd(X, 0.5, 0, reweight=True)
    sk = MinCovDet(random_state=0).fit(X)
    scale = np.sqrt(np.diag(np.cov(X[~is_out].T)))
    assert np.linalg.norm((ours.location - sk.location_) / scale) < 0.1 and (ours.d2[is_out] > alg.threshold(ours, 0.975)).all()


def test_a_plane_that_misses_all_points_uses_up_a_level_instead_of_ending_the_tree():
    """Regression: ein Schnitt, der alle Punkte auf eine Seite legt, machte früher den ganzen Knoten zum Blatt (bei ℓ >= 1 in etwa jedem achten Baum schon die Wurzel)."""
    X, _ = _data(n=64, p=3)
    tree = eif.build_tree(X, np.random.default_rng(0), 6, 99)
    assert tree.split[0] and 0 in (tree.size[tree.left[0]], tree.size[tree.right[0]])
    empty = np.flatnonzero((tree.size == 0) & ~tree.split)
    assert len(empty) >= 1 and (tree.normal[empty] == 0).all()
    assert eif.c_factor(0) == 0
    roots = [eif.build_tree(X, np.random.default_rng(s), 6, 2).split[0] for s in range(60)]
    assert all(roots)
