"""Szenario (identisch zu den Vorgänger-Demos, neu: Drehung), Kennzahlen mit Handinstanzen, Analyse und Schwellen für vier Detektoren, Score-Anisotropie, Urteil."""

import numpy as np
import pytest

import eif_algorithm as eif
import eif_constants as C
import eif_ee_algorithm as alg
import eif_evaluation as ev
import eif_isolation_forest as isf
import eif_scenario as sc

# Zeilensummen der ersten acht Zeilen der PCA-Demo (dieselben normalen Zeilen wie in der Elliptic-Envelope- und der Isolation-Forest-Demo): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_and_the_predecessor_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)


def test_default_dataset_is_frozen_at_the_predecessor_values():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


def test_noise_features_are_appended_and_change_nothing_else():
    base = ev.make_dataset()
    for n_noise in (1, 10, 40):
        ds = ev.make_dataset(n_noise=n_noise)
        assert ds.X.shape == (300, 12 + n_noise) and np.array_equal(ds.X[:, :12], base.X) and np.array_equal(ds.anomaly, base.anomaly)
        assert len(ds.names) == 12 + n_noise and ds.names[-1] == f"Rauschmerkmal {n_noise}" and ds.n_noise == n_noise
    ds = ev.make_dataset(n_noise=40)
    extra = ds.X[:, 12:]
    assert abs(extra.mean() - C.EXTRA_MEAN) < 1.0 and abs(extra.std() - C.EXTRA_SCALE) < 1.0
    off = np.abs(np.corrcoef(extra.T))
    np.fill_diagonal(off, 0.0)
    assert off.max() < 0.3 and abs(np.corrcoef(np.c_[extra[:, 0], ds.X[:, 0]].T)[0, 1]) < 0.3
    assert np.array_equal(ev.make_dataset(p=30, n_noise=5).X[:, :30], ev.make_dataset(p=30).X)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_gap_needs_two_modes_and_kinds_have_their_geometry():
    assert ev.make_dataset(kind="gap").kind == "scattered"
    gap = ev.make_dataset(n_modes=2, kind="gap")
    assert np.abs(gap.z[gap.anomaly]).max() < 1.2
    clu = ev.make_dataset(kind="cluster", contamination=20)
    zc = clu.z[clu.anomaly]
    assert np.linalg.norm(zc.mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3, n_noise=4, rotation=0.5).X, ev.make_dataset(seed=3, n_noise=4, rotation=0.5).X)


# --- Drehung -------------------------------------------------------------------------------------------------------------------------


def test_rotation_zero_is_bit_identical_to_the_predecessor_scenario():
    assert np.array_equal(ev.make_dataset(rotation=0.0).X, ev.make_dataset().X) and ev.make_dataset().rotation == 0.0
    assert not any("gedreht" in name for name in ev.make_dataset().names) and all(name.endswith("(gedreht)") for name in ev.make_dataset(rotation=0.5).names[:12])


def test_rotation_is_orthogonal_in_the_standardised_units_of_the_unrotated_data():
    X0 = ev.make_dataset(p=6).X
    mu, sd = X0.mean(axis=0), X0.std(axis=0)
    Z0 = (X0 - mu) / sd
    theta_full = np.radians(C.ROTATION_MAX_DEG)
    for r in (0.25, 0.5, 1.0):
        Z1 = (sc.rotate_features(X0, r) - mu) / sd
        assert np.allclose(Z0[:, 0::2] ** 2 + Z0[:, 1::2] ** 2, Z1[:, 0::2] ** 2 + Z1[:, 1::2] ** 2, atol=1e-9)             # je Paar bleibt der Abstand zum Mittelpunkt
        assert np.allclose(Z1.mean(axis=0), 0.0, atol=1e-9)
        th = r * theta_full
        assert np.allclose(Z1[:, 0], np.cos(th) * Z0[:, 0] - np.sin(th) * Z0[:, 1], atol=1e-9) and np.allclose(Z1[:, 1], np.sin(th) * Z0[:, 0] + np.cos(th) * Z0[:, 1], atol=1e-9)
    assert np.allclose(sc.rotate_features(X0, 0.0), X0)
    odd = sc.rotate_features(ev.make_dataset(p=5).X, 1.0)
    assert np.allclose(odd[:, 4], ev.make_dataset(p=5).X[:, 4], rtol=1e-12)                                                   # ein ungerades letztes Merkmal bleibt


def test_rotation_leaves_noise_columns_anomalies_and_the_mahalanobis_distances_alone():
    a, b = ev.make_dataset(n_noise=6), ev.make_dataset(n_noise=6, rotation=1.0)
    assert np.array_equal(a.X[:, 12:], b.X[:, 12:]) and np.array_equal(a.anomaly, b.anomaly) and not np.allclose(a.X[:, :12], b.X[:, :12])
    da, db = alg.fit_classical(ev.make_dataset().X).d2, alg.fit_classical(ev.make_dataset(rotation=1.0).X).d2
    assert np.allclose(da, db, rtol=1e-6)                                                                  # die Wurzel sieht die Drehung nicht (affin äquivariant)


# --- Kennzahlen: Handinstanzen (aus der Wurzel übernommen) ----------------------------------------------------------------------------


def test_roc_auc_and_average_precision_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    assert ev.roc_auc(s, y) == pytest.approx(np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_curve_and_flag_metrics():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.4)


def test_resolve_extension_names_numbers_and_caps():
    assert [ev.resolve_extension(k, 12) for k in ("axis", "one", "two", "half", "full")] == [0, 1, 2, 5, 11]
    assert [ev.resolve_extension(k, 2) for k in ("axis", "one", "two", "half", "full")] == [0, 1, 1, 0, 1]
    assert ev.resolve_extension(99, 12) == 11 and ev.resolve_extension(-3, 12) == 0 and ev.resolve_extension(4, 12) == 4


# --- Analyse und Schwellen ----------------------------------------------------------------------------------------------------------------


def _params(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    return (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], p["rotation"], 7)


def alg_chi(dof):
    return alg.chi2_ppf(C.DEFAULT_QUANTILE, dof)


def test_analysis_fields_and_consistency():
    a = ev.analyse_for(_params())
    assert set(a.scores) == set(ev.DETECTORS) == set(a.flags) == set(a.values) == {"iforest", "eif", "classical", "robust"}
    assert a.paths_if.shape == a.paths_eif.shape == (100, 300) and a.forest_if.psi == a.forest_eif.psi == 256 and a.robust.h == 156 and a.classical.dof == 12 and a.level == 11
    assert np.allclose(a.values["iforest"], isf.score_from_paths(a.paths_if, a.forest_if.psi)) and np.allclose(a.values["eif"], isf.score_from_paths(a.paths_eif, a.forest_eif.psi))
    for d in ev.DETECTORS:
        assert a.scores[d]["n_flagged"] == int(a.flags[d].sum())
    for d in ("iforest", "eif"):
        assert a.scores[d]["threshold"] == 0.5 and (a.flags[d] == (a.values[d] > 0.5)).all()
    assert a.scores["classical"]["threshold"] == pytest.approx(alg_chi(12)) and (a.flags["robust"] == (a.values["robust"] > alg_chi(12))).all()
    assert set(a.oracle_f1) == {"iforest", "eif"} and all(0 <= v <= 1 for v in a.oracle_f1.values()) and a.seconds["iforest"] > 0 and a.seconds["eif"] > 0


def test_share_threshold_flags_exactly_the_assumed_share_for_all_four_detectors():
    for share in (2, 10, 40):
        a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=share))
        for d in ev.DETECTORS:
            assert int(a.flags[d].sum()) == max(1, round(share / 100 * 300))
            assert a.flags[d][np.argsort(-a.values[d])[:3]].all()
    a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=10))
    assert a.scores["iforest"]["threshold"] > 0.3 and a.scores["eif"]["threshold"] > 0.3
    assert a.scores["iforest"]["f1"] == pytest.approx(a.oracle_f1["iforest"]) and a.scores["eif"]["f1"] == pytest.approx(a.oracle_f1["eif"])              # 10 % = wahrer Anteil


def test_score_cutoff_and_quantile_move_only_their_own_detectors():
    base = ev.analyse_for(_params())
    cut = ev.analyse_for(_params(), ev.Settings(cutoff=0.6))
    assert cut.scores["iforest"]["recall"] < base.scores["iforest"]["recall"] and cut.scores["eif"]["recall"] < base.scores["eif"]["recall"]
    assert cut.scores["classical"] == base.scores["classical"] and cut.scores["robust"] == base.scores["robust"]
    q = ev.analyse_for(_params(), ev.Settings(quantile=0.999))
    assert q.scores["iforest"] == base.scores["iforest"] and q.scores["eif"] == base.scores["eif"] and q.scores["robust"]["false_alarm"] <= base.scores["robust"]["false_alarm"]
    assert q.scores["classical"]["auc"] == base.scores["classical"]["auc"]                       # nur die Schwelle, nicht die Rangfolge


def test_forest_seed_is_separate_from_the_data_seed():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(), ev.Settings(start=5))
    assert np.array_equal(a.ds.X, b.ds.X)
    for d in ("iforest", "eif"):
        assert a.values[d].tolist() != b.values[d].tolist() and abs(a.scores[d]["auc"] - b.scores[d]["auc"]) < 0.02


def test_extension_setting_changes_only_the_extended_forest():
    a = ev.analyse_for(_params(), ev.Settings(extension="axis"))
    b = ev.analyse_for(_params(), ev.Settings(extension="full"))
    assert a.level == 0 and b.level == 11 and a.forest_eif.extension == 0 and b.forest_eif.extension == 11
    assert np.array_equal(a.values["iforest"], b.values["iforest"]) and a.values["robust"].tolist() == b.values["robust"].tolist()
    assert not np.allclose(a.values["eif"], b.values["eif"])
    assert ev.analyse_for(_params(p=2), ev.Settings(extension=99)).level == 1                       # begrenzt auf p - 1


def test_standardisation_matters_only_for_the_extended_forest():
    a = ev.analyse_for(_params())
    b = ev.analyse_for(_params(), ev.Settings(standardize=False))
    assert np.array_equal(a.values["iforest"], b.values["iforest"]) and a.values["robust"].tolist() == b.values["robust"].tolist() and not np.allclose(a.values["eif"], b.values["eif"])
    assert b.scores["eif"]["auc"] < a.scores["eif"]["auc"] - 0.02                                    # auf Rohdaten dominiert das Merkmal mit der größten Einheit


def test_psi_larger_than_n_is_capped_and_noise_features_are_used():
    a = ev.analyse_for(_params(n=20, p=30), ev.Settings(psi=512))
    assert a.forest_if.psi == a.forest_eif.psi == 20
    b = ev.analyse_for(_params(n_noise=10))
    assert b.ds.X.shape[1] == 22 and b.classical.dof == 22 and b.level == 21


def test_rotation_changes_the_isolation_forest_scores_but_not_the_root_or_the_anomaly_set():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(rotation=1.0))
    assert np.array_equal(a.ds.anomaly, b.ds.anomaly) and not np.allclose(a.values["iforest"], b.values["iforest"])
    assert np.allclose(a.values["classical"], b.values["classical"], rtol=1e-6) and abs(a.scores["robust"]["auc"] - b.scores["robust"]["auc"]) < 0.01


def test_sweep_rows_and_labels():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for d in ev.DETECTORS:
            assert r[f"{d}_auc_min"] <= r[f"{d}_auc"] <= r[f"{d}_auc_max"] and r[f"{d}_auc_std"] >= 0
        assert "iforest_oracle_f1" in r and "eif_oracle_f1" in r and "eif_seconds" in r
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    s = ev.sweep("cutoff", values=(0.45, 0.65))
    assert s[0]["iforest_recall"] > s[1]["iforest_recall"] and s[0]["eif_recall"] > s[1]["eif_recall"] and s[0]["classical_auc"] == s[1]["classical_auc"] and s[0]["robust_recall"] == s[1]["robust_recall"]
    e = ev.sweep("extension", values=(0, 5))
    assert e[0]["iforest_auc"] == e[1]["iforest_auc"] and e[0]["robust_auc"] == e[1]["robust_auc"] and e[0]["eif_false_alarm"] > e[1]["eif_false_alarm"]


def test_settings_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.extension, s.n_trees, s.psi, s.threshold_kind, s.cutoff, s.quantile, s.share) == (C.DEFAULT_EXTENSION, C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_THRESHOLD_KIND, C.DEFAULT_CUTOFF, C.DEFAULT_QUANTILE, C.DEFAULT_SHARE)
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005)) and s.standardize is True


# --- Score-Anisotropie -------------------------------------------------------------------------------------------------------------------


def test_anisotropy_is_positive_for_a_blob_and_nan_when_a_group_is_too_small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 2))
    F = isf.fit_forest(X, 100, 256, 0)
    assert ev.anisotropy(X, lambda G: isf.score(F, G), 2.0) > 0.03
    assert np.isnan(ev.anisotropy(X, lambda G: isf.score(F, G), 1.0, grid=3))


def test_extended_forest_has_less_anisotropy_than_the_isolation_forest():
    for seed in C.SWEEP_SEEDS[:3]:
        X2, a_if, a_eif = ev.ghost_probe({}, seed=seed)
        assert 0.04 < a_if < 0.11 and a_eif < a_if - 0.02, (seed, a_if, a_eif)
    assert len(X2) == 270


def test_score_maps_shape_and_scores_rise_away_from_the_data():
    X2, _, _ = ev.ghost_probe({})
    xs, ys, S_if, S_eif = ev.score_maps(X2, grid=40)
    assert S_if.shape == S_eif.shape == (40, 40) and len(xs) == 40 and len(ys) == 40
    for S in (S_if, S_eif):
        center = S[np.argmin(np.abs(ys - X2[:, 1].mean())), np.argmin(np.abs(xs - X2[:, 0].mean()))]
        assert S[0, 0] > center + 0.1 and S[-1, -1] > center + 0.1


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _verdict(**kw):
    settings = ev.Settings(**{k: kw.pop(k) for k in ("threshold_kind", "share", "cutoff", "psi") if k in kw})
    return ev.verdict(ev.analyse_for(_params(**kw), settings))


def test_verdict_codes_for_the_presets_and_edge_cases():
    assert _verdict()[:2] == ("success", "comparable")
    assert _verdict(psi=16)[:2] == ("success", "eif_wins")
    assert _verdict(n=20, p=30, psi=20)[:2] == ("success", "eif_wins")
    assert _verdict(kind="cluster", contamination=20)[:2] == ("warning", "dense_masking")
    assert _verdict(n_modes=2, kind="gap")[:2] == ("warning", "gap")
    assert _verdict(n_modes=2, kind="gap", contamination=2)[1] in ("if_wins", "gap")
    assert _verdict(cutoff=0.6)[:2] == ("warning", "if_wins")
    assert _verdict(threshold_kind="share", share=5)[:2] == ("warning", "threshold_off")


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _verdict(kind="cluster", contamination=20)
    assert code == "dense_masking"
    for key in ("iforest_auc", "eif_auc", "classical_auc", "robust_auc", "iforest_recall", "eif_recall", "iforest_false_alarm", "eif_false_alarm", "iforest_f1", "eif_f1", "classical_f1", "robust_f1",
                "oracle_if", "oracle_eif", "contamination", "n_anomalies", "p", "level"):
        assert key in data
    assert data["p"] == 12 and data["n_noise"] == 0 and data["contamination"] == pytest.approx(20.0)
    assert set(("eif_wins", "if_wins", "gap", "dense_masking", "root_wins", "threshold_off", "comparable")) >= {_verdict()[1]}


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["iforest"] < 4.0 and a.seconds["eif"] < 6.0 and a.seconds["robust"] < 10.0
    b = ev.analyse(ev.make_dataset())
    assert b.seconds["eif"] < 1.5
