"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo der Extended Isolation Forest gegen den Isolation Forest oder die Wurzel verliert, steht das hier ebenso als Test wie dort, wo er gewinnt.
Alle Zahlen wurden NACH der Korrektur gemessen, dass eine Hyperebene, die alle Punkte auf eine Seite legt, eine Ebene verbraucht (früher machte sie den Knoten zum Blatt)."""

from functools import lru_cache

import numpy as np
import pytest

import eif_constants as C
import eif_evaluation as ev

TOL = 0.02
STANDARD = ev.Settings()


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Seitenleiste: Touren und Merkmale ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,if_f1,eif_f1,if_fa,eif_fa", [(20, 0.60, 1.00, 0.156, 0.000), (30, 0.67, 0.97, 0.119, 0.007), (50, 0.78, 0.94, 0.062, 0.004), (100, 0.89, 0.97, 0.027, 0.002), (200, 0.96, 0.96, 0.009, 0.004)])
def test_tours_sweep_only_the_isolation_forest_loses_at_small_samples(n, if_f1, eif_f1, if_fa, eif_fa):
    near(m("iforest", "f1", n=n), if_f1)
    near(m("eif", "f1", n=n), eif_f1)
    near(m("iforest", "false_alarm", n=n), if_fa, 0.01)
    near(m("eif", "false_alarm", n=n), eif_fa, 0.01)
    near(m("iforest", "auc", n=n), 1.00, 0.01)
    near(m("eif", "auc", n=n), 1.00, 0.01)


@pytest.mark.parametrize("p,if_f1,eif_f1", [(2, 0.92, 0.94), (5, 0.92, 0.96), (8, 0.94, 0.98), (12, 0.96, 0.98), (20, 0.96, 0.96), (30, 0.95, 0.97)])
def test_feature_count_sweep(p, if_f1, eif_f1):
    near(m("iforest", "auc", p=p), 1.00, 0.01)
    near(m("eif", "auc", p=p), 1.00, 0.01)
    near(m("iforest", "f1", p=p), if_f1)
    near(m("eif", "f1", p=p), eif_f1)


@pytest.mark.parametrize("nn,if_recall,eif_recall,if_f1,eif_f1,rob_auc", [(0, 0.99, 0.97, 0.96, 0.98, 1.00), (10, 0.91, 0.83, 0.94, 0.90, 0.99), (20, 0.71, 0.71, 0.82, 0.83, 0.96), (30, 0.58, 0.56, 0.72, 0.71, 0.91),
                                                                          (40, 0.39, 0.41, 0.56, 0.58, 0.88)])
def test_noise_features_do_not_help_the_slanted_cuts_and_keep_the_ranking(nn, if_recall, eif_recall, if_f1, eif_f1, rob_auc):
    near(m("iforest", "recall", n_noise=nn), if_recall)
    near(m("eif", "recall", n_noise=nn), eif_recall)
    near(m("iforest", "f1", n_noise=nn), if_f1)
    near(m("eif", "f1", n_noise=nn), eif_f1)
    near(m("robust", "auc", n_noise=nn), rob_auc, 0.015)
    for det in ("iforest", "eif"):
        near(m(det, "auc", n_noise=nn), 1.00 if nn < 30 else 0.99, 0.012)
    if nn == 40:
        near(oracle("iforest", n_noise=40), 0.84)                                                         # mit bekanntem Anteil trägt die Rangfolge
        near(oracle("eif", n_noise=40), 0.89)


@pytest.mark.parametrize("modes,if_f1,eif_f1,rob_f1,rob_auc", [(1, 0.96, 0.98, 0.84, 1.00), (2, 0.96, 0.95, 0.74, 0.95), (3, 0.97, 0.94, 0.36, 0.85)])
def test_modes_sweep(modes, if_f1, eif_f1, rob_f1, rob_auc):
    near(m("iforest", "f1", n_modes=modes), if_f1)
    near(m("eif", "f1", n_modes=modes), eif_f1)
    near(m("robust", "f1", n_modes=modes), rob_f1)
    near(m("robust", "auc", n_modes=modes), rob_auc)


@pytest.mark.parametrize("curv,if_f1,eif_f1,rob_f1", [(0.0, 0.96, 0.98, 0.84), (0.25, 0.98, 0.96, 0.49), (0.5, 0.97, 0.97, 0.41), (0.75, 0.97, 0.95, 0.39), (1.0, 0.95, 0.97, 0.38)])
def test_curvature_sweep_the_forests_do_not_care_only_the_chi2_threshold_suffers(curv, if_f1, eif_f1, rob_f1):
    near(m("iforest", "f1", curvature=curv), if_f1)
    near(m("eif", "f1", curvature=curv), eif_f1)
    near(m("robust", "f1", curvature=curv), rob_f1)


@pytest.mark.parametrize("noise,if_f1,eif_f1", [(0.0, 0.94, 0.98), (0.25, 0.96, 0.98), (0.5, 0.96, 0.97), (1.0, 0.96, 0.94)])
def test_noise_sweep(noise, if_f1, eif_f1):
    near(m("iforest", "f1", noise=noise), if_f1)
    near(m("eif", "f1", noise=noise), eif_f1)
    near(m("iforest", "auc", noise=noise), 1.00, 0.01)
    near(m("eif", "auc", noise=noise), 1.00, 0.01)


# --- Seitenleiste: Anomalien --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c,if_f1,eif_f1", [(2, 0.42, 0.69), (5, 0.81, 0.96), (10, 0.96, 0.98), (20, 0.97, 0.89), (30, 0.93, 0.72), (40, 0.87, 0.52), (45, 0.81, 0.42)])
def test_contamination_sweep_eif_wins_at_few_anomalies_and_loses_at_many(c, if_f1, eif_f1):
    near(m("iforest", "auc", contamination=c), 1.00, 0.01)
    near(m("eif", "auc", contamination=c), 1.00, 0.01)
    near(m("iforest", "f1", contamination=c), if_f1)
    near(m("eif", "f1", contamination=c), eif_f1)
    if c == 45:
        near(m("iforest", "recall", contamination=c), 0.68)
        near(m("eif", "recall", contamination=c), 0.27)


def test_dense_group_ranking_up_to_the_root_breaking_point():
    rows = ((10, 0.95, 0.97, 1.00), (15, 0.91, 0.91, 1.00), (20, 0.85, 0.85, 1.00), (25, 0.79, 0.74, 1.00), (30, 0.70, 0.57, 0.49), (40, 0.40, 0.35, 0.39), (45, 0.27, 0.20, 0.35))
    for c, if_auc, eif_auc, rob_auc in rows:
        near(m("iforest", "auc", kind="cluster", contamination=c, n_modes=1), if_auc)
        near(m("eif", "auc", kind="cluster", contamination=c, n_modes=1), eif_auc)
        near(m("robust", "auc", kind="cluster", contamination=c, n_modes=1), rob_auc)
    assert m("robust", "auc", kind="cluster", contamination=20) > m("eif", "auc", kind="cluster", contamination=20) + 0.1
    assert m("iforest", "auc", kind="cluster", contamination=30) > m("robust", "auc", kind="cluster", contamination=30) + 0.15
    near(m("robust", "f1", kind="cluster", contamination=20), 0.94)
    near(m("iforest", "f1", kind="cluster", contamination=20), 0.44)
    near(m("eif", "f1", kind="cluster", contamination=20), 0.13)
    near(m("iforest", "recall", kind="cluster", contamination=20), 0.44)
    near(m("eif", "recall", kind="cluster", contamination=20), 0.09)
    near(m("classical", "auc", kind="cluster", contamination=20), 0.62)


def test_gap_anomalies_the_extended_forest_is_worse_at_two_modes_and_no_detector_finds_them():
    near(m("iforest", "auc", n_modes=2, kind="gap"), 0.54)
    near(m("eif", "auc", n_modes=2, kind="gap"), 0.38)
    near(m("classical", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("robust", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("iforest", "auc", n_modes=2, kind="gap", contamination=2), 0.81, 0.03)                       # wenige, verstreute Lücken-Anomalien findet der Isolation Forest besser
    near(m("eif", "auc", n_modes=2, kind="gap", contamination=2), 0.56, 0.04)
    near(m("robust", "auc", n_modes=2, kind="gap", contamination=2), 0.35, 0.04)
    near(m("iforest", "auc", n_modes=3, kind="gap"), 0.27, 0.04)                                        # drei Betriebsarten: unter Raten
    near(m("eif", "auc", n_modes=3, kind="gap"), 0.31, 0.04)
    assert m("eif", "auc", n_modes=3, kind="gap") > m("iforest", "auc", n_modes=3, kind="gap")          # dort ist der Extended IF etwas besser
    near(m("iforest", "f1", n_modes=2, kind="gap"), 0.02, 0.03)


@pytest.mark.parametrize("strength,auc,if_f1,eif_f1", [(3.0, 0.96, 0.66, 0.72), (4.0, 0.99, 0.85, 0.88), (6.0, 1.00, 0.96, 0.98), (9.0, 1.00, 0.99, 1.00), (12.0, 1.00, 1.00, 1.00)])
def test_strength_sweep(strength, auc, if_f1, eif_f1):
    near(m("iforest", "auc", strength=strength), auc, 0.012)
    near(m("eif", "auc", strength=strength), auc, 0.012)
    near(m("iforest", "f1", strength=strength), if_f1)
    near(m("eif", "f1", strength=strength), eif_f1)


# --- Extended Isolation Forest: Erweiterungsgrad, Bäume, Unterstichprobe, Schwellen --------------------------------------------------------------


@pytest.mark.parametrize("level,f1,fa", [(0, 0.96, 0.008), (1, 0.97, 0.004), (2, 0.97, 0.003), (5, 0.98, 0.001), (11, 0.98, 0.001)])
def test_extension_sweep_the_ranking_is_the_same_but_the_normal_scores_fall_with_the_level(level, f1, fa):
    s = ev.Settings(extension=level)
    near(m("eif", "auc", s), 1.00, 0.01)
    near(m("eif", "f1", s), f1)
    near(m("eif", "false_alarm", s), fa, 0.008)
    near(m("iforest", "f1", s), 0.96)                                                                    # der Isolation Forest hängt nicht am Erweiterungsgrad


@pytest.mark.parametrize("trees,if_f1,eif_f1", [(10, 0.94, 0.95), (25, 0.95, 0.98), (50, 0.96, 0.97), (100, 0.96, 0.98), (200, 0.96, 0.98), (500, 0.96, 0.98)])
def test_tree_count_sweep(trees, if_f1, eif_f1):
    s = ev.Settings(n_trees=trees)
    near(m("iforest", "f1", s), if_f1)
    near(m("eif", "f1", s), eif_f1)
    near(m("iforest", "auc", s), 1.00, 0.01)
    near(m("eif", "auc", s), 1.00, 0.01)
    if trees == 10:
        near(m("eif", "recall", s), 0.91)


@pytest.mark.parametrize("psi,if_f1,eif_f1,if_fa,eif_fa", [(16, 0.56, 0.92, 0.18, 0.02), (32, 0.71, 0.96, 0.10, 0.006), (64, 0.83, 0.96, 0.05, 0.008), (128, 0.91, 0.98, 0.02, 0.003), (256, 0.96, 0.98, 0.009, 0.001)])
def test_psi_sweep_the_isolation_forest_score_depends_on_psi_much_more(psi, if_f1, eif_f1, if_fa, eif_fa):
    s = ev.Settings(psi=psi)
    near(m("iforest", "f1", s), if_f1)
    near(m("eif", "f1", s), eif_f1)
    near(m("iforest", "false_alarm", s), if_fa, 0.015)
    near(m("eif", "false_alarm", s), eif_fa, 0.012)
    near(m("iforest", "auc", s), 1.00, 0.012)
    near(m("eif", "auc", s), 1.00, 0.012)


@pytest.mark.parametrize("cut,if_f1,eif_f1,if_recall,eif_recall", [(0.45, 0.82, 0.92, 1.00, 0.99), (0.5, 0.96, 0.98, 0.99, 0.97), (0.55, 0.97, 0.85, 0.94, 0.75), (0.6, 0.83, 0.40, 0.71, 0.25), (0.65, 0.55, 0.04, 0.38, 0.02)])
def test_cutoff_sweep_the_extended_window_reaches_less_far_up(cut, if_f1, eif_f1, if_recall, eif_recall):
    s = ev.Settings(cutoff=cut)
    near(m("iforest", "f1", s), if_f1)
    near(m("eif", "f1", s), eif_f1)
    near(m("iforest", "recall", s), if_recall)
    near(m("eif", "recall", s), eif_recall)
    if cut == 0.45:
        near(m("iforest", "false_alarm", s), 0.050, 0.01)
        near(m("eif", "false_alarm", s), 0.018, 0.01)


@pytest.mark.parametrize("q,rob_f1,cla_f1", [(0.9, 0.67, 0.69), (0.95, 0.77, 0.66), (0.975, 0.84, 0.57), (0.99, 0.89, 0.44), (0.999, 0.90, 0.15)])
def test_chi2_quantile_sweep_for_the_root_detectors(q, rob_f1, cla_f1):
    s = ev.Settings(quantile=q)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "f1", s), cla_f1)
    near(m("iforest", "f1", s), m("iforest", "f1"), 1e-12)                                              # die Wälder hängen nicht am chi²-Quantil
    near(m("eif", "f1", s), m("eif", "f1"), 1e-12)


@pytest.mark.parametrize("share,if_f1,eif_f1,rob_f1,cla_f1", [(2, 0.33, 0.33, 0.33, 0.32), (5, 0.67, 0.67, 0.67, 0.56), (10, 0.97, 0.99, 0.90, 0.69), (20, 0.67, 0.67, 0.66, 0.58), (40, 0.40, 0.40, 0.40, 0.39)])
def test_assumed_share_sweep_a_wrong_share_costs_all_detectors_the_same(share, if_f1, eif_f1, rob_f1, cla_f1):
    s = ev.Settings(threshold_kind="share", share=share)
    near(m("iforest", "f1", s), if_f1)
    near(m("eif", "f1", s), eif_f1)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "f1", s), cla_f1)
    near(m("eif", "auc", s), 1.00, 0.01)


# --- Experimente: Tabellen ---------------------------------------------------------------------------------------------------------------


def test_normal_scores_of_the_isolation_forest_depend_on_the_tour_count_more():
    rows = {r["n"]: r for r in ev.threshold_table()["normal_scores"]}
    for n, if_expected, eif_expected in ((20, 0.435, 0.387), (50, 0.401, 0.382), (100, 0.382, 0.370), (300, 0.376, 0.367), (600, 0.377, 0.369)):
        near(rows[n]["iforest"], if_expected, 0.012)
        near(rows[n]["eif"], eif_expected, 0.012)
    assert rows[20]["iforest"] - rows[600]["iforest"] > 0.04 and rows[20]["eif"] - rows[600]["eif"] < 0.03
    assert all(r["eif"] < r["iforest"] for r in rows.values())                                           # der Extended IF gibt den normalen Touren niedrigere Scores


def test_threshold_table_shapes_and_wrong_share_factors():
    t = ev.threshold_table()
    assert [r["x"] for r in t["cutoff"]] == [0.45, 0.5, 0.55, 0.6, 0.65]
    assert [(r["factor"], r["x"]) for r in t["wrong_share"]] == [(0.5, 5), (1.0, 10), (2.0, 20)]
    f = {r["factor"]: r for r in t["wrong_share"]}
    near(f[1.0]["iforest_f1"], 0.97)
    near(f[1.0]["eif_f1"], 0.99)
    near(f[0.5]["iforest_f1"], 0.67)
    near(f[2.0]["eif_f1"], 0.67)
    assert f[1.0]["eif_f1"] > f[1.0]["robust_f1"] > f[1.0]["classical_f1"]


def test_rotation_has_no_measurable_effect_on_the_detection_and_the_scores_move_only_as_much_as_a_new_forest_seed():
    t = ev.rotation_table()
    assert [r["x"] for r in t["rows"]] == [0.0, 0.5, 1.0]
    base = t["rows"][0]
    for r in t["rows"]:
        for det in ("iforest", "eif"):
            near(r[f"{det}_auc"], 1.00, 0.01)
            near(r[f"{det}_f1"], base[f"{det}_f1"], 0.03)
        near(r["robust_auc"], base["robust_auc"], 0.005)                                                 # die Wurzel sieht die Drehung nicht
    near(t["floor"]["iforest"], 0.94, 0.02)
    near(t["floor"]["eif"], 0.92, 0.02)
    for r in t["rows"][1:]:
        near(r["iforest_corr"], t["floor"]["iforest"], 0.03)                                             # nicht mehr Verschiebung als durch einen anderen Wald-Seed
        near(r["eif_corr"], t["floor"]["eif"], 0.03)


def test_ghost_anisotropy_shrinks_with_the_extended_forest():
    rows = ev.ghost_table()
    by = {(r["n_modes"], r["distance"]): r for r in rows}
    near(by[(1, 1.0)]["iforest"], 0.081, 0.012)
    near(by[(1, 1.0)]["eif"], 0.029, 0.012)
    near(by[(2, 2.0)]["iforest"], 0.061, 0.012)
    near(by[(2, 2.0)]["eif"], 0.012, 0.012)
    near(by[(3, 2.0)]["iforest"], 0.061, 0.012)
    near(by[(3, 2.0)]["eif"], 0.001, 0.012)
    assert all(r["n_valid"] >= 4 and r["eif"] < r["iforest"] - 0.02 for r in rows)
    assert all(by[(k, 1.0)]["iforest"] > by[(k, 2.0)]["iforest"] for k in (1, 2, 3))                     # nahe am Datenrand stärker


def test_modes_table_the_documented_cells():
    rows = ev.modes_table()
    assert len(rows) == 8 and not any(r["n_modes"] == 1 and r["kind"] == "gap" for r in rows)
    by = {(r["n_modes"], r["kind"]): r for r in rows}
    for modes in (1, 2, 3):
        near(by[(modes, "scattered")]["iforest_auc"], 1.00, 0.01)
        near(by[(modes, "scattered")]["eif_auc"], 1.00, 0.01)
    near(by[(1, "cluster")]["iforest_auc"], 0.95)
    near(by[(1, "cluster")]["eif_auc"], 0.97)
    near(by[(1, "cluster")]["robust_auc"], 1.00, 0.01)
    near(by[(2, "cluster")]["robust_auc"], 1.00, 0.01)
    near(by[(3, "cluster")]["robust_auc"], 0.78)
    near(by[(3, "scattered")]["robust_auc"], 0.85)
    near(by[(2, "gap")]["iforest_auc"], 0.54)
    near(by[(2, "gap")]["eif_auc"], 0.38)
    near(by[(3, "gap")]["eif_auc"], 0.31, 0.04)
    assert max(by[(k, "gap")][f"{d}_auc"] for k in (2, 3) for d in ev.DETECTORS) < 0.55


def test_masking_table_and_psi_effect():
    mk = ev.masking_table()
    assert [r["x"] for r in mk["rows"]] == list(ev.MASKING_CONTAMINATION) and [r["x"] for r in mk["psi"]] == [16, 32, 64, 128, 256]
    psi = {r["x"]: r for r in mk["psi"]}
    near(psi[16]["iforest_auc"], 0.84)
    near(psi[256]["iforest_auc"], 0.70)
    near(psi[16]["eif_auc"], 0.85)
    near(psi[256]["eif_auc"], 0.57)
    assert psi[16]["iforest_auc"] > psi[256]["iforest_auc"] and psi[16]["eif_auc"] > psi[256]["eif_auc"]


def test_dimension_table_no_blind_spot_for_the_forests_and_the_extended_forest_does_not_lose_at_small_samples():
    cells = {(c["n"], c["p"]): c for c in ev.dimension_table()}
    assert min(c[f"{d}_auc"] for c in cells.values() for d in ("iforest", "eif")) >= 0.985
    near(cells[(20, 30)]["iforest_f1"], 0.61, 0.03)
    near(cells[(20, 30)]["eif_f1"], 1.00, 0.03)
    near(cells[(20, 12)]["eif_f1"], 1.00, 0.03)
    near(cells[(200, 12)]["iforest_f1"], 0.96, 0.03)
    near(cells[(200, 12)]["eif_f1"], 0.96, 0.03)
    assert all(cells[(n, p)]["eif_f1"] >= cells[(n, p)]["iforest_f1"] - 0.03 for n in (20, 30, 50, 100) for p in (2, 5, 12, 20, 30))
    assert cells[(400, 12)]["robust_auc"] > 0.98


def test_costs_the_extended_forest_is_slower_but_by_less_than_a_factor_of_two_and_needs_standardised_units():
    t = ev.cost_table()
    assert [r["p"] for r in t["times"]] == [2, 12, 30]
    for r in t["times"]:
        assert 1.0 < r["eif"] / r["iforest"] < 2.0, r                                                    # Zeitangaben sind rechnerabhängig: nur das Verhältnis wird behauptet
    raw = [r for r in t["units"] if not r["standardize"]][0]
    std = [r for r in t["units"] if r["standardize"]][0]
    near(std["eif_auc"], 1.00, 0.01)
    near(std["eif_f1"], 0.98)
    near(raw["eif_auc"], 0.93, 0.03)
    near(raw["eif_f1"], 0.65, 0.05)
    near(raw["eif_recall"], 0.73, 0.05)
    assert raw["iforest_auc"] == std["iforest_auc"]                                                    # der Isolation Forest ist skaleninvariant: derselbe Wald, dieselbe AUC


# --- Presets ----------------------------------------------------------------------------------------------------------------------------


def test_preset_help_numbers_standard_and_small_psi():
    near(m("iforest", "f1"), 0.96)
    near(m("eif", "f1"), 0.98)
    near(m("classical", "auc"), 0.95, 0.01)
    s16 = ev.Settings(psi=16)
    near(m("iforest", "auc", s16), 0.99, 0.01)
    near(m("iforest", "false_alarm", s16), 0.18, 0.02)
    near(m("iforest", "f1", s16), 0.56)
    near(m("eif", "false_alarm", s16), 0.02, 0.012)
    near(m("eif", "f1", s16), 0.92)


def test_preset_help_numbers_few_tours_many_features():
    p3 = dict(n=20, p=30)
    s3 = ev.Settings(psi=20)
    near(m("iforest", "auc", s3, **p3), 1.00, 0.01)
    near(m("eif", "auc", s3, **p3), 1.00, 0.01)
    for det in ("classical", "robust"):                                                                # n < p: singuläre Kovarianz, der Wert hängt an der Plattform-Rundung
        assert 0.3 <= m(det, "auc", s3, **p3) <= 0.75, det
    assert m("classical", "recall", s3, **p3) == 0.0 and m("robust", "recall", s3, **p3) == 0.0
    near(m("iforest", "false_alarm", s3, **p3), 0.144, 0.02)
    near(m("iforest", "f1", s3, **p3), 0.61, 0.03)
    near(m("eif", "false_alarm", s3, **p3), 0.0, 0.01)
    near(m("eif", "f1", s3, **p3), 1.00, 0.02)


def test_preset_help_numbers_the_rest():
    near(m("robust", "f1", kind="cluster", contamination=20), 0.94)
    near(m("classical", "f1"), 0.57)
    near(m("robust", "f1"), 0.84)


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["iforest"] < 4.0 and a.seconds["eif"] < 6.0 and a.seconds["robust"] < 10.0
