"""Auswertung der Extended-Isolation-Forest-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für den
Isolation Forest, den Extended Isolation Forest und die beiden Schätzer der Wurzel (klassisch, robust), Sweeps, Experimente auf Abruf, Score-Anisotropie und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import eif_algorithm as eif
import eif_isolation_forest as isf
import eif_ee_algorithm as alg
import eif_constants as C
import eif_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1: schräge Schnitte mischen die Merkmale, ihre Einheiten (Meter gegen Prozent) würden sonst die Richtung der Schnitte bestimmen."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def resolve_extension(extension, p_total):
    """Erweiterungsgrad ℓ aus einer Bezeichnung ("axis" = 0, "one" = 1, "two" = 2, "half", "full" = p - 1) oder einer Zahl, begrenzt auf 0 ... p - 1."""
    named = {"axis": 0, "one": 1, "two": 2, "half": (p_total - 1) // 2, "full": p_total - 1}
    level = named[extension] if isinstance(extension, str) else int(extension)
    return int(min(max(level, 0), p_total - 1))


@dataclass(frozen=True)
class Settings:
    extension: object = C.DEFAULT_EXTENSION             # Erweiterungsgrad des EIF: Bezeichnung oder Zahl (ℓ)
    n_trees: int = C.DEFAULT_TREES
    psi: int = C.DEFAULT_PSI
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": Score-Schwelle (beide Wälder) und chi²-Quantil (klassisch, robust); "share": der erwartete Anteil für alle vier
    cutoff: float = C.DEFAULT_CUTOFF                    # Score-Schwelle der beiden Wälder (nominell 0.5)
    quantile: float = C.DEFAULT_QUANTILE
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed der Wälder und der MCD-Starts (entkoppelt vom Seed der Aufnahme)
    standardize: bool = True                            # der EIF arbeitet auf standardisierten Kennzahlen (der IF ist skaleninvariant, der EIF nicht)


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength", "rotation")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, rotation=C.DEFAULT_ROTATION)
DETECTORS = ("iforest", "eif", "classical", "robust")
DETECTOR_NAMES = {"iforest": "Isolation Forest", "eif": "Extended Isolation Forest", "classical": "klassisch", "robust": "robust (MCD)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, rotation=C.DEFAULT_ROTATION, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise, rotation)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, rotation, seed)
    level: int                    # verwendeter Erweiterungsgrad ℓ
    forest_if: isf.Forest
    forest_eif: eif.Forest
    paths_if: np.ndarray          # (Bäume, n) Pfadlängen
    paths_eif: np.ndarray
    values: dict                  # Detektor -> Anomalie-Wert je Tour (Wälder: Score, klassisch/robust: quadrierter Mahalanobis-Abstand)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor (iforest, eif) -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den Wald-Reglern: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, classical, robust):
    """Markierung und Schwellenwert je Detektor: Standard = Score-Schwelle bzw. chi²-Quantil, sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["iforest"] = thr["eif"] = settings.cutoff
        thr["classical"] = alg.threshold(classical, settings.quantile)
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    p_total = X.shape[1]
    level = resolve_extension(settings.extension, p_total)
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, settings.n_trees, settings.psi, settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    Xe = standardise(X) if settings.standardize else X
    t0 = time.perf_counter()
    forest_eif = eif.fit_forest(Xe, settings.n_trees, settings.psi, settings.start, level)
    paths_eif = eif.path_lengths(forest_eif, Xe)
    secs["eif"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    values = {"iforest": isf.score_from_paths(paths_if, forest_if.psi), "eif": isf.score_from_paths(paths_eif, forest_eif.psi), "classical": classical.d2, "robust": robust.d2}
    flags, thr = _thresholds(values, settings, classical, robust)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in ("iforest", "eif")}
    return Analysis(ds, settings, params, level, forest_if, forest_eif, paths_if, paths_eif, values, classical, robust, scores, flags, oracle, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, rotation, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


# --- Score-Anisotropie und Score-Karte (Geister-Regionen) ---------------------------------------------------------------------------------


def anisotropy(X, score_fn, distance=1.0, grid=60):
    """Ein Wald über zwei Merkmale bewertet ein Raster um die Daten (`score_fn`: Punkte in Einheiten der Merkmale -> Score). Verglichen werden Rasterpunkte im GLEICHEN Abstand (standardisiert) zum nächsten Datenpunkt:
    solche außerhalb des Wertebereichs beider Merkmale ("Ecken") gegen solche, die in mindestens einem Bereich liegen ("Korridore"). Ein isotroper Detektor gäbe beiden dieselbe Bewertung; Rückgabe: mittlerer Score
    der Ecken minus der Korridore (NaN, wenn eine Gruppe zu klein ist) - positiv heißt: Punkte neben dem Datenbereich entlang der Achsen gelten als normaler (Geister-Regionen)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    pad = 2.0 * distance
    xs = np.linspace(lo[0] - pad, hi[0] + pad, grid)
    ys = np.linspace(lo[1] - pad, hi[1] + pad, grid)
    G = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    dist = np.sqrt(((G[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    s = score_fn(G * sd + mu)
    band = np.abs(dist - distance) < 0.15
    corridor = ((G[:, 0] >= lo[0]) & (G[:, 0] <= hi[0])) | ((G[:, 1] >= lo[1]) & (G[:, 1] <= hi[1]))
    corner, side = s[band & ~corridor], s[band & corridor]
    if len(corner) < 4 or len(side) < 4:
        return float("nan")
    return float(corner.mean() - side.mean())


def score_maps(X2, settings=Settings(), grid=80, pad=0.5):
    """Score-Karten beider Wälder über zwei Merkmale (Wälder nur auf diesen zwei Merkmalen, der EIF mit dem vollen Erweiterungsgrad ℓ = 1): (x-Achse, y-Achse, Scores IF [y, x], Scores EIF [y, x])."""
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    G = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    f_if = isf.fit_forest(X2, settings.n_trees, settings.psi, settings.start)
    mu, sd = X2.mean(axis=0), X2.std(axis=0)
    f_eif = eif.fit_forest((X2 - mu) / sd, settings.n_trees, settings.psi, settings.start, 1)
    return xs, ys, isf.score(f_if, G).reshape(grid, grid), eif.score(f_eif, (G - mu) / sd).reshape(grid, grid)


def ghost_probe(base_params, settings=Settings(), seed=C.DEFAULT_SEED, distance=1.0):
    """Die zwei ersten Merkmale (Distanz, Zeitfenster-Enge) ohne Anomalien: Anisotropie beider Wälder und die Daten für die Score-Karten. Rückgabe (X2, Anisotropie IF, Anisotropie EIF)."""
    ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base_params, "n_noise": 0, "p": 2, "rotation": 0.0})
    X2 = ds.X[~ds.anomaly]
    mu, sd = X2.mean(axis=0), X2.std(axis=0)
    f_if = isf.fit_forest(X2, settings.n_trees, settings.psi, settings.start)
    f_eif = eif.fit_forest((X2 - mu) / sd, settings.n_trees, settings.psi, settings.start, 1)
    return X2, anisotropy(X2, lambda G: isf.score(f_if, G), distance), anisotropy(X2, lambda G: eif.score(f_eif, (G - mu) / sd), distance)


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "extension": (0, 1, 2, 5, 11),
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "n_trees": (10, 25, 50, 100, 200, 500),
    "psi": (16, 32, 64, 128, 256),
    "cutoff": (0.45, 0.5, 0.55, 0.6, 0.65),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"extension": "Erweiterungsgrad ℓ (0 = achsenparallel)", "n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten",
                "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen", "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "n_trees": "Anzahl Bäume",
                "psi": "Größe der Unterstichprobe ψ", "cutoff": "Score-Schwelle der Wälder", "quantile": "chi²-Quantil der Schwelle (klassisch, robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("extension", "n_trees", "psi", "cutoff", "quantile", "share")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    out["iforest_oracle_f1"], out["eif_oracle_f1"] = a.oracle_f1["iforest"], a.oracle_f1["eif"]
    out["iforest_seconds"], out["eif_seconds"] = a.seconds["iforest"], a.seconds["eif"]
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der vier Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


def rotation_table(settings=Settings(), **base):
    """Drehung der Merkmale (0, ½, 1): AUC und F1 beider Wälder, dazu die Rangkorrelation der Scores der normalen Touren gegen die ungedrehten (und als Boden die zweier Wälder mit verschiedenem Seed auf denselben Daten)."""
    rows = []
    for r in (0.0, 0.5, 1.0):
        row = _mean_over_seeds(settings, **{**base, "rotation": r})
        corr = {"iforest": [], "eif": []}
        for seed in C.SWEEP_SEEDS:
            a0 = _analyse_seed(seed, settings, {**base, "rotation": 0.0})
            a1 = _analyse_seed(seed, settings, {**base, "rotation": r})
            normal = ~a0.ds.anomaly
            for d in corr:
                corr[d].append(_spearman(a0.values[d][normal], a1.values[d][normal]))
        row["x"] = r
        row["iforest_corr"], row["eif_corr"] = float(np.mean(corr["iforest"])), float(np.mean(corr["eif"]))
        rows.append(row)
    floor = {"iforest": [], "eif": []}
    for seed in C.SWEEP_SEEDS:
        a0 = _analyse_seed(seed, settings, base)
        a1 = _analyse_seed(seed, Settings(**{**settings.__dict__, "start": settings.start + 1}), base)
        normal = ~a0.ds.anomaly
        for d in floor:
            floor[d].append(_spearman(a0.values[d][normal], a1.values[d][normal]))
    return {"rows": rows, "floor": {d: float(np.mean(v)) for d, v in floor.items()}}


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def modes_table(settings=Settings(), **base):
    """Betriebsarten × Art der Anomalien: AUC der vier Detektoren (die Art 'Lücke' gibt es erst ab zwei Betriebsarten)."""
    rows = []
    for n_modes in (1, 2, 3):
        for kind in C.KINDS:
            if kind == "gap" and n_modes == 1:
                continue
            row = _mean_over_seeds(settings, **{**base, "n_modes": n_modes, "kind": kind})
            row.update(n_modes=n_modes, kind=kind)
            rows.append(row)
    return rows


MASKING_CONTAMINATION = (2, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def masking_table(settings=Settings(), **base):
    """Dichte Gruppe abseits: AUC der vier Detektoren über den Anteil, dazu der Einfluss der Unterstichprobe ψ bei 30 %."""
    rows = []
    for c in MASKING_CONTAMINATION:
        row = _mean_over_seeds(settings, **{**base, "kind": "cluster", "contamination": c, "n_modes": 1})
        row["x"] = c
        rows.append(row)
    psi_rows = []
    for psi in (16, 32, 64, 128, 256):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "psi": psi}), **{**base, "kind": "cluster", "contamination": 30, "n_modes": 1})
        row["x"] = psi
        psi_rows.append(row)
    return {"rows": rows, "psi": psi_rows}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen beider Wälder über die Score-Schwelle; (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil; (3) der mittlere Score der normalen Touren je Tourenzahl für beide Wälder."""
    cut = sweep("cutoff", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    normal_scores = []
    for n in (20, 50, 100, 300, 600):
        meds = {"iforest": [], "eif": []}
        for seed in C.SWEEP_SEEDS:
            a = _analyse_seed(seed, settings, {**base, "n": n})
            for d in meds:
                meds[d].append(np.median(a.values[d][~a.ds.anomaly]))
        normal_scores.append({"n": n, "iforest": float(np.mean(meds["iforest"])), "eif": float(np.mean(meds["eif"]))})
    return {"cutoff": cut, "wrong_share": wrong, "normal_scores": normal_scores}


DIMENSION_N = (20, 30, 50, 100, 200, 400)
DIMENSION_P = (2, 5, 12, 20, 30)


def dimension_table(settings=Settings(), **base):
    """Hohe Dimension: F1 beider Wälder bei der Schwelle über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten); ψ folgt der Tourenzahl."""
    cells = []
    for n in DIMENSION_N:
        for p in DIMENSION_P:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, "psi": min(settings.psi, n)}), **{**base, "n": n, "p": p})
            row.update(n=n, p=p)
            cells.append(row)
    return cells


def ghost_table(settings=Settings(), **base):
    """Score-Anisotropie beider Wälder über Betriebsarten (1-3) und Abstand zum Datenrand (1 und 2 Streuungen): Mittel über die Sweep-Datensätze."""
    rows = []
    for n_modes in (1, 2, 3):
        for distance in (1.0, 2.0):
            vals = [ghost_probe({**base, "n_modes": n_modes}, settings, seed, distance)[1:] for seed in C.SWEEP_SEEDS]
            a_if = np.array([v[0] for v in vals])
            a_eif = np.array([v[1] for v in vals])
            rows.append({"n_modes": n_modes, "distance": distance, "iforest": float(np.nanmean(a_if)), "eif": float(np.nanmean(a_eif)), "n_valid": int(np.sum(~np.isnan(a_if) & ~np.isnan(a_eif)))})
    return rows


def cost_table(settings=Settings(), **base):
    """Rechenzeit (Wald bauen und alle Touren bewerten) beider Wälder über die Merkmalszahl, und der EIF auf Rohdaten gegen standardisierte Kennzahlen (AUC und F1 bei der Schwelle)."""
    times = []
    for p in (2, 12, 30):
        rows = [_analyse_seed(seed, settings, {**base, "p": p}) for seed in C.SWEEP_SEEDS[:3]]
        times.append({"p": p, "iforest": float(np.mean([a.seconds["iforest"] for a in rows])), "eif": float(np.mean([a.seconds["eif"] for a in rows]))})
    units = []
    for standardize in (True, False):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "standardize": standardize}), **base)
        row["standardize"] = standardize
        units.append(row)
    return {"times": times, "units": units}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem die Wurzel als besser gilt
FOREST_AUC_MARGIN = 0.03      # AUC-Abstand zwischen den Wäldern
FOREST_F1_MARGIN = 0.10       # F1-Abstand bei der Schwelle zwischen den Wäldern
GAP_AUC = 0.8
THRESHOLD_F1_DROP = 0.15


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke, Wurzel besser (dichte Gruppe), EIF besser als IF, IF besser als EIF, falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    i, e, c, r = (a.scores[d] for d in DETECTORS)
    best_iso_auc = max(i["auc"], e["auc"])
    data = {"n": ds.n, "p": ds.p + ds.n_noise, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()), "level": a.level,
            "oracle_if": a.oracle_f1["iforest"], "oracle_eif": a.oracle_f1["eif"], **{f"{d}_{k}": v for d in DETECTORS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and best_iso_auc < GAP_AUC:
        return "warning", "gap", data
    if max(c["auc"], r["auc"]) - best_iso_auc >= WIN_MARGIN:
        return "warning", "dense_masking" if ds.kind == "cluster" else "root_wins", data
    d_auc, d_f1 = e["auc"] - i["auc"], e["f1"] - i["f1"]
    if d_auc >= FOREST_AUC_MARGIN or d_f1 >= FOREST_F1_MARGIN:
        return "success", "eif_wins", data
    if -d_auc >= FOREST_AUC_MARGIN or -d_f1 >= FOREST_F1_MARGIN:
        return "warning", "if_wins", data
    if best_iso_auc >= 0.95 and max(a.oracle_f1.values()) - max(i["f1"], e["f1"]) > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
