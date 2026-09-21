"""Plotly-Visualisierungen der Extended-Isolation-Forest-Demo: Touren in der Ebene der größten Streuung, ein achsenparalleler und ein schräger Baum als Zerlegung der Ebene, Pfadlängen, Scores und Schwelle,
ROC-Kurven, Kennzahlen-Balken, Sweeps, Score-Karten (Geister-Regionen), Drehung, Betriebsarten, Masking, Schwelle, Dimension und Kosten. Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import eif_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"iforest": PURPLE, "eif": PINK, "classical": ORANGE, "robust": TEAL}
DETECTOR_NAMES = {"iforest": "Isolation Forest", "eif": "Extended IF", "classical": "klassisch", "robust": "robust (MCD)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke"}
DETECTORS = ("iforest", "eif", "classical", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Projektion -----------------------------------------------------------------------------------------------------------------


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_tree_plane(P, anomaly, segments, picks):
    """Ein Isolationsbaum über die Ebene der zwei Hauptrichtungen: die Schnittgeraden (dunkler = früher), dazu markierte Punkte (offene Kreise). `picks` = [(Index, Name, Farbe)]."""
    fig = go.Figure(_points(P, anomaly))
    if segments:
        max_depth = max(s[4] for s in segments)
        for x0, y0, x1, y1, d in segments:
            shade = 0.25 + 0.75 * (1.0 - d / max(max_depth, 1))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=f"rgba(60,60,60,{shade:.2f})", width=1.6 if d < 3 else 1), hoverinfo="skip", showlegend=False))
    for idx, name, color in picks:
        fig.add_trace(go.Scatter(x=[P[idx, 0]], y=[P[idx, 1]], mode="markers", marker=dict(size=16, color=color, line=dict(width=3), symbol="circle-open"), name=name, hoverinfo="skip"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1", zeroline=False, range=[P[:, 0].min() - 0.5, P[:, 0].max() + 0.5]),
                      yaxis=dict(title="Hauptrichtung 2", zeroline=False, range=[P[:, 1].min() - 0.5, P[:, 1].max() + 0.5]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_path_lengths(paths_if, paths_eif, c_psi, title_if="Isolation Forest", title_eif="Extended IF"):
    """Pfadlänge einer Sonderfahrt über alle Bäume beider Wälder (Histogramm); senkrecht: c(ψ)."""
    top = max(float(paths_if.max()), float(paths_eif.max()), c_psi) + 1
    bins = dict(start=0.0, end=top, size=top / 30.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=paths_if, xbins=bins, marker_color=PURPLE, opacity=0.7, name=title_if, hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=paths_eif, xbins=bins, marker_color=PINK, opacity=0.7, name=title_eif, hoverinfo="skip"))
    fig.add_vline(x=float(c_psi), line=dict(color="black", dash="dash"), annotation_text="c(ψ)", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="Pfadlänge der Sonderfahrt in einem Baum"), yaxis=dict(title="Bäume"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_mean_paths(mean_if, mean_eif, anomaly, c_psi):
    """Mittlere Pfadlänge je Tour: Sonderfahrten (durchgezogen, dunkler) und normale Touren beider Wälder als Histogramm-Paare."""
    top = float(max(mean_if.max(), mean_eif.max(), c_psi)) + 1
    bins = dict(start=0.0, end=top, size=top / 40.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mean_if[~anomaly], xbins=bins, marker_color=PURPLE, opacity=0.55, name="IF normal", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=mean_eif[~anomaly], xbins=bins, marker_color=PINK, opacity=0.55, name="EIF normal", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=mean_if[anomaly], xbins=bins, marker_color=PURPLE, opacity=0.95, name="IF Sonderfahrten", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=mean_eif[anomaly], xbins=bins, marker_color=PINK, opacity=0.95, name="EIF Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(c_psi), line=dict(color="black", dash="dash"), annotation_text="c(ψ)", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="mittlere Pfadlänge E[h]"), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_convergence(steps, curves_if, curves_eif, anomaly_flags):
    """Anomalie-Wert einiger Touren nach den ersten k Bäumen: Isolation Forest (violett) und Extended IF (pink), Sonderfahrten kräftig, normale Touren blass."""
    fig = go.Figure()
    xs = [str(k) for k in steps]
    for curves, color, name in ((curves_if, PURPLE, "Isolation Forest"), (curves_eif, PINK, "Extended IF")):
        for j in range(curves.shape[1]):
            fig.add_trace(go.Scatter(x=xs, y=curves[:, j], mode="lines", line=dict(color=color, width=2 if anomaly_flags[j] else 1), opacity=0.9 if anomaly_flags[j] else 0.35, hoverinfo="skip",
                                     name=name, showlegend=j == 0))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Anzahl Bäume", type="category"), yaxis=dict(title="Anomalie-Wert", range=[0.2, 1.0]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_scores(values_if, values_eif, anomaly, thr, extra=None):
    """Histogramme des Anomalie-Werts beider Wälder (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle."""
    hi = float(max(values_if.max(), values_eif.max(), thr * 1.05))
    lo = float(min(values_if.min(), values_eif.min(), thr * 0.95))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 40.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values_if[~anomaly], xbins=bins, marker_color=PURPLE, opacity=0.5, name="IF normal", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=values_eif[~anomaly], xbins=bins, marker_color=PINK, opacity=0.5, name="EIF normal", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values_if[anomaly], xbins=bins, marker_color=PURPLE, opacity=0.95, name="IF Sonderfahrten", hoverinfo="skip"))
        fig.add_trace(go.Histogram(x=values_eif[anomaly], xbins=bins, marker_color=PINK, opacity=0.95, name="EIF Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="Anomalie-Wert (Score)"), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_score_maps(xs, ys, S_if, S_eif, X2, names, ellipse=None):
    """Score-Karten beider Wälder über zwei Merkmale nebeneinander (gleiche Farbskala; dunkler = normaler, heller = auffälliger) mit den Touren; die Bänder entlang der Achsen (links) sind die Geister-Regionen
    der achsenparallelen Schnitte. Optional die Ellipse der robusten Schätzung (gestrichelt)."""
    lo, hi = float(min(S_if.min(), S_eif.min())), float(max(S_if.max(), S_eif.max()))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Isolation Forest (achsenparallel)", "Extended IF (schräg)"), horizontal_spacing=0.06, shared_yaxes=True)
    for col, S in ((1, S_if), (2, S_eif)):
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="Viridis", reversescale=True, zmin=lo, zmax=hi, showscale=col == 2, colorbar=dict(title="Score", thickness=12), hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers", marker=dict(size=3, color="white", line=dict(color="black", width=0.4)), hoverinfo="skip", showlegend=False), row=1, col=col)
        if ellipse is not None:
            fig.add_trace(go.Scatter(x=ellipse[:, 0], y=ellipse[:, 1], mode="lines", line=dict(color=TEAL, width=2.5, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title=names[0])
    fig.update_yaxes(title=names[1], col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_anisotropy(rows):
    """Score-Anisotropie (Ecken minus Korridore bei gleichem Abstand zum Datenrand) je Betriebsart und Abstand: Isolation Forest gegen Extended IF; positiv = Geister-Regionen entlang der Achsen."""
    labels = [f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}<br>Abstand {r['distance']:g} σ" for r in rows if r["n_valid"] >= 3]
    sel = [r for r in rows if r["n_valid"] >= 3]
    fig = go.Figure()
    for key, color, name in (("iforest", PURPLE, "Isolation Forest"), ("eif", PINK, "Extended IF")):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in sel], name=name, marker_color=color, text=[f"{r[key]:.3f}" for r in sel], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Ecken − Korridore", range=[0, 0.11]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_rotation(table):
    """Drehung der Merkmale: links AUC (durchgezogen) und F1 bei der Schwelle (gestrichelt) beider Wälder, rechts die Rangkorrelation der Scores der normalen Touren mit den ungedrehten (waagerechte Linien: der Boden
    zweier Wälder mit verschiedenem Seed auf denselben Daten)."""
    rows = table["rows"]
    xs = [r["x"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC und F1 über die Drehung", "Rangkorrelation der Scores (normale Touren)"), horizontal_spacing=0.12)
    for det in ("iforest", "eif"):
        color = DETECTOR_COLORS[det]
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=f"AUC {DETECTOR_NAMES[det]}", line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", name=f"F1 {DETECTOR_NAMES[det]}", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_corr"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_hline(y=table["floor"][det], line=dict(color=color, dash="dot"), row=1, col=2)
    fig.update_xaxes(title="Drehung (1 = 45° je Merkmalspaar)")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_modes(rows):
    """AUC der vier Detektoren je Anzahl Betriebsarten und Art der Anomalien (die Linie bei 0.5 ist Raten)."""
    labels = [f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}<br>{KIND_NAMES[r['kind']]}" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_auc"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=420, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="AUC", range=[0, 1.15]), legend=dict(orientation="h", y=-0.55))
    return lock_axes(fig)


def build_masking(table):
    """Dichte Gruppe abseits: AUC der vier Detektoren über den Anteil (links) und beide Wälder bei 30 % über die Unterstichprobe ψ (rechts)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC über den Anteil der Gruppe", "Bei 30 %: Unterstichprobe ψ"), horizontal_spacing=0.12)
    xs = [r["x"] for r in table["rows"]]
    for det in DETECTORS:
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in table["rows"]], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=DETECTOR_COLORS[det], width=3), hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    ps = table["psi"]
    for det in ("iforest", "eif"):
        fig.add_trace(go.Scatter(x=[r["x"] for r in ps], y=[r[f"{det}_auc"] for r in ps], mode="lines+markers", line=dict(color=DETECTOR_COLORS[det], width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="Anteil der Gruppe [%]", row=1, col=1)
    fig.update_xaxes(title="ψ", row=1, col=2)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_cutoff(rows):
    """F1 (durchgezogen) und Fehlalarmrate (gestrichelt) beider Wälder über die Score-Schwelle."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    for det in ("iforest", "eif"):
        color = DETECTOR_COLORS[det]
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", name=f"F1 {DETECTOR_NAMES[det]}", line=dict(color=color, width=3), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", name=f"Fehlalarm {DETECTOR_NAMES[det]}", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Score-Schwelle", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_dimension(cells):
    """F1 bei der Schwelle des Isolation Forest (links) und des Extended IF (rechts) über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten)."""
    ns = sorted({c["n"] for c in cells})
    ps = sorted({c["p"] for c in cells})
    grid = {(c["n"], c["p"]): c for c in cells}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("F1 Isolation Forest", "F1 Extended IF"), horizontal_spacing=0.14)
    for col, key, scale in ((1, "iforest_f1", "Purples"), (2, "eif_f1", "Reds")):
        z = [[grid[(n, p)][key] for p in ps] for n in ns]
        fig.add_trace(go.Heatmap(z=z, x=[f"p = {p}" for p in ps], y=[f"n = {n}" for n in ns], colorscale=scale, zmin=0.3, zmax=1, text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", showscale=False,
                                 hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Wald bauen und alle Touren bewerten) beider Wälder über die Merkmalszahl."""
    fig = go.Figure()
    for key, color, name in (("iforest", PURPLE, "Isolation Forest"), ("eif", PINK, "Extended IF")):
        y = [t[key] for t in times]
        fig.add_trace(go.Bar(x=[f"p = {t['p']}" for t in times], y=y, name=name, marker_color=color, text=[f"{v:.2f} s" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=280, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Sekunden"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
