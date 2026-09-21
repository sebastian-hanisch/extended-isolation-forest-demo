"""Extended Isolation Forest - schräge Schnitte statt Geister-Regionen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Extended Isolation Forest - und lässt stattdessen das Beispiel wachsen.
Drittes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: die Fortsetzung des Isolation-Forest-Astes (Wurzel: Elliptic Envelope), die die achsenparallelen Schnitte des Vorgängers durch zufällige
Hyperebenen ersetzt. Gemessen wird, was das bringt - und was nicht. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import eif_algorithm as eif
import eif_constants as C
import eif_ee_algorithm as alg
import eif_isolation_forest as isf
from eif_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    Settings,
    analyse_for,
    anisotropy,
    cost_table,
    dimension_table,
    ghost_table,
    masking_table,
    modes_table,
    resolve_extension,
    roc_curve,
    rotation_table,
    score_maps,
    sweep,
    threshold_table,
    verdict,
)
from eif_presets import (
    apply_preset,
    bounds,
    extension_options,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    psi_max,
    randomize_seed,
    sync_query_params,
)
from eif_visualization import (
    build_anisotropy,
    build_convergence,
    build_costs,
    build_cutoff,
    build_dimension,
    build_features,
    build_masking,
    build_mean_paths,
    build_method_bars,
    build_modes,
    build_path_lengths,
    build_roc,
    build_rotation,
    build_scatter,
    build_score_maps,
    build_scores,
    build_sweep,
    build_tree_plane,
    build_wrong_share,
    projection,
)

st.set_page_config(page_title="Extended Isolation Forest – Sebastian Hanisch", layout="wide")
BLUE_TXT, RED_TXT = "#1f77b4", "#d62728"


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _maps(X2, settings):
    xs, ys, S_if, S_eif = score_maps(X2, settings)
    mu, sd = X2.mean(axis=0), X2.std(axis=0)
    f_if = isf.fit_forest(X2, settings.n_trees, settings.psi, settings.start)
    f_eif = eif.fit_forest((X2 - mu) / sd, settings.n_trees, settings.psi, settings.start, 1)
    robust = alg.fit_mcd(X2, C.DEFAULT_SUPPORT, 0)
    ellipse = alg.ellipse_points(robust.location, robust.covariance, alg.chi2_ppf(0.975, 2), np.eye(2))
    aniso = (anisotropy(X2, lambda G: isf.score(f_if, G), 1.0), anisotropy(X2, lambda G: eif.score(f_eif, (G - mu) / sd), 1.0))
    return xs, ys, S_if, S_eif, ellipse, aniso


@st.cache_data(show_spinner=False)
def _rotation(base, settings):
    return rotation_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _ghost(base, settings):
    return ghost_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _modes(base, settings):
    return modes_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _dimension(base, settings):
    return dimension_table(settings, **dict(base)), sweep("n_noise", settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _masking(base, settings):
    return masking_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("🌳 Extended Isolation Forest – schräge Schnitte statt Geister")
st.markdown(
    """
Der **Isolation Forest** aus dem vorigen Stück trennt Touren mit zufälligen Schnitten ab - aber jeder Schnitt liegt **parallel zu einer Merkmalsachse**. Das hat eine Nebenwirkung: Punkte, die neben dem Datenbereich **entlang einer Achse** liegen,
bekommen andere Scores als gleich weit entfernte Punkte in den Ecken - **Geister-Regionen**, Bänder im Score. Der **Extended Isolation Forest (EIF)** ersetzt den achsenparallelen Schnitt durch eine **zufällige Hyperebene**:
ein zufälliger Normalenvektor und ein zufälliger Punkt im Wertebereich des Knotens. Ein Regler, der **Erweiterungsgrad ℓ**, legt fest, wie viele Merkmale in einen Schnitt eingehen (ℓ = 0: nur eines, das ist der Isolation Forest).
Was schräge Schnitte bringen - und was nicht - misst diese Demo, mit Siegen und Niederlagen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - drittes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, Isolation Forest und die Schätzer der Wurzel (klassisch, robust per MCD) sind wortgleich aus den Vorgänger-Demos übernommen, damit der Vergleich denselben Boden hat. Die Linie hat keinen Konvergenzpunkt; "
    "dieses Stück schließt den Ast der Zufallsbäume, die nächsten (lokale Dichte, Kernel-Grenze, ...) setzen an anderen Schwächen der Wurzel an."
)

with st.expander("So funktioniert der Extended Isolation Forest", expanded=True):
    st.markdown(
        """
1. **Wie der Isolation Forest.** Jeder Baum sieht $\\psi$ zufällig gezogene Touren; je Knoten wird geschnitten, bis jede Tour allein ist (oder die Tiefe $\\lceil \\log_2 \\psi \\rceil$ erreicht ist). Der Anomalie-Wert ist $s(x) = 2^{-E[h(x)] / c(\\psi)}$ - dieselbe Pfadlänge $h$ und Normierung $c$ wie im Vorgänger.
2. **Zufällige Hyperebene.** Statt "Merkmal $q$, Schnittpunkt $t$": ein Normalenvektor $n$ mit $\\ell + 1$ Gauß'schen Komponenten (die übrigen sind null) und ein Punkt $q$, gleichverteilt im Wertebereich des Knotens. Links liegt, was $(x - q) \\cdot n \\le 0$ erfüllt.
3. **Erweiterungsgrad $\\ell$.** $\\ell = 0$: eine Komponente, der Schnitt steht senkrecht zu einer Achse - der Isolation Forest. $\\ell = p - 1$: alle Komponenten, volle schräge Schnitte. Dazwischen liegen Mischformen.
4. **Standardisieren.** Eine Hyperebene mischt Merkmale in Metern, Minuten und Prozent - deshalb rechnet der EIF hier auf **standardisierten** Kennzahlen (der Isolation Forest ist skaleninvariant, der EIF nicht).
5. **Schwelle.** Wie im Vorgänger ohne kalibrierte Fehlalarmrate: die Faustregel "Score über 0,5" oder ein **erwarteter Anteil**.

Was **nicht** vorausgesetzt wird: eine Verteilungsform, ein Normalbereich, mehr Touren als Merkmale. Was der EIF **nicht** verspricht: bessere Erkennung in jedem Fall - das messen die Experimente unten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. Bei der Schwelle 0.5 hat der Isolation Forest F1 0.60 / 0.67 / 0.78 / 0.89 / 0.96 bei 20 / 30 / 50 / 100 / 200 Touren, der Extended IF 1.00 / 0.97 / 0.94 / 0.97 / 0.96 - der Isolation Forest "
             "verliert bei kleinen Stichproben (seine normalen Scores liegen höher, Fehlalarmrate 0.16 bei 20 Touren), der Extended IF kaum. Die AUC ist überall 1.00.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). F1 bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: Isolation Forest 0.92 / 0.92 / 0.94 / 0.96 / 0.96 / 0.95, Extended IF 0.94 / 0.96 / 0.98 / 0.98 / 0.96 / 0.97; "
             "die AUC ist bei beiden 1.00.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. Bei 0 / 10 / 20 / 30 / 40: Recall bei der Schwelle 0.5 beim Isolation Forest 0.99 / 0.91 / 0.71 / 0.58 / 0.39, beim Extended IF 0.97 / 0.83 / 0.71 / 0.56 / 0.41 "
             "(F1 0.96 / 0.94 / 0.82 / 0.72 / 0.56 gegen 0.98 / 0.90 / 0.83 / 0.71 / 0.58) - kein Vorteil für schräge Schnitte. Die AUC bleibt bei beiden 0.99-1.00, robust 1.00 / 0.99 / 0.96 / 0.91 / 0.88.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. F1 bei 1 / 2 / 3 Betriebsarten: Isolation Forest 0.96 / 0.96 / 0.97, Extended IF 0.98 / 0.95 / 0.94, robuste Schätzung der Wurzel 0.84 / 0.74 / 0.36 "
             "(AUC 1.00 / 0.95 / 0.85).",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 bleibt der F1 beider Wälder hoch (Isolation Forest 0.96 / 0.98 / 0.97 / 0.97 / 0.95, Extended IF 0.98 / 0.96 / 0.97 / 0.95 / 0.97); "
             "die robuste Schätzung mit χ²-Schwelle fällt auf 0.84 / 0.49 / 0.41 / 0.39 / 0.38.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. Bei 0 / 0.25 / 0.5 / 1.0: F1 des Isolation Forest 0.94 / 0.96 / 0.96 / 0.96, des Extended IF 0.98 / 0.98 / 0.97 / 0.94 (AUC beide 1.00).",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. F1 bei 2 / 5 / 10 / 20 / 30 / 40 / 45 %: Isolation Forest 0.42 / 0.81 / 0.96 / 0.97 / 0.93 / 0.87 / 0.81, Extended IF 0.69 / 0.96 / 0.98 / 0.89 / 0.72 / 0.52 / 0.42 - bei wenigen Anomalien "
             "ist der Extended IF besser (weniger Fehlalarme), bei vielen schlechter (Recall 0.27 gegen 0.68 bei 45 %); die AUC ist bei beiden überall 1.00.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe: alle beieinander - bei 10 % AUC Isolation Forest 0.95, Extended IF 0.97, robust 1.00. In der Lücke (ab zwei Betriebsarten): Isolation Forest 0.54, Extended IF 0.38, "
             "robust 0.40 - bei nur 2 % Anomalien 0.81 gegen 0.56.",
    )
    if kind != "gap":
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind. Bei 3 / 4 / 6 / 9 / 12: AUC beider Wälder 0.96 / 0.99 / 1.00 / 1.00 / 1.00; F1 des Isolation Forest 0.66 / 0.85 / 0.96 / 0.99 / 1.00, des Extended IF "
                 "0.72 / 0.88 / 0.98 / 1.00 / 1.00.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))
        st.session_state["strength_slider"] = strength

    st.markdown("**Extended Isolation Forest**")
    p_total = int(p_features) + int(n_noise)
    ext_options = extension_options(p_total)
    extension_kind = st.selectbox(
        "Erweiterungsgrad ℓ", ext_options, key="extension_select", format_func=lambda k: f"{C.EXTENSION_LABELS[k].split(' (')[0]} (ℓ = {resolve_extension(k, p_total)})" if k in ("half", "full") else
        f"ℓ = {resolve_extension(k, p_total)}" + (" (achsenparallel = Isolation Forest)" if k == "axis" else ""),
        help="Wie viele Merkmale in einen Schnitt eingehen (ℓ + 1). Im Standardfall bleibt die AUC bei 1.00 für alle Grade; der F1 bei der Schwelle 0.5 ist 0.96 / 0.97 / 0.97 / 0.98 / 0.98 bei ℓ = 0 / 1 / 2 / 5 / 11 "
             "(Fehlalarmrate 0.008 / 0.004 / 0.003 / 0.001 / 0.001: mit mehr Merkmalen je Schnitt sinken die Scores der normalen Touren). Die Geister-Regionen sind mit ℓ = 1 (bei zwei Merkmalen schon voll) kleiner (Anisotropie 0.081 → 0.029 bei 1 σ Abstand).",
    )
    n_trees = st.slider(
        "Bäume", *bounds("trees_slider"), key="trees_slider", step=10,
        help="Anzahl der Bäume je Wald. Der F1 bei der Schwelle 0.5 bei 10 / 25 / 50 / 100 / 200 / 500 Bäumen: Isolation Forest 0.94 / 0.95 / 0.96 / 0.96 / 0.96 / 0.96, Extended IF 0.95 / 0.98 / 0.97 / 0.98 / 0.98 / 0.98 "
             "(Recall des Extended IF bei 10 Bäumen 0.91). Die AUC ist überall 1.00.",
    )
    psi_hi = psi_max(int(n_tours))
    psi = st.slider(
        "Unterstichprobe ψ", C.PSI_MIN, psi_hi, key="psi_slider", step=1,
        help="Touren je Baum (höchstens die Tourenzahl). F1 bei ψ = 16 / 32 / 64 / 128 / 256: Isolation Forest 0.56 / 0.71 / 0.83 / 0.91 / 0.96 (Fehlalarmrate 0.18 bei 16), Extended IF 0.92 / 0.96 / 0.96 / 0.98 / 0.98 (0.02 bei 16) - "
             "der Score des Isolation Forest hängt stark an ψ, der des Extended IF wenig; die AUC bleibt bei 0.99-1.00.",
    )
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: die Wälder markieren Score über der Schwelle (nominell 0.5), klassisch und robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        cutoff = st.slider(
            "Score-Schwelle (beide Wälder)", *bounds("cutoff_slider"), key="cutoff_slider", step=0.01,
            help="Ab welchem Anomalie-Wert eine Tour markiert wird. Bei 0.45 / 0.5 / 0.55 / 0.6 / 0.65: F1 des Isolation Forest 0.82 / 0.96 / 0.97 / 0.83 / 0.55, des Extended IF 0.92 / 0.98 / 0.85 / 0.40 / 0.04 (Recall 0.99 / 0.97 / 0.75 / 0.25 / 0.02): "
                 "die Scores des Extended IF liegen niedriger, sein Fenster reicht weniger weit nach oben.",
        )
        quantile = st.slider(
            "Schwelle: χ²-Quantil (klassisch, robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei den Schätzern der Wurzel als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90, "
                 "der klassischen 0.69 / 0.66 / 0.57 / 0.44 / 0.15.",
        )
        st.session_state["_cutoff_kept"] = cutoff
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
        st.session_state["share_slider"] = share
    else:
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % beim Isolation Forest "
                 "0.33 / 0.67 / 0.97 / 0.67 / 0.40, beim Extended IF 0.33 / 0.67 / 0.99 / 0.67 / 0.40, für die robuste Schätzung 0.33 / 0.67 / 0.90 / 0.66 / 0.40, für die klassische 0.32 / 0.56 / 0.69 / 0.58 / 0.39.",
        )
        st.session_state["_share_kept"] = share
        cutoff = float(st.session_state.get("_cutoff_kept", C.DEFAULT_CUTOFF))
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
        st.session_state["cutoff_slider"], st.session_state["quantile_slider"] = cutoff, quantile
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "extension_select": extension_kind, "trees_slider": int(n_trees), "psi_slider": int(psi),
    "threshold_kind_select": threshold_kind, "cutoff_slider": float(cutoff), "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), 0.0, int(seed))
settings = Settings(extension=extension_kind, n_trees=int(n_trees), psi=int(psi), threshold_kind=threshold_kind, cutoff=float(round(cutoff, 2)), quantile=float(quantile), share=int(share))
with st.spinner("Baue die Wälder..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ifs, es, cs, rs = (a.scores[d] for d in ("iforest", "eif", "classical", "robust"))
n_anom = int(ds.anomaly.sum())
n_total = ds.p + ds.n_noise
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- Extended Isolation Forest in Aktion --------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Extended Isolation Forest in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Schräge Schnitte", 3: "3 · Pfadlängen und Score", 4: "4 · Geister-Regionen", 5: "5 · Ergebnis"}
if "eif_step" not in st.session_state or st.session_state.get("eif_step_owner") != data_key:
    st.session_state["eif_step"] = 1
    st.session_state["eif_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="eif_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
c_psi = float(isf.c_factor(a.forest_if.psi))
vals_if, vals_eif = a.values["iforest"], a.values["eif"]
flag_e = a.flags["eif"]
i_anom = int(np.argmax(np.where(ds.anomaly, vals_if + vals_eif, -np.inf)))
i_norm = int(np.argmin(np.where(~ds.anomaly, np.abs(P).sum(axis=1), np.inf)))
tree_axis = eif.build_tree(P, np.random.default_rng([seed, 99]), isf.height_limit(len(P)), 0)
tree_slant = eif.build_tree(P, np.random.default_rng([seed, 98]), isf.height_limit(len(P)), 1)
plane = (P[:, 0].min() - 0.5, P[:, 0].max() + 0.5, P[:, 1].min() - 0.5, P[:, 1].max() + 0.5)
seg_axis, seg_slant = eif.tree_segments(tree_axis, plane), eif.tree_segments(tree_slant, plane)
steps_c = sorted({1, 2, 5, 10, 20, 50, 100, 200, 500} & set(range(1, a.paths_if.shape[0] + 1)) | {a.paths_if.shape[0]})
conv_ids = list(np.flatnonzero(ds.anomaly)[:4]) + list(np.flatnonzero(~ds.anomaly)[:4])
picks = [(i_norm, "normale Tour", BLUE_TXT), (i_anom, "Sonderfahrt", RED_TXT)]


def _depth(tree, idx):
    return len(eif.point_path(tree, P[idx])[0])


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {ds.names[0]} gegen {ds.names[min(1, ds.p - 1)]}** (Einheiten wie gemessen)")
            c2.plotly_chart(build_features(ds.X, ds.anomaly, ds.names, 0, 1), width="stretch", key="step_features")
        elif current_step == 2:
            c1, c2 = st.columns(2)
            c1.markdown("**Achsenparallele Schnitte** (ℓ = 0: ein Isolationsbaum auf den zwei Hauptrichtungen, Illustration)")
            c1.plotly_chart(build_tree_plane(P, ds.anomaly, seg_axis, picks), width="stretch", key="step_tree_axis")
            c2.markdown("**Schräge Schnitte** (ℓ = 1: zufällige Geraden, derselbe Baumaufbau)")
            c2.plotly_chart(build_tree_plane(P, ds.anomaly, seg_slant, picks), width="stretch", key="step_tree_slant")
            st.markdown(f"**Pfadlängen derselben zwei Touren in allen {a.paths_if.shape[0]} Bäumen der echten Wälder (alle {n_total} Merkmale, ℓ = {a.level})**")
            st.plotly_chart(build_path_lengths(a.paths_if[:, i_anom], a.paths_eif[:, i_anom], c_psi), width="stretch", key="step_paths")
        elif current_step == 3:
            c1, c2 = st.columns(2)
            c1.markdown("**Mittlere Pfadlänge je Tour** (Sonderfahrten kräftig, normale Touren blass; gestrichelt: c(ψ))")
            c1.plotly_chart(build_mean_paths(a.paths_if.mean(axis=0), a.paths_eif.mean(axis=0), ds.anomaly, c_psi), width="stretch", key="step_mean_paths")
            c2.markdown("**Anomalie-Wert nach den ersten k Bäumen** (vier Sonderfahrten, vier normale Touren)")
            conv_if = isf.score_convergence(a.paths_if[:, conv_ids], a.forest_if.psi, steps_c)
            conv_eif = isf.score_convergence(a.paths_eif[:, conv_ids], a.forest_eif.psi, steps_c)
            c2.plotly_chart(build_convergence(steps_c, conv_if, conv_eif, ds.anomaly[conv_ids]), width="stretch", key="step_convergence")
            st.markdown("**Anomalie-Wert (Score) mit der Schwelle**")
            st.plotly_chart(build_scores(vals_if, vals_eif, ds.anomaly, ifs["threshold"]), width="stretch", key="step_scores")
        elif current_step == 4:
            X2 = ds.X[~ds.anomaly][:, :2]
            xs, ys, S_if, S_eif, ell, aniso = _maps(X2, settings)
            st.markdown(f"**Score-Karten über die ersten zwei Merkmale ({ds.names[0]}, {ds.names[1]}), ohne Anomalien** (Wälder nur auf diesen zwei Merkmalen; dunkler = auffälliger; gestrichelt: robuste Ellipse)")
            st.plotly_chart(build_score_maps(xs, ys, S_if, S_eif, X2, ds.names[:2], ell), width="stretch", key="step_maps")
            st.session_state["_aniso"] = aniso
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("iforest", "eif", "classical", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {n_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung in standardisierten Kennzahlen; die Wälder arbeiten mit allen Kennzahlen.")
elif step == 2:
    st.caption(f"Oben: zwei Isolationsbäume auf denselben Punkten - links mit achsenparallelen Schnitten, rechts mit schrägen Geraden. Die Sonderfahrt ist im achsenparallelen Baum nach {_depth(tree_axis, i_anom)} Schnitten allein, im schrägen nach "
               f"{_depth(tree_slant, i_anom)}; die normale Tour braucht {_depth(tree_axis, i_norm)} bzw. {_depth(tree_slant, i_norm)}. Unten die echten Wälder: mittlere Pfadlänge der Sonderfahrt "
               f"{a.paths_if[:, i_anom].mean():.1f} (Isolation Forest) und {a.paths_eif[:, i_anom].mean():.1f} (Extended IF, ℓ = {a.level}); c(ψ) = {c_psi:.1f}, ψ = {a.forest_if.psi}.")
elif step == 3:
    st.caption(f"Mittlerer Score der normalen Touren: Isolation Forest {vals_if[~ds.anomaly].mean():.3f}, Extended IF {vals_eif[~ds.anomaly].mean():.3f}; der Sonderfahrten {vals_if[ds.anomaly].mean():.3f} und {vals_eif[ds.anomaly].mean():.3f}. "
               "Der Extended IF gibt den normalen Touren meist niedrigere Scores als der Isolation Forest (in den Experimenten: 0.37 gegen 0.38 bei 300 Touren, bei 20 Touren 0.39 gegen 0.43) - die Schwelle 0.5 trifft ihn deshalb anders.")
elif step == 4:
    aniso = st.session_state.get("_aniso", (float("nan"), float("nan")))
    st.caption(f"Links die Bänder entlang der Achsen (Geister-Regionen), rechts der Extended IF mit ℓ = 1 (bei zwei Merkmalen der volle Grad). Anisotropie bei 1 σ Abstand zum Datenrand (Score der Ecken minus der Korridore): "
               f"Isolation Forest {aniso[0]:.3f}, Extended IF {aniso[1]:.3f}. Die Ellipse der Wurzel ist im Abstand isotrop.")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Wälder gefunden haben – Isolation Forest gegen Extended IF gegen Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: Score über der Score-Schwelle bei den Wäldern, χ²-Quantil bei klassisch und robust). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (Extended IF)", f"{es['auc']:.2f}", delta=f"Isolation Forest {ifs['auc']:.2f} · robust {rs['auc']:.2f}", delta_color="off", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m2.metric("Recall (Extended IF)", _pct(es["recall"]), delta=f"Isolation Forest {_pct(ifs['recall'])} · robust {_pct(rs['recall'])}", delta_color="off", help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m3.metric("Fehlalarmrate (Extended IF)", f"{es['false_alarm']:.1%}", delta=f"Isolation Forest {ifs['false_alarm']:.1%} · robust {rs['false_alarm']:.1%}", delta_color="off", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m4.metric("F1 (Extended IF)", f"{es['f1']:.2f}", delta=f"Isolation Forest {ifs['f1']:.2f} · mit bekanntem Anteil {a.oracle_f1['eif']:.2f}", delta_color="off",
          help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter der F1 des Isolation Forest und der des Extended IF, wenn der wahre Anteil bekannt wäre.")

_t = vd
if code == "eif_wins":
    st.success(f"✅ Der Extended IF ist besser: F1 {_t['eif_f1']:.2f} gegen {_t['iforest_f1']:.2f} bei der Schwelle (Recall {_pct(_t['eif_recall'])} gegen {_pct(_t['iforest_recall'])}; AUC {_t['eif_auc']:.2f} gegen {_t['iforest_auc']:.2f}). "
               f"Mit bekanntem Anteil wären es {a.oracle_f1['eif']:.2f} gegen {a.oracle_f1['iforest']:.2f}. Der Vorteil liegt hier vor allem bei der Schwelle: der Score des Extended IF hängt weniger an kleinen Stichproben und Unterstichproben.")
elif code == "if_wins":
    st.warning(f"⚠️ Der einfachere Isolation Forest ist besser: AUC {_t['iforest_auc']:.2f} gegen {_t['eif_auc']:.2f} (Extended IF), F1 {_t['iforest_f1']:.2f} gegen {_t['eif_f1']:.2f}. Der Extended IF umschließt dichte Gruppen mit schrägen Schnitten noch besser - "
               "für Anomalien, die selbst eine dichte Gruppe bilden (oder in der Lücke liegen), ist das ein Nachteil.")
elif code == "comparable":
    st.success(f"✅ Beide Wälder ranken gleich gut (AUC {_t['iforest_auc']:.2f} und {_t['eif_auc']:.2f}); bei der Schwelle: F1 {_t['iforest_f1']:.2f} (Isolation Forest) und {_t['eif_f1']:.2f} (Extended IF), robust {_t['robust_f1']:.2f}, klassisch {_t['classical_f1']:.2f}. "
               "Die Geister-Regionen des Isolation Forest sind messbar (Schritt 4), ändern die Erkennung in diesem Szenario aber nicht.")
elif code == "dense_masking":
    st.warning(f"⚠️ Dichte Gruppe: die robuste Schätzung der Wurzel ist besser (AUC {_t['robust_auc']:.2f} gegen {_t['iforest_auc']:.2f} beim Isolation Forest und {_t['eif_auc']:.2f} beim Extended IF). Die Gruppe ({_t['contamination']:.0f} % der Touren) ist auch in der "
               f"Unterstichprobe dicht. Bei der Schwelle: F1 {_t['iforest_f1']:.2f} (Isolation Forest) und {_t['eif_f1']:.2f} (Extended IF), Recall {_pct(_t['iforest_recall'])} und {_pct(_t['eif_recall'])}.")
elif code == "root_wins":
    st.warning(f"⚠️ Die robuste Schätzung der Wurzel trennt besser: AUC {_t['robust_auc']:.2f} gegen {_t['iforest_auc']:.2f} (Isolation Forest) und {_t['eif_auc']:.2f} (Extended IF).")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['eif_auc']:.2f} beim Extended IF (meist schlechter: schräge Schnitte umschließen die dichte Gruppe in der Lücke noch besser), "
               f"{_t['robust_auc']:.2f} robust, {_t['classical_auc']:.2f} klassisch. Bei nur 2 % Anomalien in der Lücke findet der Isolation Forest sie im Mittel noch (AUC 0.81), der Extended IF nicht mehr (0.56).")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist perfekt (AUC {_t['iforest_auc']:.2f} und {_t['eif_auc']:.2f}), aber bei der gewählten Schwelle ist F1 {_t['iforest_f1']:.2f} (Isolation Forest) und {_t['eif_f1']:.2f} (Extended IF); "
               f"mit dem wahren Anteil wären es {a.oracle_f1['iforest']:.2f} und {a.oracle_f1['eif']:.2f}. Beide Wälder haben keine kalibrierte Fehlalarmrate.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Schwelle", "threshold", "{:.2f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "Isolation Forest": [r[2].format(ifs[r[1]]) for r in rows], "Extended IF": [r[2].format(es[r[1]]) for r in rows],
              "klassisch": [r[2].format(cs[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Bäume × ψ", "Schnitte", "mittlerer Score (normal / Anomalie)"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} × {a.forest_if.psi}", "achsenparallel (1 Merkmal)", f"{vals_if[~ds.anomaly].mean():.2f} / {vals_if[ds.anomaly].mean():.2f}"],
              "Extended IF": [f"{a.seconds['eif'] * 1000:.0f} ms", f"{len(a.forest_eif.trees)} × {a.forest_eif.psi}", f"schräg ({a.level + 1} Merkmale)", f"{vals_eif[~ds.anomaly].mean():.2f} / {vals_eif[ds.anomaly].mean():.2f}"],
              "robust (MCD)": [f"{a.seconds['robust'] * 1000:.0f} ms", f"h = {a.robust.h}", "Kovarianz (alle Merkmale)", f"d² {a.values['robust'][~ds.anomaly].mean():.1f} / {a.values['robust'][ds.anomaly].mean():.1f}"]})
    st.caption("Der Isolation Forest ist skaleninvariant, der Extended IF nicht: er rechnet auf standardisierten Kennzahlen (auf Rohdaten fiele seine AUC im Standardfall von 1.00 auf 0.93, siehe Experiment Kosten und Einheiten).")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Erweiterungsgrad, Anomalien und Wald ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind == "gap" and k in ("strength", "n_modes")) or (threshold_kind == "share" and k in ("cutoff", "quantile")) or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"extension": resolve_extension(extension_kind, p_total), "n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise),
           "contamination": int(contamination), "strength": float(strength), "n_trees": int(n_trees), "psi": int(psi), "cutoff": float(cutoff), "quantile": float(quantile), "share": int(share)}[sweep_param]
sweep_values = SWEEP_VALUES[sweep_param]
if sweep_param == "extension":
    sweep_values = tuple(sorted({min(v, p_total - 1) for v in SWEEP_VALUES["extension"]} | {p_total - 1}))
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, sweep_values)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Beim Erweiterungsgrad ändern sich nur die Linien des Extended IF, bei den Reglern der Wälder (Bäume, Unterstichprobe, Schwelle) nicht die der Wurzel.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Drehung der Merkmale: hängt der Isolation Forest an den Achsen?")
if st.button("Drehung 0, ½ und 1 vergleichen (dauert etwa 10 Sekunden)", key="rotation_start"):
    st.session_state["rotation_on"] = True
if st.session_state.get("rotation_on"):
    with st.spinner("Drehe die Merkmale und vergleiche die Scores..."):
        rt = _rotation(base_data, settings)
    st.plotly_chart(build_rotation(rt), width="stretch", key="rotation_chart")
    st.caption("Je zwei aufeinanderfolgende Merkmale werden in standardisierten Einheiten um bis zu 45° gedreht (die Mahalanobis-Abstände und damit die robuste Schätzung ändern sich dabei nicht). Erwartet war: der Isolation Forest verliert mit der Drehung, "
               "der Extended IF nicht. Gemessen ist das **nicht** so: AUC und F1 bleiben gleich, und die Scores der normalen Touren ändern sich nicht stärker als durch einen anderen Wald-Seed auf denselben Daten (Rangkorrelation "
               f"{rt['rows'][-1]['iforest_corr']:.2f} gegen den Boden {rt['floor']['iforest']:.2f} beim Isolation Forest; {rt['rows'][-1]['eif_corr']:.2f} gegen {rt['floor']['eif']:.2f} beim Extended IF). "
               "Die Kennzahlen des Szenarios sind über die zwei Faktoren ohnehin schräg zueinander - es gibt keine achsenparallele Struktur, die eine Drehung zerstören könnte.")

st.markdown("---")

st.subheader("🔬 Geister-Regionen: was schräge Schnitte beseitigen")
if st.button("Anisotropie je Betriebsart berechnen (dauert etwa 15 Sekunden)", key="ghost_start"):
    st.session_state["ghost_on"] = True
if st.session_state.get("ghost_on"):
    with st.spinner("Berechne die Anisotropie..."):
        g_rows = _ghost(tuple(kv for kv in base_data if kv[0] in ("n", "noise", "curvature")), settings)
    st.plotly_chart(build_anisotropy(g_rows), width="stretch", key="ghost_aniso")
    st.caption("Score der Rasterpunkte außerhalb der Wertebereiche beider Merkmale (Ecken) minus der neben dem Datenbereich (Korridore), bei gleichem Abstand zum nächsten Datenpunkt; nur die zwei ersten Merkmale, Wälder auf diesen zwei Merkmalen "
               "(Mittel über 5 feste Datensätze; Schritt 4 zeigt die Karten). Der Extended IF verkleinert die Bänder auf etwa ein Drittel (eine Betriebsart, 1 σ Abstand: 0.081 → 0.029) bis fast null (drei Betriebsarten, 2 σ: 0.061 → 0.001). "
               "Die Erkennung im Szenario ändert das nicht: die AUC ist bei beiden 1.00.")

st.markdown("---")

st.subheader("🔬 Die Schwächen der Wurzel: Lücke, Betriebsarten, dichte Gruppe")
if st.button("Betriebsarten × Art der Anomalien vergleichen (dauert etwa 15 Sekunden)", key="modes_start"):
    st.session_state["modes_on"] = True
if st.session_state.get("modes_on"):
    with st.spinner("Vergleiche 1-3 Betriebsarten × 3 Arten × 5 Datensätze..."):
        mt = _modes(tuple(kv for kv in base_data if kv[0] not in ("kind", "n_modes")), settings)
    st.plotly_chart(build_modes(mt), width="stretch", key="modes_chart")
    st.caption("AUC der vier Detektoren (Mittel über 5 feste Datensätze). Verstreute Anomalien findet jeder Wald bei jeder Zahl von Betriebsarten (AUC 1.00). Bei einer dichten Gruppe abseits sind beide Wälder etwa gleich (0.95 und 0.97 bei einer Betriebsart), "
               "die robuste Schätzung bei ein bis zwei Betriebsarten besser (1.00). Anomalien in der Lücke findet keiner: der Extended IF ist bei zwei Betriebsarten schlechter als der Isolation Forest (0.38 gegen 0.54), "
               "bei drei etwas besser (0.31 gegen 0.27) - alle vier bleiben unter 0.55.")

st.markdown("---")

st.subheader("🔬 Rauschmerkmale und wenige Touren bei vielen Merkmalen")
if st.button("Rauschmerkmale und Tourenzahl × Merkmalszahl durchfahren (dauert etwa 70 Sekunden)", key="dimension_start"):
    st.session_state["dimension_on"] = True
if st.session_state.get("dimension_on"):
    with st.spinner("Rechne 6 Tourenzahlen × 5 Merkmalszahlen × 5 Datensätze und die Rauschmerkmale..."):
        dt, nt = _dimension(tuple(kv for kv in base_data if kv[0] not in ("n", "p", "n_noise")), settings)
    st.markdown("**Rauschmerkmale 0 bis 40**")
    st.plotly_chart(build_sweep(nt, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_chart")
    st.markdown("**F1 bei der Schwelle: Tourenzahl × Merkmalszahl**")
    st.plotly_chart(build_dimension(dt), width="stretch", key="dimension_chart")
    st.caption("Mittel über 5 feste Datensätze. Die AUC beider Wälder bleibt bei jeder Tourenzahl und Merkmalszahl bei 1.00 und bei Rauschmerkmalen bei 0.99-1.00. Bei 40 Rauschmerkmalen fallen beide an der Schwelle 0.5 gleich stark (F1 0.56 und 0.58, Recall 0.39 und 0.41) - "
               "schräge Schnitte helfen dabei nicht. Bei wenigen Touren (n ≤ 50) verliert nur der Isolation Forest (F1 0.54-0.79 bei n ≤ 50; Extended IF 0.67-1.00), er markiert mehr normale Touren.")

st.markdown("---")

st.subheader("🔬 Die Schwelle: der Score des Extended IF liegt tiefer und reicht weniger weit")
if st.button("Schwellen vergleichen (dauert etwa 25 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 5 Schwellen × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**F1 und Fehlalarmrate beider Wälder je Score-Schwelle**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.table({"Tourenzahl": [r["n"] for r in tt["normal_scores"]], "mittlerer Score der normalen Touren: Isolation Forest": [f"{r['iforest']:.3f}" for r in tt["normal_scores"]],
              "Extended IF": [f"{r['eif']:.3f}" for r in tt["normal_scores"]]})
    st.caption("Links: die Scores des Extended IF liegen niedriger - bei der Schwelle 0.45 markiert er nur 1.8 % der normalen Touren (Isolation Forest 5.0 %), bei 0.6 aber nur noch 25 % der Anomalien (Isolation Forest 71 %): "
               "sein brauchbares Fenster (F1 ≥ 0.8) liegt bei 0.45-0.5 und endet unter 0.55, das des Isolation Forest reicht von 0.45 bis 0.6. "
               "Rechts: ein falsch angenommener Anteil kostet alle vier Detektoren gleich viel, denn dann entscheidet nur die Rangfolge. Tabelle: der mittlere Score der normalen Touren fällt beim Isolation Forest mit der Tourenzahl von 0.43 auf 0.38, "
               "beim Extended IF von 0.39 auf 0.37 - er hängt weniger an der Stichprobengröße.")

st.markdown("---")

st.subheader("🔬 Masking: wenn die Anomalien eine dichte Gruppe bilden")
if st.button("Anteil der dichten Gruppe von 2 bis 45 % durchfahren (dauert etwa 25 Sekunden)", key="masking_start"):
    st.session_state["masking_on"] = True
if st.session_state.get("masking_on"):
    with st.spinner("Rechne 10 Anteile × 5 Datensätze und 5 Unterstichproben..."):
        mk = _masking(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_masking(mk), width="stretch", key="masking_chart")
    st.caption("Mittel über 5 feste Datensätze, eine Betriebsart. Bis 15 % Gruppenanteil sind die Wälder gleich (AUC 0.91), bei 20 % ebenfalls (0.85) - aber bei der Schwelle findet der Isolation Forest 44 % der Gruppe, der Extended IF nur 9 % (F1 0.44 und 0.13). "
               "Die robuste Schätzung trennt bis 25 % perfekt (1.00) und kippt bei 30 % auf 0.49, die Wälder fallen langsamer (0.70 und 0.57 bei 30 %). Ab 40 % versagen alle: die Gruppe ist dann der Normalbereich, der Extended IF mit 0.35 und 0.20 am deutlichsten unter Raten. "
               "Rechts: eine kleinere Unterstichprobe hilft beim Isolation Forest (AUC 0.84 mit ψ = 16, 0.70 mit ψ = 256), beim Extended IF weniger (0.85 gegen 0.57).")

st.markdown("---")

st.subheader("🔬 Kosten und Einheiten")
if st.button("Rechenzeit und Rohdaten-Vergleich berechnen (dauert etwa 10 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Rechenzeit: Wald bauen und alle Touren bewerten**")
    c1.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    c2.markdown("**Extended IF auf standardisierten Kennzahlen gegen Rohdaten**")
    c2.table({"": ["standardisiert", "Rohdaten (Meter, Minuten, ...)"], "AUC": [f"{ct['units'][0]['eif_auc']:.2f}", f"{ct['units'][1]['eif_auc']:.2f}"], "F1 bei der Schwelle": [f"{ct['units'][0]['eif_f1']:.2f}", f"{ct['units'][1]['eif_f1']:.2f}"],
              "Recall": [f"{ct['units'][0]['eif_recall']:.2f}", f"{ct['units'][1]['eif_recall']:.2f}"]})
    st.caption("Mittel über 3 bzw. 5 feste Datensätze. Beide Wälder sind numpy-Implementierungen mit derselben Baum-Bauweise; der Extended IF ist etwas langsamer als der Isolation Forest (auf diesem Rechner 0.09 s gegen 0.07 s bei 2 Merkmalen, 0.18 s gegen 0.14 s bei 12, 0.29 s gegen 0.27 s bei 30). "
               "Die Einheiten spielen für den Isolation Forest keine Rolle, für den Extended IF schon: auf Rohdaten dominiert die Distanz in Metern die Richtung der Schnitte (AUC 0.93, F1 0.65 statt 1.00 und 0.98).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Schräge Schnitte lösen die Geister-Regionen** | Sie schrumpfen (Anisotropie 0.081 → 0.029 bei 1 σ, drei Betriebsarten 2 σ: 0.061 → 0.001), verschwinden aber bei kleinen Abständen nicht ganz. Die Erkennung ändert das im Szenario nicht: AUC 1.00 bei beiden, Drehung der Merkmale ohne Wirkung (Rangkorrelation 0.94 = Seed-Rauschen). | (der Gewinn liegt dort, wo Anomalien wirklich in den Bändern liegen - im Szenario nicht der Fall) |
| **Der Score ist eine Schwelle** | Der Score hat auch beim Extended IF keine kalibrierte Fehlalarmrate, und sein Fenster liegt anders: Fehlalarmrate 0.018 bei Score-Schwelle 0.45 (Isolation Forest 0.050), Recall 0.25 bei 0.6 (Isolation Forest 0.71). Der Isolation Forest hängt stärker an ψ und n: Fehlalarmrate 0.18 bei ψ = 16 (Extended IF 0.02), 0.16 bei 20 Touren (Extended IF 0.00). | Kalibrierung; parameterfreie Verfahren (**ECOD**) |
| **Irrelevante Merkmale kosten Schnitte** | Bei 40 Rauschmerkmalen und Schwelle 0.5: Recall Isolation Forest 0.39, Extended IF 0.41 (F1 0.56 gegen 0.58) - die AUC ist bei beiden 0.99. Schräge Schnitte helfen dabei **nicht**: auch sie verschwenden Schnitte an Rauschen. | Ensembles über Merkmalsteilmengen (**Feature Bagging**) |
| **Dichte Anomaliegruppen sind schwer zu isolieren** | Beide Wälder gleich bei 20 % (AUC 0.85), bei der Schwelle ist der Extended IF schlechter (F1 0.13 gegen 0.44), beide hinter der robusten Schätzung (1.00). **Anomalien in der Lücke**: Isolation Forest 0.54, Extended IF 0.38 (2 %: 0.81 gegen 0.56) - schräge Schnitte helfen dort nicht, sie schaden. | lokale Dichte (**LOF**) |
| **Einheiten spielen keine Rolle** | Für den Isolation Forest stimmt das (skaleninvariant), für den Extended IF nicht: auf Rohdaten AUC 0.93 und F1 0.65 statt 1.00 und 0.98 - die Kennzahlen müssen standardisiert werden. | (Vorverarbeitung) |
| **Mehr Bäume sind nur Rechenzeit** | Für die AUC stimmt das (1.00 schon bei 10 Bäumen); an der Schwelle 0.5 hat der Extended IF bei 10 Bäumen Recall 0.91 statt 0.97 bei 100 (F1 0.95 gegen 0.98, Isolation Forest 0.94 und 0.96). | mehr Bäume |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope und der Isolation Forest (gebaut), LOF und Feature Bagging, One-Class SVM und Deep SVDD, ECOD und ein Autoencoder (noch nicht gebaut). "
    "Keiner ist überlegen: der Extended IF beseitigt ein messbares Artefakt des Isolation Forest, bringt aber sonst nur in einigen Fällen (Rauschmerkmale, dichte Gruppen) und bleibt ein Schwellenproblem."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Hyperebenen-Schnitt.** Auf einer Unterstichprobe $S$ ($|S| = \psi$, $x \in \mathbb{R}^p$): Knoten mit Punktmenge $Q$: $n \in \mathbb{R}^p$ mit $\ell + 1$ zufällig gewählten Komponenten $\sim \mathcal{N}(0, 1)$, die übrigen $= 0$
(bevorzugt auf nicht konstanten Merkmalen); $q$ gleichverteilt im Quader $\prod_j [\min_{x \in Q} x_j, \max_{x \in Q} x_j]$; $Q_L = \{x: (x - q)^\top n \le 0\}$, $Q_R = Q \setminus Q_L$. Ein Knoten mit leerem Kind oder gleichen Punkten bleibt Blatt.
Abbruch bei $|Q| = 1$ oder Tiefe $\lceil \log_2 \psi \rceil$. Für $\ell = 0$ ist $n = \pm\, e_j$ ein Einheitsvektor: der Schnitt steht senkrecht zur Achse $j$ (Isolation Forest).

**Pfadlänge und Score** (unverändert): $h(x) = e + c(|Q_\text{Blatt}|)$, $c(n) = 2\,(\ln(n-1) + \gamma) - 2(n-1)/n$; $s(x, \psi) = 2^{-E[h(x)] / c(\psi)}$.

**Skalenverhalten.** Der Isolation Forest ist gegen Skalierung und Verschiebung jedes Merkmals unempfindlich, der Extended IF nicht: die Richtung $n$ wird in den Koordinaten der Merkmale gezogen. Darum rechnet er auf standardisierten Kennzahlen $z_j = (x_j - \bar x_j)/s_j$.

**Score-Anisotropie.** Mittlerer Score der Rasterpunkte außerhalb der Wertebereiche beider Merkmale ("Ecken") minus der in mindestens einem Wertebereich ("Korridore"), bei gleichem standardisiertem Abstand zum nächsten Datenpunkt ($\pm 0{,}15$).

**Drehung.** In je zwei aufeinanderfolgenden Merkmalen wird der standardisierte Raum um $\theta = r \cdot 45^\circ$ gedreht ($r \in [0, 1]$), dann in die ursprüngliche Größenordnung zurück; orthogonal und affin: Mahalanobis-Abstände und MCD ändern sich nicht.

**Schwelle und Vergleich.** Standard: $s > 0{,}5$ (Faustregel) bzw. das $\chi^2$-Quantil der Wurzel; alternativ die $\lceil \alpha n \rceil$ größten Werte bei angenommenem Anteil $\alpha$. Kennzahlen: AUC (Rangsumme, Bindungen halb), mittlere Präzision, Precision, Recall, F1, Fehlalarmrate.

**Grenzen.** (1) Die Geister-Regionen schrumpfen, verschwinden aber nicht. (2) Dichte Gruppen und die Lücke werden von schrägen Schnitten noch besser umschlossen (Masking ab 20 %). (3) Keine kalibrierte Schwelle; die Scores liegen tiefer und hängen weniger an n und ψ als beim Isolation Forest. (4) Nicht skaleninvariant.

Implementiert in `eif_algorithm.py` (Hyperebenen-Baum, Wald, Pfadlängen, Zerlegung der Ebene), `eif_isolation_forest.py` und `eif_ee_algorithm.py` (Isolation Forest, klassisch und MCD, wortgleich aus den Vorgängern),
`eif_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen, Drehung), `eif_evaluation.py` (Kennzahlen, Sweeps, Experimente, Anisotropie, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
