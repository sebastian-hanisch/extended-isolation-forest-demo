"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import eif_constants as C
from eif_evaluation import SWEEP_LABELS
from eif_presets import PRESET_KEYS, psi_max

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"Standardfall": "success", "Kleine Unterstichprobe (ψ = 16)": "success", "Wenige Touren, viele Merkmale": "success", "Dichte Gruppe abseits (20 %)": "warning",
                 "Anomalien in der Lücke": "warning", "Wenige Anomalien in der Lücke (2 %)": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    """Erst Tourenzahl und Betriebsarten setzen und laufen lassen (die Grenzen der Unterstichprobe und die Art "Lücke" hängen davon ab; AppTest prüft gegen die Optionen des vorigen Laufs), dann alles andere."""
    at.session_state["n_tours_slider"] = p["n"]
    at.session_state["n_modes_slider"] = p["n_modes"]
    at.run()
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox)}


def test_default_renders_without_exception():
    at = _run()
    assert any("Extended Isolation Forest in Aktion" in m.value for m in at.markdown)
    assert not at.error and len(at.success) == 1 and not at.warning


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["p_slider"] = C.P_MAX
        at.session_state["n_noise_slider"] = C.N_NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MIN
        at.session_state["trees_slider"] = C.TREES_MIN
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["p_slider"] = C.P_MIN
        at.session_state["n_modes_slider"] = C.N_MODES_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MAX
        at.session_state["kind_select"] = "cluster"
        at.session_state["strength_slider"] = C.STRENGTH_MIN
        at.session_state["trees_slider"] = C.TREES_MAX
    _run(large)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 12, 30])
def test_every_step_renders(step, p):
    def setup(at):
        at.session_state["p_slider"] = p
        at.session_state["eif_step"] = step
    _run(setup)


def test_every_step_renders_with_few_tours_many_features_noise_and_the_share_threshold():
    def setup(at):
        at.session_state["n_tours_slider"] = 20
        at.session_state["p_slider"] = 30
        at.session_state["n_noise_slider"] = 40
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["eif_step"] = step
        at.run()
        assert not at.exception
    at = _run(lambda a: a.session_state.__setitem__("threshold_kind_select", "share"))
    for step in (4, 5):
        at.session_state["eif_step"] = step
        at.run()
        assert not at.exception


def test_psi_slider_is_capped_by_the_number_of_tours():
    at = _run()
    psi_slider = [s for s in at.sidebar.slider if s.label == "Unterstichprobe ψ"][0]
    assert psi_slider.max == psi_max(300) == 300
    at.session_state["n_tours_slider"] = 100
    at.run()
    assert not at.exception
    psi_slider = [s for s in at.sidebar.slider if s.label == "Unterstichprobe ψ"][0]
    assert psi_slider.max == 100 and psi_slider.value == 100                                   # der Zustand wurde auf die Tourenzahl begrenzt


def test_threshold_controls_are_hidden_per_kind_and_their_values_are_kept():
    at = _run(lambda a: (a.session_state.__setitem__("cutoff_slider", 0.55), a.session_state.__setitem__("quantile_slider", 0.99)))
    assert {"Score-Schwelle (beide Wälder)", "Schwelle: χ²-Quantil (klassisch, robust)"} <= _labels(at) and "Angenommener Anteil der Anomalien [%]" not in _labels(at)
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert not at.exception and "Angenommener Anteil der Anomalien [%]" in _labels(at) and not ({"Score-Schwelle (beide Wälder)", "Schwelle: χ²-Quantil (klassisch, robust)"} & _labels(at))
    at.session_state["share_slider"] = 25
    at.run()
    at.session_state["threshold_kind_select"] = "standard"
    at.run()
    assert not at.exception
    assert [s for s in at.sidebar.slider if s.label.startswith("Score-Schwelle")][0].value == 0.55 and [s for s in at.sidebar.slider if s.label.startswith("Schwelle: χ²")][0].value == 0.99
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert [s for s in at.sidebar.slider if s.label.startswith("Angenommener")][0].value == 25


def test_sweep_options_follow_the_threshold_kind_and_the_gap_kind():
    at = _run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["cutoff"] in options and SWEEP_LABELS["quantile"] in options and SWEEP_LABELS["share"] not in options
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["share"] in options and SWEEP_LABELS["cutoff"] not in options and SWEEP_LABELS["quantile"] not in options
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert not ({SWEEP_LABELS["strength"], SWEEP_LABELS["n_modes"]} & set(options)) and SWEEP_LABELS["contamination"] in options


def test_strength_slider_is_hidden_for_gap_anomalies_and_its_value_is_kept():
    at = _run(lambda a: (a.session_state.__setitem__("strength_slider", 9.0)))
    assert "Abstand der Anomalien (Faktor-σ)" in _labels(at)
    at.session_state["n_modes_slider"] = 2
    at.run()
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and "Abstand der Anomalien (Faktor-σ)" not in _labels(at)
    at.session_state["kind_select"] = "scattered"
    at.run()
    assert not at.exception and [s for s in at.sidebar.slider if s.label.startswith("Abstand")][0].value == 9.0


def test_gap_kind_needs_two_modes_and_falls_back_when_modes_drop():
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    at.session_state["n_modes_slider"] = 1
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] not in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    assert at.session_state["kind_select"] == "scattered"


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["eif_step"] = 4
    at.run()
    assert not at.exception and at.session_state["eif_step"] == 4
    at.session_state["contamination_slider"] = 20
    at.run()
    assert not at.exception and at.session_state["eif_step"] == 1


def test_experiments_run_on_demand_and_sweeps_run():
    at = _run()
    for parameter in ("extension", "n_noise", "psi", "cutoff", "contamination"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        [b for b in at.button if b.key == "sweep_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    for key in ("rotation_start", "ghost_start", "modes_start", "dimension_start", "threshold_start", "masking_start", "cost_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert all(at.session_state[k] for k in ("rotation_on", "ghost_on", "modes_on", "dimension_on", "threshold_on", "masking_on", "cost_on"))


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "eif_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_unique_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    parts = source.split("plotly_chart(")[1:]
    keys = [re.search(r'key="([a-z_]+)"', part.split("plotly_chart(")[0]) for part in parts]
    assert len(parts) >= 20 and all(keys)
    names = [k.group(1) for k in keys]
    assert len(set(names)) == len(names)


def test_app_has_no_dead_file_links_and_the_verbatim_footer():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.findall(r"\]\([a-z_]+\.py\)", source)
    assert "https://sebastianhanisch.net/kontakt.html" in source and "Interesse an einer maßgeschneiderten Lösung für" in source


def test_extension_options_follow_the_number_of_features_and_the_state_is_kept_valid():
    at = _run()
    label = "Erweiterungsgrad ℓ"
    assert len([s for s in at.sidebar.selectbox if s.label == label][0].options) == 5                  # 12 Merkmale: 0, 1, 2, halb (5), voll (11)
    at.session_state["extension_select"] = "half"
    at.session_state["p_slider"] = 2
    at.run()
    assert not at.exception
    assert len([s for s in at.sidebar.selectbox if s.label == label][0].options) == 2                   # 2 Merkmale: nur 0 und 1
    assert at.session_state["extension_select"] == "full"                                             # "halb" gibt es nicht mehr -> zurück auf den Standard
    at.session_state["n_noise_slider"] = 40
    at.run()
    assert not at.exception and len([s for s in at.sidebar.selectbox if s.label == label][0].options) == 5


@pytest.mark.parametrize("ext", ["axis", "one", "full"])
def test_every_extension_renders_the_ghost_step(ext):
    def setup(at):
        at.session_state["extension_select"] = ext
        at.session_state["eif_step"] = 4
    _run(setup)
