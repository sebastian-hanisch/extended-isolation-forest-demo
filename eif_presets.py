"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe isf_presets.py in isolation-forest-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import eif_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "n_noise_slider": SettingSpec("nn", int, C.DEFAULT_N_NOISE, C.N_NOISE_MIN, C.N_NOISE_MAX),
    "n_modes_slider": SettingSpec("modes", int, C.DEFAULT_N_MODES, C.N_MODES_MIN, C.N_MODES_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "contamination_slider": SettingSpec("cont", int, C.DEFAULT_CONTAMINATION, C.CONTAMINATION_MIN, C.CONTAMINATION_MAX),
    "kind_select": SettingSpec("kind", _choice(C.KINDS), C.DEFAULT_KIND),
    "strength_slider": SettingSpec("str", float, C.DEFAULT_STRENGTH, C.STRENGTH_MIN, C.STRENGTH_MAX),
    "extension_select": SettingSpec("ext", _choice(C.EXTENSION_KINDS), C.DEFAULT_EXTENSION),
    "trees_slider": SettingSpec("trees", int, C.DEFAULT_TREES, C.TREES_MIN, C.TREES_MAX),
    "psi_slider": SettingSpec("psi", int, C.DEFAULT_PSI, C.PSI_MIN, C.PSI_MAX),
    "threshold_kind_select": SettingSpec("thr", _choice(C.THRESHOLD_KINDS), C.DEFAULT_THRESHOLD_KIND),
    "cutoff_slider": SettingSpec("cut", float, C.DEFAULT_CUTOFF, C.CUTOFF_MIN, C.CUTOFF_MAX),
    "quantile_slider": SettingSpec("q", float, C.DEFAULT_QUANTILE, C.QUANTILE_MIN, C.QUANTILE_MAX),
    "share_slider": SettingSpec("share", int, C.DEFAULT_SHARE, C.SHARE_MIN, C.SHARE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n": "n_tours_slider", "p": "p_slider", "n_noise": "n_noise_slider", "n_modes": "n_modes_slider", "curvature": "curvature_slider", "noise": "noise_slider",
               "contamination": "contamination_slider", "kind": "kind_select", "strength": "strength_slider", "extension": "extension_select", "trees": "trees_slider", "psi": "psi_slider", "threshold_kind": "threshold_kind_select",
               "cutoff": "cutoff_slider", "quantile": "quantile_slider", "share": "share_slider", "seed": "seed_input"}
# Regler, die je nach Einstellung ausgeblendet sind (Abstand bei "Lücke"; Score-Schwelle und chi²-Quantil beim erwarteten Anteil; angenommener Anteil bei der Standardschwelle): der Wert bleibt hier erhalten
KEPT = {"strength_slider": "_strength_kept", "cutoff_slider": "_cutoff_kept", "quantile_slider": "_quantile_kept", "share_slider": "_share_kept"}
PRESET_KEPT = {"strength": "_strength_kept", "cutoff": "_cutoff_kept", "quantile": "_quantile_kept", "share": "_share_kept"}


def kind_options(n_modes):
    """Die Art 'in der Lücke' gibt es erst ab zwei Betriebsarten."""
    return tuple(k for k in C.KINDS if k != "gap" or n_modes >= 2)


def extension_options(p_total):
    """Erweiterungsgrade, die bei `p_total` Merkmalen verschieden sind (gleiche Grade doppelt anzubieten wäre ein toter Unterschied): je Grad zählt die erste Bezeichnung in der Reihenfolge voll, 0, 1, 2, halb."""
    levels = {"axis": 0, "one": 1, "two": 2, "half": (p_total - 1) // 2, "full": p_total - 1}
    keep, seen = [], set()
    for kind in ("full", "axis", "one", "two", "half"):                         # "voll" und "0" zuerst, damit sie nie wegfallen
        lvl = min(levels[kind], p_total - 1)
        if lvl not in seen:
            seen.add(lvl)
            keep.append(kind)
    return tuple(k for k in C.EXTENSION_KINDS if k in keep)


def psi_max(n_tours):
    """Die Unterstichprobe kann nicht größer sein als die Tourenzahl."""
    return int(max(C.PSI_MIN + 4, min(C.PSI_MAX, n_tours)))


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; ausgeblendete Regler kehren zum zuletzt gewählten Wert zurück; eine ungültige Art wird zurückgesetzt, die Unterstichprobe auf die Tourenzahl begrenzt."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default
    if st.session_state["kind_select"] not in kind_options(st.session_state["n_modes_slider"]):
        st.session_state["kind_select"] = C.DEFAULT_KIND
    if st.session_state["extension_select"] not in extension_options(st.session_state["p_slider"] + st.session_state["n_noise_slider"]):
        st.session_state["extension_select"] = C.DEFAULT_EXTENSION
    st.session_state["psi_slider"] = min(st.session_state["psi_slider"], psi_max(st.session_state["n_tours_slider"]))


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("curvature_slider", 4), ("noise_slider", 20), ("strength_slider", 2), ("cutoff_slider", 100)):
        st.session_state[key] = round(st.session_state.get(key, SETTING_SPECS[key].default) * step) / step
    st.session_state["quantile_slider"] = round(st.session_state.get("quantile_slider", C.DEFAULT_QUANTILE) * 1000) / 1000
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    for key, kept in PRESET_KEPT.items():
        st.session_state[kept] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
