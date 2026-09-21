# Extended Isolation Forest – schräge Schnitte statt Geister-Regionen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-extended-isolation-forest-demo.streamlit.app/)**

Drittes Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Extended Isolation Forest** – an einem wachsenden Beispiel, gegen den
[isolation-forest-demo](../isolation-forest-demo) und die beiden Schätzer der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo): klassisch und robust per MCD). Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo)
(Szenario, Isolation Forest und Wurzel-Schätzer wortgleich übernommen, per Test gegen eingefrorene Werte geprüft); neu ist die **Drehung der Merkmale** (dabei ohne messbare Wirkung, siehe unten).

**Einordnung in die Reihe (die Kanten des Graphen):** der Isolation Forest schneidet mit zufälligen **achsenparallelen** Schnitten – das hinterlässt in der Bewertung Bänder entlang der Achsen ("Geister-Regionen", im Vorgänger als Score-Anisotropie 0.06–0.08 gemessen).
Der Extended Isolation Forest ersetzt jeden Schnitt durch eine **zufällige Hyperebene** (zufälliger Normalenvektor, zufälliger Punkt im Wertebereich des Knotens); der **Erweiterungsgrad ℓ** legt fest, wie viele Merkmale eingehen (ℓ = 0 ist der Isolation Forest).
Dieses Stück schließt den Ast der Zufallsbäume; die Linie hat **keinen Konvergenzpunkt**.
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ECOD                       (Kontrast: verteilungsfrei)                       [nicht gebaut]
  ├─ LOF → Feature Bagging      (lokale Dichte; Ensembles gegen viele Merkmale)   [nicht gebaut]
  ├─ One-Class SVM → Deep SVDD  (gelernte Grenze)                                 [nicht gebaut]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)        [dieses Stück: Ast geschlossen]
  └─ Autoencoder                (Rekonstruktionsfehler)                           [nicht gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; 100 Bäume, ψ = 256, ℓ = 11 (voll), Score-Schwelle 0.5 bzw. χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall | ✅ Rangfolge bei beiden Wäldern perfekt (AUC 1.00); F1 an der Standardschwelle **0.96** (Isolation Forest, Fehlalarmrate 0.9 %) und **0.98** (Extended IF, 0.1 %) gegen 0.84 (robust) und 0.57 (klassisch) |
| Geister-Regionen | ✅ Anisotropie (Ecken minus Korridore) bei 1 σ Abstand **0.081 → 0.029**, bei drei Betriebsarten und 2 σ **0.061 → 0.001** – die Bänder schrumpfen auf ein Drittel bis fast null. ❌ *Aber:* an der Erkennung ändert das im Szenario nichts (AUC beider 1.00) |
| Drehung der Merkmale | ❌ **Kein Effekt.** Je zwei Merkmale um bis zu 45° gedreht: AUC und F1 beider Wälder unverändert; die Scores der normalen Touren korrelieren mit denen der ungedrehten Daten (Rang) zu 0.93–0.94 – so viel wie zwischen zwei Wäldern mit verschiedenem Seed auf denselben Daten (0.94 / 0.92) |
| Kleine Stichproben und Unterstichproben | ✅ **Der eigentliche Gewinn**: bei der Schwelle 0.5 F1 **0.60 / 0.67 / 0.78 / 0.89** (Isolation Forest) gegen **1.00 / 0.97 / 0.94 / 0.97** (Extended IF) bei 20 / 30 / 50 / 100 Touren; ψ = 16: **0.56 gegen 0.92** (Fehlalarmrate 0.18 gegen 0.02). 20 Touren × 30 Merkmale: 0.61 gegen 1.00. Der Score des Extended IF hängt weniger an n und ψ (mittlerer Score der Normalen 0.387 → 0.369 gegen 0.435 → 0.377 bei 20 → 600 Touren) |
| Erweiterungsgrad ℓ | ✅ AUC 1.00 bei jedem Grad; F1 0.96 / 0.97 / 0.97 / 0.98 / 0.98 und Fehlalarmrate 0.008 / 0.004 / 0.003 / 0.001 / 0.001 bei ℓ = 0 / 1 / 2 / 5 / 11 – der volle Grad ist der beste, kein Zwischenwert schlägt ihn |
| Rauschmerkmale | ❌ **Kein Vorteil**: bei 40 Rauschmerkmalen AUC 0.99 bei beiden, Recall an der Schwelle 0.39 (Isolation Forest) gegen 0.41 (Extended IF), F1 0.56 gegen 0.58 (robust: AUC 0.88, Recall 0.81 mit 31 % Fehlalarmen) |
| Dichte Gruppe abseits | ❌ AUC 0.95 / 0.85 / 0.70 gegen 0.97 / 0.85 / 0.57 (Isolation Forest / Extended IF) bei 10 / 20 / 30 %; an der Schwelle bei 20 % **F1 0.44 gegen 0.13** (Recall 0.44 gegen 0.09); robust 1.00 bis 25 %, dann 0.49 |
| Anomalien in der Lücke | ❌ zwei Betriebsarten, 10 %: AUC **0.54** (Isolation Forest), **0.38** (Extended IF), 0.40 (klassisch und robust) – die schrägen Schnitte umschließen die dichte Gruppe zwischen den Betriebsarten noch besser; bei 2 %: 0.81 gegen 0.56; drei Betriebsarten 0.27 gegen 0.31 |
| Viele Anomalien | ❌ bei 45 %: F1 0.81 (Isolation Forest) gegen 0.42 (Extended IF), Recall 0.68 gegen 0.27; wenige Anomalien dagegen ✅: bei 2 % F1 0.42 gegen 0.69 |
| Schwelle | ⚠️ Score-Schwelle 0.45 / 0.5 / 0.55 / 0.6 / 0.65: F1 0.82 / 0.96 / **0.97** / 0.83 / 0.55 (Isolation Forest) gegen **0.92 / 0.98** / 0.85 / 0.40 / 0.04 (Extended IF): die Scores des Extended IF liegen niedriger, das brauchbare Fenster liegt bei 0.45–0.5. Ein falsch angenommener Anteil (½× / 2×) senkt F1 bei allen Detektoren gleich (0.97 / 0.99 → 0.67) |
| Bäume | ✅ AUC schon bei 10 Bäumen 1.00; F1 des Extended IF an der Schwelle 0.95 / 0.98 / 0.97 bei 10 / 25 / 50 Bäumen (Recall bei 10 Bäumen 0.91) |
| Einheiten | ⚠️ Der Extended IF ist **nicht skaleninvariant**: auf Rohdaten (Meter, Minuten, ...) AUC 0.93, F1 0.65, Recall 0.73 statt 1.00, 0.98 und 0.97 – die Kennzahlen müssen standardisiert werden (der Isolation Forest ist skaleninvariant) |
| Rechenzeit | ✅ Der Extended IF ist etwas langsamer, aber nicht um Größenordnungen: Faktor 1.1–1.3 (0.09 / 0.18 / 0.29 s gegen 0.07 / 0.14 / 0.27 s bei 2 / 12 / 30 Merkmalen; rechnerabhängig) |

## Was die Demo zeigt

1. **Extended Isolation Forest in Aktion** (Schritt-Slider + Abspielen): **Touren** → **Schräge Schnitte** (zwei Isolationsbäume auf denselben Punkten in der Ebene der zwei größten Streuungsrichtungen: achsenparallel links, Geraden rechts; der Weg einer Sonderfahrt und einer normalen Tour;
   Pfadlängen in allen Bäumen der echten Wälder) → **Pfadlängen und Score** (mittlere Pfadlänge je Tour, Score nach k Bäumen, Histogramm mit Schwelle) → **Geister-Regionen** (Score-Karten Isolation Forest | Extended IF über die ersten zwei Merkmale mit der Ellipse der Wurzel) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Wälder gefunden haben – gegen die Wurzel:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → dichte Gruppe → Wurzel besser → Extended IF besser → Isolation Forest besser → falsche Schwelle bei guter Rangfolge → gleichauf), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über Erweiterungsgrad, Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, Bäume, Unterstichprobe, Score-Schwelle, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** Drehung der Merkmale, Geister-Regionen (Anisotropie je Betriebsart), Wurzel-Schwächen (Betriebsarten × Art der Anomalien), Rauschmerkmale und Tourenzahl × Merkmalszahl, Schwelle (Score-Schwelle, falscher Anteil, mittlerer Score je Tourenzahl), Masking (dichte Gruppe 2–45 %, Einfluss von ψ), Kosten und Einheiten (Rechenzeit, Rohdaten gegen standardisiert).
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Geister-Regionen, Score als Schwelle, irrelevante Merkmale, dichte Anomaliegruppen und Lücke, Einheiten, Baumzahl).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten), Abstand (bei "Lücke" ausgeblendet, Wert bleibt erhalten),
**Erweiterungsgrad ℓ** (0, 1, 2, halb, voll – nur Grade, die bei der Merkmalszahl verschieden sind), Bäume (10–500), Unterstichprobe ψ (bis zur Tourenzahl), **Schwelle** (Standard: Score-Schwelle und χ²-Quantil, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC IF | AUC EIF | F1 IF | F1 EIF | AUC robust | Urteil |
|---|---|---|---|---|---|---|
| Standardfall | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | gleich gut |
| Kleine Unterstichprobe (ψ = 16) | 1.00 | 1.00 | 0.69 | 0.98 | 1.00 | Extended IF besser (Schwelle) |
| Wenige Touren, viele Merkmale (20 × 30) | 1.00 | 1.00 | 0.57 | 1.00 | 0.64 | Extended IF besser (Schwelle) |
| Dichte Gruppe abseits (20 %) | 0.91 | 0.86 | 0.64 | 0.19 | 1.00 | dichte Gruppe: Wurzel besser |
| Anomalien in der Lücke | 0.63 | 0.54 | 0.05 | 0.04 | 0.35 | Anomalien in der Lücke |
| Wenige Anomalien in der Lücke (2 %) | 0.84 | 0.58 | 0.10 | 0.00 | 0.34 | Isolation Forest besser |

## Modell und Verfahren

- **Szenario** (`eif_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale). Neu: **`rotation`** – je zwei aufeinanderfolgende Merkmale werden
  in standardisierten Einheiten um `rotation · 45°` gedreht (orthogonal, affin: Mahalanobis-Abstände und MCD ändern sich nicht); `rotation = 0` ist bit-identisch zum Vorgänger (per Test).
- **Extended Isolation Forest** (`eif_algorithm.py`, numpy von Grund auf): Unterstichprobe ψ ohne Zurücklegen; Baum Ebene für Ebene, je Knoten ein Normalenvektor mit ℓ + 1 Gauß'schen Komponenten (bevorzugt auf nicht konstanten Merkmalen, die übrigen null) und ein Punkt gleichverteilt im Wertebereich des Knotens;
  links geht, was (x − q) · n ≤ 0 erfüllt. Legt die Ebene alle Punkte auf eine Seite, verbraucht sie eine Ebene (leeres Blatt, Größe 0, c = 0) – wie im Original. Höhenlimit ⌈log₂ ψ⌉; Pfadlänge und Score wie beim Isolation Forest (c(n)-Normierung). Der Wald rechnet auf **standardisierten** Kennzahlen.
  Für die Darstellung wird ein Baum über die Ebene in Polygone zerlegt (`tree_segments`).
- **Isolation Forest** (`eif_isolation_forest.py`) und **Wurzel** (`eif_ee_algorithm.py`: χ² ohne scipy, klassisch, FastMCD): wortgleich aus den Vorgängern.
- **Auswertung** (`eif_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; **Score-Anisotropie** (Ecken minus Korridore bei gleichem standardisiertem Abstand zum nächsten Datenpunkt ±0.15, Wälder über die ersten zwei Merkmale, Extended IF mit ℓ = 1);
  Sweeps, Experiment-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Ein Fehler in der ersten Fassung – und was er verfälscht hätte:** die erste Version machte aus einem Knoten, dessen Hyperebene alle Punkte auf eine Seite legt, ein **Blatt** (bei ℓ ≥ 1 in etwa jedem achten Baum schon die Wurzel – ein Baum, der nichts isoliert). Das ließ die Scores der normalen Touren zu hoch liegen und erzeugte
  Ergebnisse, die schön aussahen und falsch waren: "der Extended IF ist bei Rauschmerkmalen besser (Recall 0.86 gegen 0.39)" und "seine Scores sind zusammengedrückt, die Schwelle 0.5 trifft ihn schlechter". Nach der Korrektur (die Ebene verbraucht eine Ebene, der Rest läuft weiter, wie im Original) wurde **alles neu gemessen**; beide Aussagen sind verschwunden bzw. haben sich umgekehrt.
  Ein Regressionstest (`test_a_plane_that_misses_all_points_uses_up_a_level_instead_of_ending_the_tree`) hält die Korrektur fest.
- **Vorab-Vermutungen, die nicht stimmten:** (1) "Mit zunehmender Drehung fällt der Isolation Forest, der Extended IF nicht" – **kein Effekt** (AUC 1.00, F1 unverändert, Scores ändern sich nicht mehr als durch einen anderen Wald-Seed); die Kennzahlen des Szenarios sind über die zwei Faktoren ohnehin schräg zueinander, es gibt keine achsenparallele Struktur, die eine Drehung zerstören könnte.
  Die Drehung ist deshalb kein Regler der Seitenleiste, sondern ein Experiment auf Abruf. (2) "Der Extended IF ist um den Faktor 2–5 langsamer" – gemessen 1.1–1.3. (3) "Schräge Schnitte verschlimmern (oder ändern nichts bei) Rauschmerkmale(n)" – sie ändern nichts (Recall 0.41 gegen 0.39). (4) "ℓ = p − 1 ist nicht immer am besten" – doch: die AUC hängt nicht an ℓ,
  F1 und Fehlalarmrate werden mit ℓ leicht besser. (5) "Der Vorteil zeigt sich in der Rangfolge" – die AUC ist im Standardfall schon bei 1.00; der gemessene Vorteil liegt **an der Schwelle bei kleinen Stichproben und Unterstichproben**, nicht in der Rangfolge.
- **Geister-Regionen sind messbar und harmlos für die Erkennung:** der Extended IF verkleinert sie auf ein Drittel bis fast null, aber die AUC ist bei beiden 1.00. Der Gewinn liegt dort, wo Anomalien wirklich in den Bändern liegen – im Szenario nicht der Fall.
- **Der Extended IF ist nicht skaleninvariant:** ohne Standardisieren fällt seine AUC auf 0.93 und der F1 auf 0.65; der Isolation Forest braucht das nicht.
- **Schräge Schnitte verstärken Masking:** dichte Gruppen und die Lücke zwischen Betriebsarten werden noch besser umschlossen (AUC 0.38 gegen 0.54 in der Lücke; F1 0.13 gegen 0.44 bei 20 % dichter Gruppe; Recall 0.27 gegen 0.68 bei 45 % Anomalien).
- **Der Score ist auch hier keine kalibrierte Wahrscheinlichkeit:** das Fenster für die Score-Schwelle des Extended IF liegt niedriger und reicht weniger weit nach oben (F1 0.40 bei 0.6, Isolation Forest 0.83).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Anomalien liegen im Faktorraum weit draußen (Abstand 3–12 σ), was allen Verfahren entgegenkommt. Literatur nur mit Namen: Liu, Ting und Zhou (Isolation Forest); Hariri, Kind und Brunner (Extended Isolation Forest).

## Verifikation

- Extended Isolation Forest: Baumstruktur (Knotengrößen = Summe der Kinder, Blattgrößen = wirklich dort landende Punkte, höchstens ℓ + 1 Komponenten je Schnitt, Grad auf p − 1 begrenzt, Höhenlimit, gleiche Punkte in einem Blatt, konstante Merkmale ignoriert); Handinstanz einer Hyperebene; **ℓ = 0 gegen den Isolation Forest und gegen `sklearn.ensemble.IsolationForest`** (Rangkorrelation, AUC);
  Skalenverhalten (der Isolation Forest ist invariant, der Extended IF nicht) und Verhalten unter Drehung; Determinismus; Regressionstest für Ebenen, die alle Punkte auf eine Seite legen.
- Übernommene Bausteine: Isolation Forest (Struktur, c(n), Pfadlänge = Weg, sklearn-Kreuzprüfung) und Wurzel-Schätzer (χ² gegen scipy, MCD gegen `sklearn.covariance.MinCovDet`). Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `rotation = 0` bit-identisch, Drehung orthogonal und ohne Wirkung auf Mahalanobis-Abstände, Rauschspalten und Anomalien.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Drehungs-/Geister-/Betriebsarten-/Masking-/Schwellen-/Dimensions-/Kostentabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; Rechenzeiten nur als Verhältnis);
  alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, jeder Erweiterungsgrad im Geister-Schritt, ψ-Grenze folgt der Tourenzahl, Erweiterungsgrade folgen der Merkmalszahl, ausgeblendete Regler behalten ihre Werte, Art "Lücke" nur ab zwei Betriebsarten, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Drehung, Geister, Wurzel-Schwächen, Dimension, Schwelle, Masking, Kosten), 🚧 Grenzen, Mathe |
| `eif_algorithm.py` | Hyperebenen-Baum, Wald, Pfadlängen, Score, Zerlegung der Ebene für die Darstellung |
| `eif_isolation_forest.py`, `eif_ee_algorithm.py` | Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD (wortgleich aus den Vorgängern) |
| `eif_scenario.py`, `eif_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen und Drehung; Konstanten, Presets |
| `eif_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Anisotropie, Urteil |
| `eif_presets.py`, `eif_visualization.py` | Permalink/Presets (ausgeblendete Regler, ψ- und ℓ-Grenze), Plotly-Figuren (achsengesperrt) |
| `tests/` | Wald (Handinstanzen, sklearn-Kreuzprüfung, Invarianzen), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
