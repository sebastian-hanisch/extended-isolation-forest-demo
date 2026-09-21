"""Extended Isolation Forest (numpy von Grund auf, nach Hariri, Kind und Brunner): wie der Isolation Forest, aber jeder Schnitt ist eine zufällige **Hyperebene** statt eines achsenparallelen Schnitts.
Je Knoten: ein zufälliger Normalenvektor n (Gauß'sche Komponenten; nur ℓ + 1 davon sind ungleich null, ℓ = Erweiterungsgrad) und ein zufälliger Durchstoßpunkt q im Wertebereich des Knotens; geteilt wird nach (x - q) · n <= 0.
ℓ = 0 ist der normale Isolation Forest (ein Merkmal, ein Schnittpunkt), ℓ = p - 1 sind volle schräge Schnitte. Pfadlänge und Anomalie-Wert sind wie im Isolation Forest."""

import math
from dataclasses import dataclass

import numpy as np

from eif_isolation_forest import c_factor, flag_top, height_limit, score_convergence, score_from_paths  # noqa: F401  (Pfadlänge, Score und Schwelle unverändert)


@dataclass(frozen=True)
class Tree:
    normal: np.ndarray          # (Knoten, p) Normalenvektor des Schnitts (Blatt: null)
    point: np.ndarray           # (Knoten, p) Durchstoßpunkt
    left: np.ndarray
    right: np.ndarray
    size: np.ndarray            # Zahl der Unterstichproben-Punkte im Knoten
    depth: np.ndarray           # Tiefe des Knotens
    split: np.ndarray           # True bei inneren Knoten


@dataclass(frozen=True)
class Forest:
    trees: tuple
    psi: int
    subsets: tuple
    max_depth: int
    extension: int              # Erweiterungsgrad ℓ (0 = Isolation Forest)


def _nonzero_mask(valid, extension, rng):
    """Für jeden Knoten ℓ + 1 zufällige Komponenten (bevorzugt nicht konstante Merkmale) als Maske."""
    m, p = valid.shape
    pick = rng.random((m, p))
    pick[~valid] = -1.0
    rank = np.argsort(np.argsort(-pick, axis=1), axis=1)                       # 0 = größter Wert
    return (rank < min(extension + 1, p)) & valid


def build_tree(S, rng, max_depth, extension=0):
    """Ein erweiterter Isolationsbaum über die Unterstichprobe S (psi x p), Ebene für Ebene: n mit ℓ + 1 Gauß'schen Komponenten (die übrigen null), q gleichverteilt im Wertebereich des Knotens;
    nach links geht, was (x - q) · n <= 0 erfüllt. Ein Knoten mit gleichen Punkten bleibt Blatt; trifft die Ebene keinen Punkt einer Seite, entsteht dort ein leeres Blatt (Größe 0, c = 0) und der Rest läuft eine Ebene tiefer weiter (wie im Original)."""
    psi, p = S.shape
    extension = int(min(max(extension, 0), p - 1))
    normal, point, left, right, size, depth, split = [np.zeros(p)], [np.zeros(p)], [-1], [-1], [psi], [0], [False]
    node_of = np.zeros(psi, dtype=int)
    frontier = [0]
    for level in range(max_depth):
        active = [k for k in frontier if size[k] > 1]
        if not active:
            break
        m = len(active)
        local = np.full(len(size), -1, dtype=int)
        local[active] = np.arange(m)
        rows = np.flatnonzero(local[node_of] >= 0)
        loc = local[node_of[rows]]
        mins = np.full((m, p), np.inf)
        maxs = np.full((m, p), -np.inf)
        np.minimum.at(mins, loc, S[rows])
        np.maximum.at(maxs, loc, S[rows])
        valid = maxs > mins
        mask = _nonzero_mask(valid, extension, rng)
        n_vec = rng.standard_normal((m, p)) * mask
        q = mins + rng.random((m, p)) * (maxs - mins)
        go_left = ((S[rows] - q[loc]) * n_vec[loc]).sum(axis=1) <= 0
        n_left = np.bincount(loc[go_left], minlength=m)
        n_right = np.bincount(loc[~go_left], minlength=m)
        splittable = valid.any(axis=1)                                            # eine Ebene, die alle Punkte auf eine Seite legt, verbraucht eine Ebene (leeres Kind, Größe 0), wie im Original
        lid_of = np.full(m, -1, dtype=int)
        rid_of = np.full(m, -1, dtype=int)
        next_frontier = []
        for j, node in enumerate(active):
            if not splittable[j]:
                continue
            lid, rid = len(size), len(size) + 1
            for cnt in (n_left[j], n_right[j]):
                normal.append(np.zeros(p))
                point.append(np.zeros(p))
                left.append(-1)
                right.append(-1)
                size.append(int(cnt))
                depth.append(level + 1)
                split.append(False)
            normal[node], point[node], left[node], right[node], split[node] = n_vec[j], q[j], lid, rid, True
            lid_of[j], rid_of[j] = lid, rid
            next_frontier += [lid, rid]
        moved = lid_of[loc] >= 0
        node_of[rows[moved]] = np.where(go_left[moved], lid_of[loc[moved]], rid_of[loc[moved]])
        frontier = next_frontier
    return Tree(np.array(normal), np.array(point), np.array(left), np.array(right), np.array(size), np.array(depth), np.array(split))


def fit_forest(X, n_trees=100, psi=256, seed=0, extension=0):
    """Wald aus `n_trees` erweiterten Bäumen, je auf einer Unterstichprobe der Größe min(psi, n) ohne Zurücklegen (Seed getrennt von dem der Aufnahme und dem des Isolation Forest)."""
    n, p = X.shape
    psi = int(min(psi, n))
    extension = int(min(max(extension, 0), p - 1))
    rng = np.random.default_rng([seed, 4243])
    limit = height_limit(psi)
    trees, subsets = [], []
    for _ in range(n_trees):
        idx = np.sort(rng.choice(n, size=psi, replace=False))
        trees.append(build_tree(X[idx], rng, limit, extension))
        subsets.append(idx)
    return Forest(tuple(trees), psi, tuple(subsets), limit, extension)


def tree_path_length(tree, X):
    """Pfadlänge jedes Punkts in einem Baum: Tiefe des Blatts plus c(Blattgröße)."""
    n = len(X)
    node = np.zeros(n, dtype=int)
    depth = np.zeros(n)
    while True:
        internal = tree.split[node]
        if not internal.any():
            break
        proj = ((X - tree.point[node]) * tree.normal[node]).sum(axis=1)
        node = np.where(internal, np.where(proj <= 0, tree.left[node], tree.right[node]), node)
        depth += internal
    return depth + c_factor(tree.size[node])


def path_lengths(forest, X):
    """(Bäume, n) Pfadlängen aller Punkte in allen Bäumen."""
    return np.stack([tree_path_length(t, X) for t in forest.trees])


def score(forest, X):
    return score_from_paths(path_lengths(forest, X), forest.psi)


# --- Darstellung ------------------------------------------------------------------------------------------------------------------


def point_path(tree, x):
    """Weg eines Punkts durch einen Baum: [(Normalenvektor, Durchstoßpunkt, ging nach links)] bis zum Blatt, dazu die Blattgröße."""
    node, steps = 0, []
    while tree.split[node]:
        went_left = bool(((x - tree.point[node]) * tree.normal[node]).sum() <= 0)
        steps.append((tree.normal[node], tree.point[node], went_left))
        node = tree.left[node] if went_left else tree.right[node]
    return steps, int(tree.size[node])


def _clip(polygon, n, q, keep_left):
    """Konvexes Polygon an der Geraden n · (x - q) = 0 abschneiden; `keep_left`: der Teil mit n · (x - q) <= 0."""
    sign = 1.0 if keep_left else -1.0
    out = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        da, db = sign * float(np.dot(n, a - q)), sign * float(np.dot(n, b - q))
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            out.append(a + (b - a) * (da / (da - db)))
    return out


def tree_segments(tree, bounds):
    """Schnittlinien eines Baums über zwei Merkmale als Strecken (x0, y0, x1, y1, Tiefe): je innerem Knoten die Schnittgerade im (konvexen) Gebiet des Knotens - zum Zeichnen der Zerlegung der Ebene."""
    xmin, xmax, ymin, ymax = bounds
    root = [np.array([xmin, ymin]), np.array([xmax, ymin]), np.array([xmax, ymax]), np.array([xmin, ymax])]
    segments = []

    def walk(node, polygon):
        if not tree.split[node] or len(polygon) < 3:
            return
        n, q = tree.normal[node][:2], tree.point[node][:2]
        # Schnittpunkte der Geraden mit dem Polygonrand
        hits = []
        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i + 1) % len(polygon)]
            da, db = float(np.dot(n, a - q)), float(np.dot(n, b - q))
            if (da < 0 < db) or (db < 0 < da):
                hits.append(a + (b - a) * (da / (da - db)))
        if len(hits) >= 2:
            segments.append((hits[0][0], hits[0][1], hits[1][0], hits[1][1], int(tree.depth[node])))
        walk(tree.left[node], _clip(polygon, n, q, True))
        walk(tree.right[node], _clip(polygon, n, q, False))

    walk(0, root)
    return segments


def normal_angle(normal):
    """Winkel der Schnittrichtung in der Ebene (Grad, 0 = achsenparallel zur x-Achse)."""
    return math.degrees(math.atan2(normal[1], normal[0]))
