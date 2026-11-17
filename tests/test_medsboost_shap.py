"""MedsBoost.shap_values against brute-force Shapley values computed from the saved model file.

The reference reads a model saved with node statistics (format version 2), follows MEDSBoost's binning (a
present value falls in the first bin whose upper bound is at least the value; an absent one in the column's
zero bin) and computes, for every subset S of the used columns, the path-dependent expectation E[f | x_S]: at
a split on a column in S the row's branch is taken, at any other split both branches are averaged by their
training row counts. The Shapley values of these set functions are exactly what path-dependent TreeSHAP
computes, so the native values must match them to rounding.
"""

import itertools
import math
import tempfile
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from MEDS_tabular_automl import medsboost
from MEDS_tabular_automl.medsboost import MedsBoost

pytestmark = pytest.mark.skipif(not medsboost.is_available(), reason="medsboost is not built.")


def read_model(path: Path):
    lines = iter(path.read_text().splitlines())
    assert next(lines) == "medsboost 2"
    next(lines)  # num_columns
    init = float(next(lines).split()[1])
    features = {}
    for _ in range(int(next(lines).split()[1])):
        fields = next(lines).split()
        upper = [np.float32(np.inf) if u == "inf" else np.float32(float(u)) for u in fields[3:]]
        features[int(fields[0])] = (int(fields[2]), upper)
    trees = []
    for _ in range(int(next(lines).split()[1])):
        n_nodes, n_leaves = (int(v) for v in next(lines).split()[1:])
        nodes = []
        for _ in range(n_nodes):
            feat, thr, left, right, _gain, _hess, count = next(lines).split()
            nodes.append((int(feat), int(thr), int(left), int(right), int(count)))
        values = [float(v) for v in next(lines).split()]
        next(lines)  # leaf hessian sums
        counts = [int(v) for v in next(lines).split()]
        assert len(values) == len(counts) == n_leaves
        trees.append((nodes, values, counts))
    return init, features, trees


def reference_shap(path: Path, row: dict[int, float]):
    init, features, trees = read_model(path)
    used = sorted({feat for nodes, values, _ in trees if len(values) > 1 for feat, *_ in nodes})

    def bin_of(feat):
        zero_bin, upper = features[feat]
        return bisect_left(upper, np.float32(row[feat])) if feat in row else zero_bin

    bins = {feat: bin_of(feat) for feat in used}

    def expect(tree, node, subset):
        nodes, values, counts = tree
        if node < 0:
            return values[~node]
        feat, thr, left, right, count = nodes[node]
        if feat in subset:
            return expect(tree, left if bins[feat] <= thr else right, subset)

        def weight(child):
            return nodes[child][4] if child >= 0 else counts[~child]

        return (
            weight(left) * expect(tree, left, subset) + weight(right) * expect(tree, right, subset)
        ) / count

    def value(subset):
        return init + sum(tree[1][0] if len(tree[1]) == 1 else expect(tree, 0, subset) for tree in trees)

    m = len(used)
    phi = {}
    for feat in used:
        others = [f for f in used if f != feat]
        total = 0.0
        for size in range(m):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for subset in itertools.combinations(others, size):
                total += weight * (value(set(subset) | {feat}) - value(set(subset)))
        phi[feat] = total
    return phi, value(set())


@pytest.mark.parametrize(
    "n_cols, num_leaves, rounds",
    [
        (4, 15, 12),  # few columns, many leaves: a column recurs along paths (unwinding the TreeSHAP path)
        (6, 7, 20),
    ],
)
def test_shap_values_match_brute_force_shapley(n_cols, num_leaves, rounds):
    X = sp.random(500, n_cols, density=0.5, format="csr", random_state=3, dtype=np.float32)
    X.data = np.round(X.data * 10) - 3  # negative, zero-crossing and repeated values, as in real features
    dense = X.toarray()
    y = ((dense[:, 0] > 1) ^ (dense[:, 1] < 0) | (dense[:, 2] > 4)).astype(np.float32)
    model = MedsBoost(
        threads=2, num_leaves=num_leaves, min_child_samples=5, max_rounds=rounds, early_stopping=1000, seed=7
    ).fit(X[:400], y[:400], X[400:], y[400:])
    rows = X[400:430]
    values, columns, expected = model.shap_values(rows)
    assert np.allclose(values.sum(1) + expected, model.predict_raw(rows), rtol=0, atol=1e-9)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.txt"
        model.save(path, save_stats=True)
        for r in range(rows.shape[0]):
            row = {int(c): float(v) for c, v in zip(rows[r].indices, rows[r].data, strict=True)}
            phi, ref_expected = reference_shap(path, row)
            assert sorted(phi) == columns.tolist()
            assert np.isclose(expected, ref_expected, rtol=0, atol=1e-9)
            assert np.allclose(values[r], [phi[c] for c in columns], rtol=0, atol=1e-9), r


def test_deep_trees_match_brute_force_shapley():
    """Trees 16 levels deep (63 leaves on 5 columns), where the path weights are divided down many times."""
    rng = np.random.default_rng(1)
    X = sp.random(4000, 5, density=0.7, format="csr", random_state=5, dtype=np.float32)
    X.data = np.round(X.data * 20) - 5
    d = X.toarray()
    y = ((np.sin(d[:, 0]) + 0.3 * d[:, 1] - 0.2 * d[:, 2] * d[:, 3] + rng.normal(0, 0.5, 4000)) > 0).astype(
        np.float32
    )
    model = MedsBoost(
        threads=2, num_leaves=63, min_child_samples=5, max_rounds=8, early_stopping=1000, seed=7
    ).fit(X[:3200], y[:3200], X[3200:], y[3200:])
    rows = X[3200:3220]
    values, columns, expected = model.shap_values(rows)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.txt"
        model.save(path, save_stats=True)
        _, _, trees = read_model(path)

        def depth(nodes, node):
            return 0 if node < 0 else 1 + max(depth(nodes, nodes[node][2]), depth(nodes, nodes[node][3]))

        assert max(depth(nodes, 0) for nodes, values_, _ in trees if len(values_) > 1) >= 12
        for r in range(rows.shape[0]):
            row = {int(c): float(v) for c, v in zip(rows[r].indices, rows[r].data, strict=True)}
            phi, ref_expected = reference_shap(path, row)
            assert np.isclose(expected, ref_expected, rtol=0, atol=1e-9)
            assert np.allclose(values[r], [phi[c] for c in columns], rtol=0, atol=1e-9), r


def test_very_deep_leaf_wise_trees_stay_exact():
    """On wide sparse data MEDSBoost's leaf-wise trees go 100+ levels deep.

    The original TreeSHAP path extension / unwinding loses all precision there (contributions off by ~1e13);
    values must still add up to the raw scores.
    """
    rng = np.random.default_rng(0)
    X = sp.random(12000, 20000, density=0.01, format="csc", random_state=1, dtype=np.float32)
    X.data = np.round(X.data * 30) + 1
    signal = np.asarray(X[:, :50].sum(1)).ravel()
    y = ((signal + rng.normal(0, 2, 12000)) > np.median(signal)).astype(np.float32)
    model = MedsBoost(threads=2, max_rounds=20, early_stopping=1000).fit(
        X[:11000], y[:11000], X[11000:], y[11000:]
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.txt"
        model.save(path, save_stats=True)
        _, _, trees = read_model(path)

    def depth(nodes, node):
        return 0 if node < 0 else 1 + max(depth(nodes, nodes[node][2]), depth(nodes, nodes[node][3]))

    assert max(depth(nodes, 0) for nodes, values_, _ in trees if len(values_) > 1) >= 60
    rows = X[11000:11300].tocsr()
    values, _, expected = model.shap_values(rows)
    assert np.allclose(values.sum(1) + expected, model.predict_raw(rows), rtol=0, atol=1e-9)


def test_rows_without_any_value_and_empty_input():
    X = sp.random(300, 5, density=0.4, format="csr", random_state=0, dtype=np.float32)
    y = (X[:, 0].toarray().ravel() > 0.5).astype(np.float32)
    model = MedsBoost(threads=2, num_leaves=7, min_child_samples=5, max_rounds=10).fit(
        X[:200], y[:200], X[200:], y[200:]
    )
    empty_rows = sp.csr_matrix((3, 5), dtype=np.float32)
    values, _, expected = model.shap_values(empty_rows)
    assert np.allclose(values.sum(1) + expected, model.predict_raw(empty_rows), rtol=0, atol=1e-12)
    values, columns, expected = model.shap_values(empty_rows[:0])
    assert values.shape == (0, len(columns)) and np.isnan(expected)
