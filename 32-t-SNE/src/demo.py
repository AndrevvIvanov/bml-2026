"""Reproduce the t-SNE presentation's numbers and editable chart data.

Run: python demo.py
The bundled sklearn digits data requires no network download.
"""
from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent.parent / "results"


def low_affinities(y):
    weights = 1 / (1 + cdist(y, y, "sqeuclidean"))
    np.fill_diagonal(weights, 0)
    return weights / weights.sum(), weights


def kl(p, y):
    q, _ = low_affinities(y)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def gradient(p, y):
    q, weights = low_affinities(y)
    return 4 * np.sum(
        ((p - q) * weights)[:, :, None] * (y[:, None] - y[None, :]), axis=1
    )


def numerical_example():
    # Fixed sigma=1 is a teaching simplification. Real t-SNE fits sigma_i
    # separately for every row to match a shared target perplexity.
    x = np.array([[0.0], [1.0], [4.0], [5.0]])
    squared = cdist(x, x, "sqeuclidean")
    weights = np.exp(-squared / 2)
    np.fill_diagonal(weights, 0)
    conditional = weights / weights.sum(axis=1, keepdims=True)
    p = (conditional + conditional.T) / (2 * len(x))
    # Start with the close pair A,B separated; a single gradient step fixes it.
    bad = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    good = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    grad = gradient(p, bad)
    rate = 1.0
    after = bad - rate * grad
    # Check the analytical gradient against independent finite differences.
    h = 1e-5
    finite = np.zeros_like(bad)
    for i in range(len(bad)):
        for j in range(2):
            plus, minus = bad.copy(), bad.copy()
            plus[i, j] += h
            minus[i, j] -= h
            finite[i, j] = (kl(p, plus) - kl(p, minus)) / (2 * h)
    np.testing.assert_allclose(grad, finite, atol=1e-8)
    assert kl(p, after) < kl(p, bad)
    assert np.isclose(p.sum(), 1) and np.allclose(p, p.T)
    q_bad, _ = low_affinities(bad)
    q_good, _ = low_affinities(good)
    return dict(
        labels=list("ABCD"), x=x.ravel().tolist(), sigma=1,
        gaussian_weights=weights.tolist(), row_normalizers=weights.sum(axis=1).tolist(),
        squared_distances=squared.tolist(), conditional=conditional.tolist(),
        p=p.tolist(), bad=bad.tolist(), good=good.tolist(),
        q_bad=q_bad.tolist(), q_good=q_good.tolist(), gradient=grad.tolist(),
        low_weights_bad=low_affinities(bad)[1].tolist(),
        low_normalizer_bad=float(low_affinities(bad)[1].sum()),
        learning_rate=rate, after=after.tolist(),
        kl_bad=kl(p, bad), kl_good=kl(p, good), kl_after=kl(p, after),
    )


def main():
    OUT.mkdir(exist_ok=True)
    digits = load_digits()
    # All pixels have the same units. Divide by 16 without reweighting pixels.
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target
    results = {}
    with threadpool_limits(limits=1):
        start = perf_counter()
        model = PCA(n_components=2, svd_solver="full")
        embedding = model.fit_transform(x)
        seconds = perf_counter() - start
        results["pca"] = dict(
            embedding=embedding.tolist(), seconds=seconds,
            trustworthiness=float(trustworthiness(x, embedding, n_neighbors=10)),
            explained_variance=float(model.explained_variance_ratio_.sum()),
        )
        runs = [(5, 42, "pca"), (30, 42, "pca"), (50, 42, "pca"),
                (30, 0, "random"), (30, 42, "random")]
        for perplexity, seed, init in runs:
            name = f"tsne_p{perplexity}_seed{seed}_{init}"
            start = perf_counter()
            model = TSNE(n_components=2, perplexity=perplexity, init=init,
                         learning_rate="auto", max_iter=1000,
                         random_state=seed, n_jobs=1)
            embedding = model.fit_transform(x)
            seconds = perf_counter() - start
            results[name] = dict(
                embedding=embedding.tolist(), seconds=seconds,
                trustworthiness=float(trustworthiness(x, embedding, n_neighbors=10)),
                kl=float(model.kl_divergence_), iterations=int(model.n_iter_),
                perplexity=perplexity, seed=seed, init=init,
            )
            print(f"{name}: {seconds:.2f}s, T(10)={results[name]['trustworthiness']:.4f}", flush=True)
    output = dict(
        dataset=dict(name="sklearn.load_digits", samples=len(x), features=x.shape[1], classes=len(np.unique(y)),
                     image_shape=list(digits.images.shape[1:]),
                     sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                     scaling="X / 16", target=y.tolist()),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__,
                         threads=1, platform=platform.platform()),
        runs=results, numerical=numerical_example(),
        kernels=dict(distance=[i / 10 for i in range(51)],
                     gaussian=[float(np.exp(-((i / 10) ** 2) / 2)) for i in range(51)],
                     student=[1 / (1 + (i / 10) ** 2) for i in range(51)]),
    )
    (OUT / "experiment.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {name: {k: v for k, v in run.items() if k != "embedding"}
               for name, run in results.items()}
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["numerical"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
