#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-free out-of-sample evaluation for Supervised K-ISOMAP.

Protocol
--------
1. Apply the predefined dataset subsampling rules when required.
2. Split the original feature matrix into train/test partitions (50/50).
3. Fit every data-dependent preprocessing step on X_train only:
      - categorical encoding
      - optional high-dimensional PCA (MNIST/CIFAR)
      - StandardScaler
4. Fit each dimensionality-reduction or metric-learning method on training data.
   For supervised methods, only y_train is used.
5. Transform X_test without using y_test.
6. Train the eight downstream classifiers on the training embedding only.
7. Use y_test only for final balanced-accuracy / macro-F1 evaluation.

Four external out-of-sample mappings are evaluated:
    - geodesic : Isomap-style attachment to the learned training manifold
    - krr      : multi-output Kernel Ridge mapping X_train -> Z_train
    - local    : inverse-distance local interpolation
    - rbf      : RBF interpolation X_train -> Z_train
"""

import os
import time
import datetime
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import scipy as sp

import sklearn.datasets as skdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.manifold import Isomap, trustworthiness
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score, pairwise_distances
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import RBFInterpolator
from scipy.stats import wilcoxon

import umap
from metric_learn import NCA, LMNN, LFDA

from supervised_k_isomap import SupervisedKIsomap

warnings.simplefilter(action="ignore")


# 
# EXPERIMENT CONFIGURATION
# 

TEST_SIZE = 0.50
SPLIT_RANDOM_STATE = 42

# Common target dimensionality for all compared representations.
# Methods with an intrinsic lower ceiling (notably LDA: c-1) use the
# largest admissible dimensionality not exceeding this value.
NUM_RUNS = 10
EMBEDDING_DIM = 2
D_VARIATION = [EMBEDDING_DIM]
NOISE_LEVELS = [0]

# OOS variants to compare for SK-ISOMAP.
SKISO_OOS_METHODS = ("geodesic", "krr", "local", "rbf")

# Fixed OOS parameters; no outer-test-set tuning is performed.
KRR_PARAMS = {
    "alpha": 1.0,
    "kernel": "rbf",
    "gamma": None,
}

LOCAL_PARAMS = {
    "weights": "distance",
    "power": 1.0,
}

GEODESIC_PARAMS = {
    "entry_scale": 1.0,
}

# A local RBF is used to avoid a dense global RBF system on large datasets.
# degree=0 avoids high-dimensional polynomial-rank requirements.
RBF_MAX_NEIGHBORS = 50
RBF_PARAMS = {
    # Linear radial basis with a small smoothing term.
    "kernel": "linear",
    "smoothing": 1e-8,
    "degree": 0,
}

# S-Isomap (Geng, Zhan & Zhou) parameters. Beta is estimated from training data
# as the mean pairwise Euclidean distance.
S_ISOMAP_ALPHA = 0.5

# Main method used in the pairwise summary tables.
PRIMARY_SKISO_PREFIX = f"LF_SUP_KISO_RBF_d{EMBEDDING_DIM}"


# Full dataset list used by the current experimental suite.
# Individual dataset failures are logged without stopping the remaining runs.
DATASETS = [
    ("zoo", 1),
    ("glass", 1),
    ("ecoli", 1),
    ("balance-scale", 1),
    ("energy-efficiency", 1),
    ("vehicle", 1),
    ("vowel", 1),
    ("collins", 1),
    ("cnae-9", 1),
    
    ("wine-quality-red", 1),
    ("one-hundred-plants-texture", 1),
    ("one-hundred-plants-shape", 1),
    ("car-evaluation", 1),
    ("digits", 1),
    ("mfeat-karhunen", 1),
    ("mfeat-pixel", 1),
    ("pendigits", 1),
    ("Indian_pines", 1),

    ("GesturePhaseSegmentationProcessed", 1),
    ("artificial-characters", 1),
    ("thyroid-dis", 1),
    ("led24", 1),
    ("nursery", 1),
    ("eye_movements", 1),
    ("MNIST_784", 1),
    ("CIFAR_10_small", 1),
    ("wine-quality-white", 1),

    ("waveform-5000", 1),
    ("wall-robot-navigation", 1),
    ("optdigits", 1),
    ("satimage", 1),
    ("tic-tac-toe", 1),
    ("diabetes", 1),
    ("grub-damage", 2),
    ("banknote-authentication", 1),
    ("ionosphere", 1),
]


def min_class_size(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.min()


def safe_stratify(y):
    """Return y for stratification only when every class has >= 2 samples."""
    counts = Counter(y)
    return y if counts and min(counts.values()) >= 2 else None


def add_gaussian_noise_train_test(X_train, X_test, noise_level=0.01, random_state=42):
    """
    Add independent Gaussian noise to train/test using a scale estimated ONLY
    from X_train. This prevents test-distribution statistics from affecting the
    perturbation magnitude.
    """
    if noise_level <= 0:
        return X_train.copy(), X_test.copy()

    sigma = noise_level * np.std(X_train, axis=0)

    rng_train = np.random.default_rng(random_state)
    rng_test = np.random.default_rng(random_state + 100000)

    X_train_noisy = X_train + rng_train.normal(0, sigma, X_train.shape)
    X_test_noisy = X_test + rng_test.normal(0, sigma, X_test.shape)

    return X_train_noisy, X_test_noisy


def encode_train_test_features(X_train, X_test):
    """
    Encode categorical/object features using TRAIN only.

    Returns numeric numpy arrays with the original column ordering preserved.
    No scaling is performed here; StandardScaler is fitted afterwards.
    """
    if isinstance(X_train, pd.DataFrame):
        train_df = X_train.copy()
        test_df = X_test.copy()

        categorical_cols = list(train_df.select_dtypes(include=["category", "object", "bool"]).columns)

        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # Convert categorical columns to object before assigning numerical
            # codes; assigning floats directly into pandas Categorical columns
            # can raise a dtype error.
            train_df[categorical_cols] = train_df[categorical_cols].astype(object)
            test_df[categorical_cols] = test_df[categorical_cols].astype(object)
            train_df.loc[:, categorical_cols] = encoder.fit_transform(train_df[categorical_cols].astype(str))
            test_df.loc[:, categorical_cols] = encoder.transform(test_df[categorical_cols].astype(str))

        X_train_num = train_df.to_numpy(dtype=float)
        X_test_num = test_df.to_numpy(dtype=float)
        return X_train_num, X_test_num

    return np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)


def apply_optional_train_only_pca(dataset_name, X_train, X_test):
    """
    Reproduce the previous dimensionality cap for MNIST/CIFAR, but fit the PCA
    only on training data and then transform the held-out test samples.
    """
    name = dataset_name.lower()

    if name == "cifar_10_small":
        requested = 30
    elif name == "mnist_784":
        requested = 50
    else:
        return X_train, X_test, None

    n_components = min(requested, X_train.shape[1], max(1, X_train.shape[0] - 1))

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca


def standardize_train_test(X_train, X_test):
    """Fit StandardScaler on train only and apply the same transform to test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def sanitize_train_test_embedding(Z_train, Z_test):
    """
    Numerical cleanup only.

    No embedding-specific scaling is performed here. All representations are
    standardized in exactly the same train-only way inside
    ClassificationLeakageFree(), while geometric quality scores receive the
    native embedding coordinates produced by each method.
    """
    Z_train = np.asarray(Z_train).real
    Z_test = np.asarray(Z_test).real

    Z_train = np.nan_to_num(Z_train, nan=0.0, posinf=1e10, neginf=-1e10)
    Z_test = np.nan_to_num(Z_test, nan=0.0, posinf=1e10, neginf=-1e10)

    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
    if Z_test.ndim == 1:
        Z_test = Z_test.reshape(-1, 1)

    return Z_train, Z_test, None


def fit_supervised_pca_projection(X_train, y_train, n_components=2):
    """Fit the projection matrix used by the original SupervisedPCA()."""
    X_t = X_train.T
    n = X_t.shape[1]

    H = np.eye(n) - (1 / n) * np.ones((n, n))
    L = (y_train[:, None] == y_train[None, :]).astype(float)

    Q = X_t @ H @ L @ H @ X_t.T

    eigvals, eigvecs = np.linalg.eig(Q)
    order = eigvals.argsort()

    d = min(n_components, eigvecs.shape[1])
    Wproj = eigvecs[:, order[-d:]]

    return np.real(Wproj)


def fit_slpp_projection(X_train, y_train, n_components=2, k=5):
    """Fit Supervised Locality Preserving Projection on TRAIN only."""
    n = X_train.shape[0]
    k = min(max(1, k), n - 1)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)
    _, indices = knn.kneighbors(X_train)

    W = np.zeros((n, n))

    for i in range(n):
        for j in indices[i]:
            if y_train[i] == y_train[j]:
                W[i, j] = 1
                W[j, i] = 1

    D = np.diag(W.sum(axis=1))
    L = D - W

    eps = 1e-6
    XTX = X_train.T @ D @ X_train + eps * np.eye(X_train.shape[1])
    XTLX = X_train.T @ L @ X_train

    eigvals, eigvecs = sp.linalg.eigh(XTLX, XTX)
    idx = np.argsort(eigvals)

    d = min(n_components, eigvecs.shape[1])
    return np.real(eigvecs[:, idx[:d]])


def fit_lde_projection(X_train, y_train, n_components=2, k=5):
    """Fit Local Discriminant Embedding projection on TRAIN only."""
    n = X_train.shape[0]
    k = min(max(1, k), n - 1)

    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X_train)
    _, indices = knn.kneighbors(X_train)

    Ww = np.zeros((n, n))
    Wb = np.zeros((n, n))

    for i in range(n):
        for j in indices[i][1:]:
            if y_train[i] == y_train[j]:
                Ww[i, j] = 1
                Ww[j, i] = 1
            else:
                Wb[i, j] = 1
                Wb[j, i] = 1

    Dw = np.diag(Ww.sum(axis=1))
    Db = np.diag(Wb.sum(axis=1))

    Lw = Dw - Ww
    Lb = Db - Wb

    eps = 1e-6
    A = X_train.T @ Lb @ X_train
    B = X_train.T @ Lw @ X_train + eps * np.eye(X_train.shape[1])

    eigvals, eigvecs = sp.linalg.eigh(A, B)
    idx = np.argsort(eigvals)[::-1]

    d = min(n_components, eigvecs.shape[1])
    return np.real(eigvecs[:, idx[:d]])


def fit_mfa_projection(X_train, y_train, n_components=2, k1=5, k2=5):
    """Fit Marginal Fisher Analysis projection on TRAIN only."""
    n = X_train.shape[0]
    kmax = min(max(1, max(k1, k2)), n - 1)

    knn = NearestNeighbors(n_neighbors=kmax + 1)
    knn.fit(X_train)
    _, indices = knn.kneighbors(X_train)

    Ww = np.zeros((n, n))
    Wb = np.zeros((n, n))

    for i in range(n):
        count_w = 0
        count_b = 0

        # Preserve the original MFA implementation, which iterates over
        # the complete neighbor list returned by kneighbors().
        for j in indices[i]:
            if y_train[i] == y_train[j] and count_w < k1:
                Ww[i, j] = 1
                Ww[j, i] = 1
                count_w += 1

            if y_train[i] != y_train[j] and count_b < k2:
                Wb[i, j] = 1
                Wb[j, i] = 1
                count_b += 1

    Dw = np.diag(Ww.sum(axis=1))
    Db = np.diag(Wb.sum(axis=1))

    Lw = Dw - Ww
    Lb = Db - Wb

    eps = 1e-6
    A = X_train.T @ Lb @ X_train
    B = X_train.T @ Lw @ X_train + eps * np.eye(X_train.shape[1])

    eigvals, eigvecs = sp.linalg.eigh(A, B)
    idx = np.argsort(eigvals)[::-1]

    d = min(n_components, eigvecs.shape[1])
    return np.real(eigvecs[:, idx[:d]])


def rbf_oos_from_embedding(X_train, Z_train, X_test):
    """Fit the RBF OOS mapper used by SK-ISOMAP and project X_test."""
    n_neighbors = min(RBF_MAX_NEIGHBORS, X_train.shape[0])
    mapper = RBFInterpolator(
        X_train,
        np.asarray(Z_train).real,
        neighbors=n_neighbors,
        **RBF_PARAMS,
    )
    return np.asarray(mapper(X_test), dtype=float)


def supervised_isomap_train_embedding(X_train, y_train, n_neighbors, n_components, alpha=S_ISOMAP_ALPHA):
    """
    Canonical S-Isomap training embedding (Geng, Zhan & Zhou, 2005).

    Only TRAIN labels are used. The label-dependent dissimilarity is

        same class:      sqrt(1 - exp(-d^2 / beta))
        different class: sqrt(exp(d^2 / beta) - alpha)

    with beta equal to the mean pairwise Euclidean distance in TRAIN.
    Standard Isomap is then run using this precomputed dissimilarity matrix.

    The caller applies the same RBF mapper used for the other matched OOS
    comparisons.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    D = pairwise_distances(X_train, metric="euclidean")
    tri = D[np.triu_indices_from(D, k=1)]
    beta = float(np.mean(tri)) if tri.size else 1.0
    if not np.isfinite(beta) or beta <= 0:
        beta = 1.0

    ratio = (D ** 2) / beta
    ratio = np.clip(ratio, 0.0, 700.0)

    same = y_train[:, None] == y_train[None, :]
    D_sup = np.empty_like(D, dtype=float)
    D_sup[same] = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-ratio[same])))
    D_sup[~same] = np.sqrt(
        np.maximum(0.0, np.exp(ratio[~same]) - float(alpha))
    )
    np.fill_diagonal(D_sup, 0.0)

    model = Isomap(
        n_neighbors=min(max(1, n_neighbors), X_train.shape[0] - 1),
        n_components=n_components,
        metric="precomputed",
    )
    Z_train = model.fit_transform(D_sup)
    return np.asarray(Z_train).real, beta


def sammon_stress(X_high, X_low):
    d_high = pairwise_distances(X_high)
    d_low = pairwise_distances(X_low)

    mask = ~np.eye(d_high.shape[0], dtype=bool)
    eps = 1e-12

    numerator = ( ((d_high[mask] - d_low[mask]) ** 2 / (d_high[mask] + eps)).sum() )
    denominator = d_high[mask].sum()

    return numerator / denominator


def knn_preservation(X_high, X_low, k=15):
    k = min(k, X_high.shape[0] - 1)

    neigh_high = ( NearestNeighbors(n_neighbors=k + 1).fit(X_high).kneighbors(return_distance=False) )
    neigh_low = ( NearestNeighbors(n_neighbors=k + 1).fit(X_low).kneighbors(return_distance=False) )

    intersect = [len(np.intersect1d(h[1:], l[1:])) for h, l in zip(neigh_high, neigh_low) ]

    return np.mean(intersect) / k


def continuity(X_high, X_low, k=15):
    n = X_high.shape[0]
    k = min(k, n - 1)

    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    order_high = np.argsort(D_high, axis=1)
    order_low = np.argsort(D_low, axis=1)

    rank_high = np.empty_like(order_high)
    rank_low = np.empty_like(order_low)

    rank_high[np.arange(n)[:, None], order_high] = np.arange(n)
    rank_low[np.arange(n)[:, None], order_low] = np.arange(n)

    penalty = 0.0

    for i in range(n):
        neighbors_high = order_high[i, 1 : k + 1]

        for j in neighbors_high:
            if rank_low[i, j] > k:
                penalty += rank_low[i, j] - k

    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return np.nan

    norm = 2.0 / denom
    return 1.0 - norm * penalty


def neighborhood_distribution(X, k=15):
    k = min(k, X.shape[0] - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)

    dist = dist[:, 1:]

    sigma = np.mean(dist, axis=1, keepdims=True)
    sigma[sigma == 0] = 1e-12

    sim = np.exp(-(dist**2) / (2 * sigma**2))
    denom = np.sum(sim, axis=1, keepdims=True)
    denom[denom == 0] = 1.0

    return sim / denom


def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def distribution_scores(X_high, X_low, k=15):
    P = neighborhood_distribution(X_high, k)
    Q = neighborhood_distribution(X_low, k)

    js_scores = []
    hell_scores = []

    for i in range(P.shape[0]):
        js_scores.append(jensenshannon(P[i], Q[i]) ** 2)
        hell_scores.append(hellinger_distance(P[i], Q[i]))

    return np.mean(js_scores), np.mean(hell_scores)


def embedding_quality_scores(X_train, X_test, Z_train, Z_test, y_train, y_test):
    """
    Compute descriptive embedding-quality metrics after the leakage-free
    transform. Test labels are used ONLY for silhouette evaluation, never for
    fitting or projection.
    """
    # OOS quality is evaluated on held-out samples only.
    X_all = np.asarray(X_test)
    Z_all = np.asarray(Z_test)
    y_all = np.asarray(y_test)

    n = X_all.shape[0]
    k_eval = min(15, max(1, n - 1))
    k_trust = min(k_eval, max(1, (n - 1) // 2))

    scores = {}

    try:
        scores["sc"] = silhouette_score(Z_all, y_all, metric="euclidean")
    except Exception:
        scores["sc"] = np.nan

    try:
        scores["t_score"] = trustworthiness(X_all, Z_all, n_neighbors=k_trust)
    except Exception:
        scores["t_score"] = np.nan

    try:
        scores["c_score"] = continuity(X_all, Z_all, k=k_eval)
    except Exception:
        scores["c_score"] = np.nan

    try:
        scores["knn_preserv"] = knn_preservation(X_all, Z_all, k=k_eval)
    except Exception:
        scores["knn_preserv"] = np.nan

    try:
        scores["stress"] = sammon_stress(X_all, Z_all)
    except Exception:
        scores["stress"] = np.nan

    try:
        js, hell = distribution_scores(X_all, Z_all, k=k_eval)
        scores["js"] = js
        scores["hell"] = hell
        scores["js_sim"] = 1 - js
        scores["hell_sim"] = 1 - hell
    except Exception:
        scores["js"] = np.nan
        scores["hell"] = np.nan
        scores["js_sim"] = np.nan
        scores["hell_sim"] = np.nan

    return scores


def ClassificationLeakageFree(X_train, X_test, y_train, y_test, method, random_state=42):
    """
    Train and evaluate eight downstream classifiers without another split.

    The supplied train/test partition is created before preprocessing and
    dimensionality reduction.
    """
    X_train = np.asarray(X_train).real
    X_test = np.asarray(X_test).real

    # Standardize each representation using training statistics before fitting
    # the classifiers.
    cls_scaler = StandardScaler()
    X_train = cls_scaler.fit_transform(X_train)
    X_test = cls_scaler.transform(X_test)

    acc_list = []
    f1_list = []
    per_classifier = {}

    def register(name, pred):
        acc = balanced_accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)

        acc_list.append(acc)
        f1_list.append(f1)

        per_classifier[f"{name}_acc"] = acc
        per_classifier[f"{name}_f1"] = f1

    # KNN
    k_cls = min(5, max(1, X_train.shape[0]))
    clf = KNeighborsClassifier(n_neighbors=k_cls)
    clf.fit(X_train, y_train)
    register("KNN", clf.predict(X_test))

    # SVM
    clf = SVC(gamma="auto")
    clf.fit(X_train, y_train)
    register("SVM", clf.predict(X_test))

    # Gaussian Naive Bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    register("NB", clf.predict(X_test))

    # Decision Tree
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    register("DT", clf.predict(X_test))

    # Quadratic Discriminant Analysis
    counts = pd.Series(y_train).value_counts()
    valid_classes = counts[counts > 1].index
    mask = np.isin(y_train, valid_classes)

    X_train_qda = X_train[mask]
    y_train_qda = y_train[mask]

    if len(np.unique(y_train_qda)) >= 2:
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train_qda, y_train_qda)
        register("QDA", clf.predict(X_test))
    else:
        per_classifier["QDA_acc"] = np.nan
        per_classifier["QDA_f1"] = np.nan

    # MLP
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    register("MLP", clf.predict(X_test))

    # Gaussian Process
    clf = GaussianProcessClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    register("GPC", clf.predict(X_test))

    # Random Forest
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    register("RF", clf.predict(X_test))

    if not acc_list:
        raise RuntimeError(f"No classifier could be evaluated for {method}.")

    result = {
        "acc_avg": float(np.mean(acc_list)),
        "acc_max": float(np.max(acc_list)),
        "f1_avg": float(np.mean(f1_list)),
        "f1_max": float(np.max(f1_list)),
        "n_classifiers_valid": len(acc_list),
    }
    result.update(per_classifier)

    print()
    print(f"[{method}] Maximum balanced accuracy: {result['acc_max']:.6f}")
    print(f"[{method}] Maximum macro F1-score: {result['f1_max']:.6f}")
    print(f"[{method}] Average balanced accuracy: {result['acc_avg']:.6f}")
    print(f"[{method}] Average macro F1-score: {result['f1_avg']:.6f}")

    return result


def store_results(row, prefix, elapsed, cls_result, quality_result=None):
    row[f"{prefix}_time"] = elapsed

    for key, value in cls_result.items():
        row[f"{prefix}_{key}"] = value

    if quality_result is not None:
        for key, value in quality_result.items():
            row[f"{prefix}_{key}"] = value


def load_dataset(dataset_name, version):
    if dataset_name == "digits":
        return skdata.load_digits()

    return skdata.fetch_openml( name=dataset_name, version=version, as_frame=True )


def apply_original_subsampling(dataset_name, X, y):
    """
    Apply the predefined dataset-size reductions before the outer split.
    """
    name = dataset_name.lower()

    fractions = {
        "gesturephasesegmentationprocessed": 0.25,
        "indian_pines": 0.25,
        "cifar_10_small": 0.20,
        "pendigits": 0.20,
        "artificial-characters": 0.25,
        "nursery": 0.25,
        "eye_movements": 0.30,
        "mnist_784": 0.05,
    }

    if name not in fractions:
        return X, y

    fraction = fractions[name]

    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=fraction, random_state=42, stratify=None)

    return X_keep, y_keep


def evaluate_and_store(
    row,
    prefix,
    X_train,
    X_test,
    y_train,
    y_test,
    Z_train,
    Z_test,
    elapsed,
    random_state,
    quality=True,
):
    cls_result = ClassificationLeakageFree(
        Z_train,
        Z_test,
        y_train,
        y_test,
        prefix,
        random_state=random_state,
    )

    quality_result = None
    if quality:
        quality_result = embedding_quality_scores(
            X_train,
            X_test,
            Z_train,
            Z_test,
            y_train,
            y_test,
        )

    store_results(
        row,
        prefix,
        elapsed,
        cls_result,
        quality_result,
    )


METHOD_PREFIXES = [
    "LF_RAW", "LF_PLS", "LF_S-UMAP", "LF_SUP_PCA", "LF_LDA", "LF_NCA",
    "LF_LMNN", "LF_LFDA", "LF_SLPP", "LF_LDE", "LF_MFA", "LF_ISO",
    "LF_ISO_RBF", "LF_S_ISOMAP_RBF",
    f"LF_SUP_KISO_GEO_d{EMBEDDING_DIM}",
    f"LF_SUP_KISO_KRR_d{EMBEDDING_DIM}",
    f"LF_SUP_KISO_LOCAL_d{EMBEDDING_DIM}",
    f"LF_SUP_KISO_RBF_d{EMBEDDING_DIM}",
]


def generate_run_summaries(df, output_dir):
    """Generate mean±SD and paired SK-ISOMAP-RBF comparisons."""
    valid = df.copy()
    if "status" in valid.columns:
        valid = valid[valid["status"] == "OK"].copy()

    rows = []
    for dataset_name, g in valid.groupby("dataset_name"):
        for method in METHOD_PREFIXES:
            acc_col = f"{method}_acc_avg"
            f1_col = f"{method}_f1_avg"
            if acc_col not in g.columns:
                continue
            acc = pd.to_numeric(g[acc_col], errors="coerce").dropna()
            f1 = pd.to_numeric(g[f1_col], errors="coerce").dropna() if f1_col in g.columns else pd.Series(dtype=float)
            if acc.empty:
                continue
            acc_std = float(acc.std(ddof=1)) if len(acc) > 1 else 0.0
            f1_std = float(f1.std(ddof=1)) if len(f1) > 1 else (0.0 if len(f1) == 1 else np.nan)
            rows.append({
                "dataset_name": dataset_name,
                "method": method,
                "n_runs": int(len(acc)),
                "balanced_accuracy_mean": float(acc.mean()),
                "balanced_accuracy_std": acc_std,
                "balanced_accuracy_mean_std": f"{acc.mean():.4f} ± {acc_std:.4f}",
                "macro_f1_mean": float(f1.mean()) if not f1.empty else np.nan,
                "macro_f1_std": f1_std,
                "macro_f1_mean_std": f"{f1.mean():.4f} ± {f1_std:.4f}" if not f1.empty else "",
            })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "results_leakage_free_summary_mean_std.csv")
    summary.to_csv(summary_path, index=False, sep=";", decimal=",")

    if not summary.empty:
        wide = summary.pivot(index="dataset_name", columns="method", values="balanced_accuracy_mean_std")
        wide.to_csv(os.path.join(output_dir, "table_balanced_accuracy_mean_std.csv"), sep=";")

    pair_rows = []
    primary_col = f"{PRIMARY_SKISO_PREFIX}_acc_avg"
    for dataset_name, g in valid.groupby("dataset_name"):
        if primary_col not in g.columns:
            continue
        for competitor in METHOD_PREFIXES:
            if competitor == PRIMARY_SKISO_PREFIX:
                continue
            comp_col = f"{competitor}_acc_avg"
            if comp_col not in g.columns:
                continue
            tmp = g[["run_time", primary_col, comp_col]].copy()
            tmp[primary_col] = pd.to_numeric(tmp[primary_col], errors="coerce")
            tmp[comp_col] = pd.to_numeric(tmp[comp_col], errors="coerce")
            tmp = tmp.dropna()
            if tmp.empty:
                continue
            p = tmp[primary_col].to_numpy(float)
            c = tmp[comp_col].to_numpy(float)
            diff = p - c
            eps = 1e-12
            if len(diff) >= 2 and not np.all(np.abs(diff) <= eps):
                try:
                    stat, pvalue = wilcoxon(diff, alternative="two-sided")
                except Exception:
                    stat, pvalue = np.nan, np.nan
            else:
                stat, pvalue = np.nan, np.nan
            pair_rows.append({
                "dataset_name": dataset_name,
                "primary": PRIMARY_SKISO_PREFIX,
                "competitor": competitor,
                "n_paired_runs": int(len(diff)),
                "primary_mean": float(np.mean(p)),
                "primary_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
                "competitor_mean": float(np.mean(c)),
                "competitor_std": float(np.std(c, ddof=1)) if len(c) > 1 else 0.0,
                "mean_paired_difference": float(np.mean(diff)),
                "std_paired_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                "wins": int(np.sum(diff > eps)),
                "ties": int(np.sum(np.abs(diff) <= eps)),
                "losses": int(np.sum(diff < -eps)),
                "wilcoxon_statistic_runs": stat,
                "wilcoxon_pvalue_runs": pvalue,
            })

    pairwise = pd.DataFrame(pair_rows)
    pairwise_path = os.path.join(output_dir, "pairwise_skiso_rbf_vs_all_by_dataset.csv")
    pairwise.to_csv(pairwise_path, index=False, sep=";", decimal=",")

    overall_rows = []
    if not pairwise.empty:
        for competitor, g in pairwise.groupby("competitor"):
            diffs = g["mean_paired_difference"].to_numpy(float)
            eps = 1e-12
            if len(diffs) >= 2 and not np.all(np.abs(diffs) <= eps):
                try:
                    stat, pvalue = wilcoxon(diffs, alternative="two-sided")
                except Exception:
                    stat, pvalue = np.nan, np.nan
            else:
                stat, pvalue = np.nan, np.nan
            overall_rows.append({
                "primary": PRIMARY_SKISO_PREFIX,
                "competitor": competitor,
                "n_datasets": int(len(diffs)),
                "dataset_wins": int(np.sum(diffs > eps)),
                "dataset_ties": int(np.sum(np.abs(diffs) <= eps)),
                "dataset_losses": int(np.sum(diffs < -eps)),
                "mean_difference_across_datasets": float(np.mean(diffs)),
                "std_difference_across_datasets": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                "wilcoxon_statistic_dataset_means": stat,
                "wilcoxon_pvalue_dataset_means": pvalue,
            })

    overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(output_dir, "pairwise_skiso_rbf_vs_all_overall.csv")
    overall.to_csv(overall_path, index=False, sep=";", decimal=",")

    # Diagnostic table for optional OOS failures. This makes it explicit when
    # GEO/KRR/LOCAL failed without discarding a valid RBF run.
    failure_rows = []
    status_cols = [
        "LF_SUP_KISO_GEO_status",
        "LF_SUP_KISO_KRR_status",
        "LF_SUP_KISO_LOCAL_status",
        "LF_SUP_KISO_RBF_status",
    ]
    for dataset_name, g in df.groupby("dataset_name"):
        for status_col in status_cols:
            if status_col not in g.columns:
                continue
            values = g[status_col].fillna("NOT_RUN").astype(str)
            failure_rows.append({
                "dataset_name": dataset_name,
                "oos_method": status_col.replace("LF_SUP_KISO_", "").replace("_status", ""),
                "n_total_rows": int(len(values)),
                "n_ok": int((values == "OK").sum()),
                "n_error": int((values == "ERROR").sum()),
                "n_not_run": int((values == "NOT_RUN").sum()),
            })
    failures = pd.DataFrame(failure_rows)
    failures_path = os.path.join(output_dir, "skiso_oos_failure_summary.csv")
    failures.to_csv(failures_path, index=False, sep=";", decimal=",")

    # Dataset/protocol exclusions are summarized separately from computational
    # errors. SKIPPED rows never enter mean±SD or paired statistical tests.
    skip_rows = []
    if "status" in df.columns:
        status_text = df["status"].fillna("").astype(str)
        skipped = df[status_text.str.startswith("SKIPPED_")].copy()

        if not skipped.empty:
            group_cols = ["dataset_name", "status"]
            if "skip_reason" in skipped.columns:
                group_cols.append("skip_reason")

            for keys, g in skipped.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                info = dict(zip(group_cols, keys))
                info["n_skipped_runs"] = int(len(g))

                if "min_class_support" in g.columns:
                    vals = pd.to_numeric(g["min_class_support"], errors="coerce").dropna()
                    info["min_class_support"] = int(vals.min()) if not vals.empty else np.nan

                if "rare_classes" in g.columns:
                    non_null = g["rare_classes"].dropna().astype(str).unique().tolist()
                    info["rare_classes"] = non_null[0] if len(non_null) == 1 else " | ".join(non_null)

                if "rare_class_counts" in g.columns:
                    non_null = g["rare_class_counts"].dropna().astype(str).unique().tolist()
                    info["rare_class_counts"] = non_null[0] if len(non_null) == 1 else " | ".join(non_null)

                skip_rows.append(info)

    skip_summary_columns = [
        "dataset_name",
        "status",
        "skip_reason",
        "n_skipped_runs",
        "min_class_support",
        "rare_classes",
        "rare_class_counts",
    ]
    skip_summary = pd.DataFrame(skip_rows)
    if skip_summary.empty:
        skip_summary = pd.DataFrame(columns=skip_summary_columns)
    else:
        for col in skip_summary_columns:
            if col not in skip_summary.columns:
                skip_summary[col] = np.nan
        skip_summary = skip_summary[skip_summary_columns]

    skip_path = os.path.join(output_dir, "dataset_skip_summary.csv")
    skip_summary.to_csv(skip_path, index=False, sep=";", decimal=",")

    print("\nGenerated aggregate files:")
    print(f"  - {summary_path}")
    print(f"  - {pairwise_path}")
    print(f"  - {overall_path}")
    print(f"  - {failures_path}")
    print(f"  - {skip_path}")
    return summary, pairwise, overall


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(base_dir, "files_sup_kiso_leakage_free_d2_isomap")
    os.makedirs(files_dir, exist_ok=True)

    csv_file = os.path.join(files_dir, "results_leakage_free_oos.csv")

    all_results = []

    for current_noise in NOISE_LEVELS:
        for run in range(1, NUM_RUNS + 1):
            split_seed = SPLIT_RANDOM_STATE + (run - 1)

            for dataset_name, version in DATASETS:
                row = {}

                try:
                    print("\n" + "=" * 78)
                    print(f"LEAKAGE-FREE PROCESSING: {dataset_name}")
                    print("=" * 78)

                    timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    dataset = load_dataset(dataset_name, version)
                    X = dataset["data"]
                    y = np.asarray(dataset["target"])

                    X, y = apply_original_subsampling(dataset_name, X, y)

                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)

                    n_total = len(y)
                    n_features_original = X.shape[1]
                    class_values, class_counts = np.unique(y, return_counts=True)
                    n_classes = len(class_values)
                    min_support = int(class_counts.min()) if len(class_counts) else 0
                    max_support = int(class_counts.max()) if len(class_counts) else 0

                    # Metadata available even when the dataset must be skipped.
                    row.update(
                        {
                            "Timestamp": timestamp,
                            "dataset_name": dataset_name,
                            "run_time": run,
                            "split_random_state": split_seed,
                            "protocol": "leakage_free_outer_holdout_50_50",
                            "n_samples_total": n_total,
                            "n_features_original": n_features_original,
                            "n_classes": n_classes,
                            "min_class_support": min_support,
                            "max_class_support": max_support,
                            "noise_level": current_noise,
                            "test_size": TEST_SIZE,
                            "embedding_dim_target": EMBEDDING_DIM,
                            "rbf_kernel": RBF_PARAMS["kernel"],
                            "rbf_smoothing": RBF_PARAMS["smoothing"],
                            "rbf_degree": RBF_PARAMS["degree"],
                            "rbf_max_neighbors": RBF_MAX_NEIGHBORS,
                        }
                    )

                    # 
                    # STRICT HOLDOUT FEASIBILITY CHECK
                    # 
                    # A leakage-free holdout requires at least one instance of
                    # every class in BOTH train and test. Therefore every class
                    # must contain at least two samples after the dataset-level
                    # subsampling used by the original experimental protocol.
                    rare_mask = class_counts < 2
                    if np.any(rare_mask):
                        rare_classes = class_values[rare_mask]
                        rare_counts = class_counts[rare_mask]

                        row.update(
                            {
                                "status": "SKIPPED_INSUFFICIENT_CLASS_SUPPORT",
                                "skip_reason": (
                                    "At least one class has fewer than 2 samples; "
                                    "the class cannot be represented in both train and test."
                                ),
                                "rare_classes": str(rare_classes.tolist()),
                                "rare_class_counts": str(rare_counts.tolist()),
                            }
                        )

                        print(
                            f"Skipping {dataset_name}: insufficient class support "
                            f"(minimum class size = {min_support}; "
                            f"rare counts = {rare_counts.tolist()})."
                        )

                        all_results.append(row)
                        pd.DataFrame(all_results).to_csv(
                            csv_file,
                            index=False,
                            sep=";",
                            decimal=",",
                        )
                        print(f"\nPartial results saved to: {csv_file}")
                        continue

                    # With a float test_size, sklearn uses ceil(n * test_size)
                    # samples for the test partition. Both partitions must have
                    # at least n_classes slots to represent every class.
                    n_test_expected = int(np.ceil(TEST_SIZE * n_total))
                    n_train_expected = n_total - n_test_expected

                    if n_test_expected < n_classes or n_train_expected < n_classes:
                        row.update(
                            {
                                "status": "SKIPPED_INSUFFICIENT_PARTITION_SIZE",
                                "skip_reason": (
                                    "The requested holdout size cannot place at least one "
                                    "sample of every class in both train and test."
                                ),
                                "expected_n_train": n_train_expected,
                                "expected_n_test": n_test_expected,
                            }
                        )

                        print(
                            f"Skipping {dataset_name}: partition too small for all "
                            f"{n_classes} classes (expected train={n_train_expected}, "
                            f"test={n_test_expected})."
                        )

                        all_results.append(row)
                        pd.DataFrame(all_results).to_csv(
                            csv_file,
                            index=False,
                            sep=";",
                            decimal=",",
                        )
                        print(f"\nPartial results saved to: {csv_file}")
                        continue

                    #
                    # OUTER SPLIT: must happen before learned preprocessing
                    #
                    # Every eligible dataset is stratified. No search over
                    # random seeds is performed to make a split "work".
                    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=split_seed, stratify=y)

                    # Verify that every class is represented in both partitions.
                    classes_all = set(class_values.tolist())
                    train_classes = set(np.unique(y_train).tolist())
                    test_classes = set(np.unique(y_test).tolist())

                    if train_classes != classes_all or test_classes != classes_all:
                        missing_train = sorted(classes_all - train_classes)
                        missing_test = sorted(classes_all - test_classes)

                        row.update(
                            {
                                "status": "SKIPPED_INVALID_OUTER_SPLIT",
                                "skip_reason": (
                                    "Outer split did not contain every class in both "
                                    "partitions despite feasibility checks."
                                ),
                                "missing_classes_train": str(missing_train),
                                "missing_classes_test": str(missing_test),
                            }
                        )

                        print(
                            f"Skipping {dataset_name} run {run}: invalid outer split "
                            f"(missing train={missing_train}, test={missing_test})."
                        )

                        all_results.append(row)
                        pd.DataFrame(all_results).to_csv(
                            csv_file,
                            index=False,
                            sep=";",
                            decimal=",",
                        )
                        print(f"\nPartial results saved to: {csv_file}")
                        continue

                    n_classes_train = len(np.unique(y_train))

                    #
                    # TRAIN-ONLY categorical encoding
                    # 
                    X_train_num, X_test_num = encode_train_test_features(X_train_raw, X_test_raw)

                    #
                    # TRAIN-ONLY optional high-dimensional PCA
                    #
                    X_train_num, X_test_num, pre_pca = (apply_optional_train_only_pca(dataset_name, X_train_num, X_test_num)
                    )

                    #
                    # TRAIN-ONLY standardization
                    #
                    X_train_clean, X_test_clean, scaler = standardize_train_test(X_train_num, X_test_num)

                    #
                    # Noise uses scale estimated from TRAIN only
                    #
                    X_train, X_test = add_gaussian_noise_train_test( X_train_clean, X_test_clean, noise_level=current_noise, random_state=run)

                    n_train = X_train.shape[0]
                    n_test = X_test.shape[0]
                    m_after_preproc = X_train.shape[1]
                    nn_train = max(1, min(round(np.sqrt(n_train)), n_train - 1))

                    print(f"Total samples: {n_total}")
                    print(f"Train samples: {n_train}")
                    print(f"Test samples: {n_test}")
                    print(f"Original features: {n_features_original}")
                    print(f"Features after train-only preprocessing: {m_after_preproc}")
                    print(f"Classes: {n_classes}")
                    print(f"k = round(sqrt(n_train)) = {nn_train}")
                    print(f"Noise level: {current_noise}")

                    row.update(
                        {
                            "n_train": n_train,
                            "n_test": n_test,
                            "n_features_after_preprocessing": m_after_preproc,
                            "nn_train": nn_train,
                        }
                    )

                    #
                    # RAW baseline
                    #
                    start = time.time()
                    Z_train = X_train
                    Z_test = X_test
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_RAW",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                        quality=False,
                    )

                    #
                    # PLS
                    #
                    start = time.time()
                    d_pls = min(EMBEDDING_DIM, X_train.shape[1], max(1, n_train - 1))
                    model_pls = PLSRegression(n_components=d_pls)
                    model_pls.fit(X_train, y_train)
                    Z_train = model_pls.transform(X_train)
                    Z_test = model_pls.transform(X_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_PLS",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # Supervised UMAP
                    #
                    start = time.time()
                    model_umap = umap.UMAP(
                        n_components=min(EMBEDDING_DIM, X_train.shape[1]),
                        random_state=split_seed,
                        transform_seed=split_seed,
                    )
                    model_umap.fit(X_train, y=y_train)
                    # Use the fitted training embedding and transform held-out samples.
                    Z_train = np.asarray(model_umap.embedding_).copy()
                    Z_test = model_umap.transform(X_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_S-UMAP",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # Supervised PCA
                    #
                    start = time.time()
                    W_suppca = fit_supervised_pca_projection(X_train, y_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]))
                    Z_train = X_train @ W_suppca
                    Z_test = X_test @ W_suppca
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_SUP_PCA",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # LDA
                    #
                    start = time.time()
                    lda_dim = min(EMBEDDING_DIM, n_classes_train - 1, X_train.shape[1])
                    model_lda = LinearDiscriminantAnalysis(n_components=max(1, lda_dim))
                    model_lda.fit(X_train, y_train)
                    Z_train = model_lda.transform(X_train)
                    Z_test = model_lda.transform(X_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_LDA",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # NCA
                    #
                    start = time.time()
                    model_nca = NCA(n_components=min(EMBEDDING_DIM, X_train.shape[1]), random_state=split_seed)
                    model_nca.fit(X_train, y_train)
                    Z_train = model_nca.transform(X_train)
                    Z_test = model_nca.transform(X_test)
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_NCA",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # LMNN + train-only PCA to the common target dimensionality
                    #
                    start = time.time()
                    safe_k = max(1, min(3, min_class_size(y_train) - 1))
                    model_lmnn = LMNN(n_neighbors=safe_k)
                    model_lmnn.fit(X_train, y_train)
                    Z_train_full = model_lmnn.transform(X_train)
                    Z_test_full = model_lmnn.transform(X_test)

                    lmnn_dim = min(EMBEDDING_DIM, Z_train_full.shape[1], max(1, Z_train_full.shape[0] - 1))
                    lmnn_pca = PCA(n_components=lmnn_dim, random_state=split_seed)
                    Z_train = lmnn_pca.fit_transform(Z_train_full)
                    Z_test = lmnn_pca.transform(Z_test_full)
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_LMNN",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # LFDA
                    #
                    start = time.time()
                    model_lfda = LFDA(n_components=min(EMBEDDING_DIM, X_train.shape[1]), k=nn_train)
                    model_lfda.fit(X_train, y_train)
                    Z_train = model_lfda.transform(X_train)
                    Z_test = model_lfda.transform(X_test)
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_LFDA",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # Supervised LPP
                    #
                    start = time.time()
                    W_slpp = fit_slpp_projection(X_train, y_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]), k=nn_train)
                    Z_train = X_train @ W_slpp
                    Z_test = X_test @ W_slpp
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_SLPP",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # LDE
                    #
                    start = time.time()
                    W_lde = fit_lde_projection(X_train, y_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]), k=nn_train)
                    Z_train = X_train @ W_lde
                    Z_test = X_test @ W_lde
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_LDE",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    #
                    # MFA
                    #
                    start = time.time()
                    W_mfa = fit_mfa_projection( X_train, y_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]), k1=nn_train, k2=nn_train)
                    Z_train = X_train @ W_mfa
                    Z_test = X_test @ W_mfa
                    Z_train, Z_test, _ = sanitize_train_test_embedding(Z_train, Z_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_MFA",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    # 
                    # ISOMAP
                    # 
                    start = time.time()
                    model_iso = Isomap(n_neighbors=nn_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]))
                    Z_train = model_iso.fit_transform(X_train)
                    Z_test = model_iso.transform(X_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_ISO",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train,
                        Z_test,
                        elapsed,
                        split_seed,
                    )

                    # 
                    # ISOMAP + RBF
                    #
                    start = time.time()
                    iso_rbf_model = Isomap(n_neighbors=nn_train, n_components=min(EMBEDDING_DIM, X_train.shape[1]))
                    Z_train_iso_rbf = iso_rbf_model.fit_transform(X_train)
                    Z_test_iso_rbf = rbf_oos_from_embedding(X_train, Z_train_iso_rbf, X_test)
                    elapsed = time.time() - start

                    evaluate_and_store(
                        row,
                        "LF_ISO_RBF",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train_iso_rbf,
                        Z_test_iso_rbf,
                        elapsed,
                        split_seed,
                    )

                    #
                    # SUPERVISED ISOMAP (S-Isomap) + RBF
                    #
                    start = time.time()
                    siso_dim = min(EMBEDDING_DIM, X_train.shape[1])
                    Z_train_siso, siso_beta = supervised_isomap_train_embedding(X_train, y_train, n_neighbors=nn_train, n_components=siso_dim, alpha=S_ISOMAP_ALPHA)
                    Z_test_siso = rbf_oos_from_embedding(X_train, Z_train_siso, X_test)
                    elapsed = time.time() - start
                    row["LF_S_ISOMAP_RBF_alpha"] = S_ISOMAP_ALPHA
                    row["LF_S_ISOMAP_RBF_beta"] = siso_beta

                    evaluate_and_store(
                        row,
                        "LF_S_ISOMAP_RBF",
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        Z_train_siso,
                        Z_test_siso,
                        elapsed,
                        split_seed,
                    )

                    #
                    # SUPERVISED K-ISOMAP
                    #
                    max_d = max(D_VARIATION)

                    start_fit = time.time()
                    skiso_model = SupervisedKIsomap(n_neighbors=nn_train, n_components=max_d)
                    Z_train_max = skiso_model.fit_transform(X_train, y_train)
                    skiso_fit_time = time.time() - start_fit
                    row["LF_SUP_KISO_fit_time"] = skiso_fit_time

                    skiso_oos_outputs = {}
                    skiso_oos_failures = []

                    def try_skiso_oos(oos_name, transform_callable):
                        start_transform = time.time()
                        try:
                            Z_test_max = transform_callable()
                            transform_time = time.time() - start_transform
                            Z_test_max = np.asarray(Z_test_max).real
                            skiso_oos_outputs[oos_name] = (Z_test_max, transform_time)
                            row[f"LF_SUP_KISO_{oos_name}_status"] = "OK"
                            return True
                        except Exception as oos_exc:
                            transform_time = time.time() - start_transform
                            skiso_oos_failures.append(oos_name)
                            row[f"LF_SUP_KISO_{oos_name}_status"] = "ERROR"
                            row[f"LF_SUP_KISO_{oos_name}_error_type"] = (
                                type(oos_exc).__name__
                            )
                            row[f"LF_SUP_KISO_{oos_name}_error_message"] = str(
                                oos_exc
                            )
                            row[f"LF_SUP_KISO_{oos_name}_transform_time"] = (
                                transform_time
                            )
                            print(
                                f"\n[SK-ISOMAP {oos_name}] OOS transform failed "
                                f"but the run will continue: "
                                f"{type(oos_exc).__name__}: {oos_exc}"
                            )
                            traceback.print_exc()
                            return False

                    # ------------------ GEODESIC (ablation) -------------
                    try_skiso_oos(
                        "GEO",
                        lambda: skiso_model.transform_geodesic(
                            X_test,
                            n_neighbors=nn_train,
                            **GEODESIC_PARAMS,
                        ),
                    )

                    # ------------------ KRR (ablation) ------------------
                    try_skiso_oos(
                        "KRR",
                        lambda: skiso_model.transform_krr(
                            X_test,
                            **KRR_PARAMS,
                        ),
                    )

                    # ------------------ LOCAL (ablation) ----------------
                    try_skiso_oos(
                        "LOCAL",
                        lambda: skiso_model.transform_local(
                            X_test,
                            n_neighbors=nn_train,
                            **LOCAL_PARAMS,
                        ),
                    )

                    # ------------------ RBF (PRIMARY OOS) ---------------
                    rbf_neighbors = min(RBF_MAX_NEIGHBORS, n_train)
                    rbf_ok = try_skiso_oos(
                        "RBF",
                        lambda: skiso_model.transform_rbf(
                            X_test,
                            neighbors=rbf_neighbors,
                            **RBF_PARAMS,
                        ),
                    )

                    # Evaluate every OOS mapping that actually succeeded.
                    # Missing ablation values remain NaN in the CSV and are
                    # ignored automatically by the run-level summaries.
                    for d_requested in D_VARIATION:
                        d = min(d_requested, Z_train_max.shape[1])
                        Z_train_d = Z_train_max[:, :d]

                        for oos_name, (Z_test_max, transform_time) in (
                            skiso_oos_outputs.items()
                        ):
                            Z_test_d = Z_test_max[:, :d]

                            prefix = f"LF_SUP_KISO_{oos_name}_d{d}"
                            elapsed = skiso_fit_time + transform_time

                            evaluate_and_store(
                                row,
                                prefix,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                Z_train_d,
                                Z_test_d,
                                elapsed,
                                split_seed,
                            )

                            row[f"{prefix}_fit_time"] = skiso_fit_time
                            row[f"{prefix}_transform_time"] = transform_time

                    row["LF_SUP_KISO_oos_failures"] = ",".join(
                        skiso_oos_failures
                    )

                    # RBF is the primary OOS configuration. Failures in the
                    # sensitivity-analysis mappings do not invalidate the run.
                    if rbf_ok:
                        row["status"] = "OK"
                    else:
                        row["status"] = "PARTIAL_PRIMARY_OOS_ERROR"

                except Exception as e:
                    error_timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    row.setdefault("dataset_name", dataset_name)
                    row.setdefault("run_time", run)
                    row["status"] = "ERROR"
                    row["error_type"] = type(e).__name__
                    row["error_message"] = str(e)
                    row["error_timestamp"] = error_timestamp

                    print(
                        f"\nError processing {dataset_name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    traceback.print_exc()

                all_results.append(row)

                # Overwrite after each dataset so columns remain aligned even
                # when different methods/datasets create different fields.
                pd.DataFrame(all_results).to_csv(
                    csv_file,
                    index=False,
                    sep=";",
                    decimal=",",
                )

                print(f"\nPartial results saved to: {csv_file}")

    final_df = pd.DataFrame(all_results)
    generate_run_summaries(final_df, files_dir)

    print("\n" + "=" * 78)
    print("LEAKAGE-FREE EXPERIMENT FINISHED")
    print(f"Results: {csv_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
