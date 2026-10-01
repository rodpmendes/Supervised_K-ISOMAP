#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised K-ISOMAP with multiple out-of-sample (OOS) transforms.

The training stage preserves the original Supervised K-ISOMAP formulation:
- k-NN connectivity graph
- local tangent spaces
- supervised edge reweighting with min(delta) / sum(delta)
- Floyd-Warshall geodesics
- classical MDS eigendecomposition

OOS transforms implemented without using labels of unseen samples:
- geodesic: Isomap-style attachment to the learned training graph
- krr:      Kernel Ridge Regression, X_train -> Z_train
- local:    weighted local interpolation in Z_train
- rbf:      scipy RBF interpolation, X_train -> Z_train

Preprocessing and scaling are handled outside this estimator.
"""

import numpy as np
import scipy as sp
import networkx as nx

from numpy.linalg import norm
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class SupervisedKIsomap(TransformerMixin, BaseEstimator):
    """Supervised K-ISOMAP with interchangeable OOS extensions."""

    _VALID_TRANSFORMS = {"geodesic", "krr", "local", "rbf"}

    def __init__(
        self,
        n_neighbors=5,
        n_components=2,
        transform_method="geodesic",
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.transform_method = transform_method

    #
    # Supervised K-ISOMAP training stage
    #
    def fit(self, X, y):
        """Fit Supervised K-ISOMAP on labeled training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            y = np.ravel(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        n, m = X.shape

        if not isinstance(self.n_neighbors, (int, np.integer)):
            raise TypeError("n_neighbors must be an integer.")
        if self.n_neighbors < 1 or self.n_neighbors >= n:
            raise ValueError(
                f"n_neighbors must satisfy 1 <= n_neighbors < n_samples; "
                f"got n_neighbors={self.n_neighbors}, n_samples={n}."
            )

        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer.")
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")

        if self.transform_method not in self._VALID_TRANSFORMS:
            raise ValueError(
                f"Unknown transform_method={self.transform_method!r}. "
                f"Valid options are {sorted(self._VALID_TRANSFORMS)}."
            )

        self.X_fit_ = X.copy()
        self.y_fit_ = y.copy()
        self.n_samples_fit_ = n
        self.n_features_in_ = m

        # Matrix to store tangent spaces
        matriz_pcs = np.zeros((n, m, m))

        # Generate KNN graph
        knn_graph = kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors,
            mode="connectivity",
        )
        A = knn_graph.toarray()

        # Compute local tangent spaces
        for i in range(n):
            vizinhos = A[i, :]
            indices = vizinhos.nonzero()[0]

            if len(indices) == 0:
                matriz_pcs[i, :, :] = np.eye(m)
            else:
                amostras = X[indices]
                cov = np.cov(amostras.T)

                # Prevent eig() on a scalar for 1D data.
                if np.ndim(cov) == 0:
                    cov = np.array([[cov]], dtype=float)

                v, w = np.linalg.eig(cov)
                ordem = v.argsort()
                maiores_autovetores = w[:, ordem[::-1]]
                matriz_pcs[i, :, :] = maiores_autovetores

        # Supervised edge reweighting
        B_edges = A.copy()

        for i in range(n):
            for j in range(n):
                if B_edges[i, j] > 0:
                    delta = norm(
                        matriz_pcs[i, :, :] - matriz_pcs[j, :, :],
                        axis=0,
                    )
                    if y[i] == y[j]:
                        B_edges[i, j] = min(delta)
                    else:
                        B_edges[i, j] = sum(delta)

        # Geodesic distances
        G = nx.from_numpy_array(B_edges)
        D = np.asarray(nx.floyd_warshall_numpy(G), dtype=float)

        # Classical MDS centering
        H = np.eye(n, n) - (1 / n) * np.ones((n, n))
        B_mds = -0.5 * H.dot(D**2).dot(H)

        # Sanitization behavior
        finite_non_inf = B_mds[B_mds != np.inf]
        if finite_non_inf.size == 0:
            raise ValueError(
                "The Supervised K-ISOMAP MDS matrix contains no finite values. "
                "Check graph connectivity and n_neighbors."
            )

        maximo = np.nanmax(finite_non_inf)
        B_mds[np.isnan(B_mds)] = 0
        B_mds[np.isinf(B_mds)] = maximo

        # Eigendecomposition
        lambdas, alphas = sp.linalg.eigh(B_mds)
        indices = lambdas.argsort()[::-1]
        lambdas_all = np.real(lambdas[indices])
        alphas_all = np.real(alphas[:, indices])

        d = min(self.n_components, len(lambdas_all))
        selected_lambdas = lambdas_all[:d]
        selected_alphas = alphas_all[:, :d]

        # Coordinate construction
        self.embedding_ = selected_alphas * np.sqrt(selected_lambdas)

        # State required for OOS transforms
        self.tangent_spaces_ = matriz_pcs
        self.connectivity_matrix_ = A
        self.edge_weight_matrix_ = B_edges
        self.dist_matrix_ = D
        self.mds_matrix_ = B_mds

        self.eigenvalues_all_ = lambdas_all
        self.eigenvectors_all_ = alphas_all
        self.eigenvalues_ = selected_lambdas
        self.eigenvectors_ = selected_alphas
        self.n_components_ = d

        # Quantities for classical-MDS OOS projection
        D2 = D**2
        self._train_d2_column_mean_ = np.mean(D2, axis=0)
        self._train_d2_grand_mean_ = np.mean(D2)

        # Shared NN structure for geodesic/local transforms
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.nbrs_.fit(self.X_fit_)

        # Lazy OOS estimators
        self._krr_mapper_ = None
        self._krr_params_ = None
        self._rbf_mapper_ = None
        self._rbf_params_ = None

        return self

    def fit_transform(self, X, y):
        """Fit Supervised K-ISOMAP and return the training embedding."""
        self.fit(X, y)
        return self.embedding_

    #
    # Generic transform dispatcher
    #
    def transform(self, X, method=None, **kwargs):
        """Project unseen samples using one selected OOS strategy."""
        check_is_fitted(self, attributes=["embedding_", "dist_matrix_", "nbrs_"])

        method = self.transform_method if method is None else method

        if method == "geodesic":
            return self.transform_geodesic(X, **kwargs)
        if method == "krr":
            return self.transform_krr(X, **kwargs)
        if method == "local":
            return self.transform_local(X, **kwargs)
        if method == "rbf":
            return self.transform_rbf(X, **kwargs)

        raise ValueError(
            f"Unknown transform method {method!r}. "
            f"Valid options are {sorted(self._VALID_TRANSFORMS)}."
        )

    def transform_all(self, X, methods=None, method_kwargs=None):
        """Apply multiple OOS transforms to the same unseen dataset."""
        if methods is None:
            methods = ("geodesic", "krr", "local", "rbf")
        if method_kwargs is None:
            method_kwargs = {}

        outputs = {}
        for method in methods:
            kwargs = method_kwargs.get(method, {})
            outputs[method] = self.transform(X, method=method, **kwargs)
        return outputs

    #
    # OOS method 1: Isomap-style geodesic extension
    #
    def transform_geodesic(self, X, n_neighbors=None, entry_scale=1.0):
        """
        Isomap-style OOS extension using the learned Supervised K-ISOMAP geodesic graph.

        For an unseen point x*, approximate its distance to fitted point x_j as

            D(x*, x_j) = min_i [
                entry_scale * ||x* - x_i||_2 + D_SK(x_i, x_j)
            ]

        where x_i ranges over nearest training neighbors.

        No unseen/test labels are used.
        """
        check_is_fitted(self, attributes=["embedding_", "dist_matrix_", "nbrs_"])
        X = self._validate_query_X(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors < 1 or n_neighbors > self.n_samples_fit_:
            raise ValueError("n_neighbors must satisfy 1 <= n_neighbors <= n_samples_fit_.")

        distances, indices = self.nbrs_.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)

        D_new = np.zeros((X.shape[0], self.n_samples_fit_), dtype=float)

        for i in range(X.shape[0]):
            candidate_paths = (
                self.dist_matrix_[indices[i]]
                + (entry_scale * distances[i])[:, None]
            )
            D_new[i] = np.min(candidate_paths, axis=0)

        return self._classical_mds_oos(D_new)

    #
    # OOS method 2: Kernel Ridge Regression
    #
    def transform_krr(
        self,
        X,
        alpha=1.0,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1.0,
    ):
        """Multi-output Kernel Ridge mapping X_train -> Z_train."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        X = self._validate_query_X(X)

        params = {
            "alpha": alpha,
            "kernel": kernel,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
        }

        if self._krr_mapper_ is None or self._krr_params_ != params:
            self._krr_mapper_ = KernelRidge(
                alpha=alpha,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
            )
            self._krr_mapper_.fit(self.X_fit_, self.embedding_)
            self._krr_params_ = params.copy()

        return np.asarray(self._krr_mapper_.predict(X), dtype=float)

    #
    # OOS method 3: Local interpolation
    #
    def transform_local(
        self,
        X,
        n_neighbors=None,
        weights="distance",
        eps=1e-12,
        power=1.0,
    ):
        """Weighted interpolation from neighboring training embeddings."""
        check_is_fitted(self, attributes=["embedding_", "nbrs_"])
        X = self._validate_query_X(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors < 1 or n_neighbors > self.n_samples_fit_:
            raise ValueError(
                "n_neighbors must satisfy 1 <= n_neighbors <= n_samples_fit_."
            )

        distances, indices = self.nbrs_.kneighbors(
            X,
            n_neighbors=n_neighbors,
            return_distance=True,
        )

        Z = np.zeros((X.shape[0], self.n_components_), dtype=float)

        for i in range(X.shape[0]):
            neigh_Z = self.embedding_[indices[i]]
            neigh_d = distances[i]

            exact = np.where(neigh_d <= eps)[0]
            if exact.size > 0:
                Z[i] = neigh_Z[exact[0]]
                continue

            if weights == "uniform":
                w = np.ones_like(neigh_d, dtype=float)
            elif weights == "distance":
                w = 1.0 / np.power(neigh_d + eps, power)
            else:
                raise ValueError("weights must be 'distance' or 'uniform'.")

            w /= np.sum(w)
            Z[i] = np.sum(neigh_Z * w[:, None], axis=0)

        return Z

    #
    # OOS method 4: RBF interpolation
    #
    def transform_rbf(
        self,
        X,
        kernel="thin_plate_spline",
        epsilon=None,
        smoothing=0.0,
        neighbors=None,
        degree=None,
    ):
        """RBF interpolation mapping X_train -> Z_train."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        X = self._validate_query_X(X)

        params = {
            "kernel": kernel,
            "epsilon": epsilon,
            "smoothing": smoothing,
            "neighbors": neighbors,
            "degree": degree,
        }

        if self._rbf_mapper_ is None or self._rbf_params_ != params:
            kwargs = {
                "kernel": kernel,
                "smoothing": smoothing,
                "neighbors": neighbors,
            }
            if epsilon is not None:
                kwargs["epsilon"] = epsilon
            if degree is not None:
                kwargs["degree"] = degree

            self._rbf_mapper_ = RBFInterpolator(
                self.X_fit_,
                self.embedding_,
                **kwargs,
            )
            self._rbf_params_ = params.copy()

        return np.asarray(self._rbf_mapper_(X), dtype=float)

    def _classical_mds_oos(self, D_new):
        """
        Project unseen-to-training distances into the existing classical-MDS
        coordinate system using the standard Gower extension.
        """
        D_new = np.asarray(D_new, dtype=float)

        if D_new.ndim != 2:
            raise ValueError("D_new must be a 2D array.")
        if D_new.shape[1] != self.n_samples_fit_:
            raise ValueError(
                "D_new must contain one distance to each fitted sample."
            )

        lambdas = np.asarray(self.eigenvalues_, dtype=float)
        tol = (
            np.finfo(float).eps
            * max(1.0, np.max(np.abs(lambdas)))
            * 100
        )

        if np.any(lambdas <= tol):
            bad = lambdas[lambdas <= tol]
            raise ValueError(
                "Cannot perform classical-MDS OOS projection because one or "
                f"more selected eigenvalues are non-positive/near-zero: {bad}. "
                "Use fewer n_components or another OOS transform."
            )

        d2 = D_new**2
        new_row_mean = np.mean(d2, axis=1, keepdims=True)

        B_new = -0.5 * (
            d2
            - self._train_d2_column_mean_[None, :]
            - new_row_mean
            + self._train_d2_grand_mean_
        )

        return (
            B_new
            @ self.eigenvectors_
            / np.sqrt(self.eigenvalues_)[None, :]
        )

    def _validate_query_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted Supervised K-ISOMAP "
                f"expects {self.n_features_in_}."
            )

        return X
