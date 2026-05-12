#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised K-ISOMAP for dimensionality-reduction-based metric learning.

Python code to reproduce the experimental results reported in the paper.

This script evaluates Supervised K-ISOMAP against representative
dimensionality reduction, metric learning, manifold learning, and
discriminant analysis baselines.

Methods:

    Linear Supervised Methods:
        Variance-Preserving Projection Methods:
            Supervised PCA

        Discriminant Linear Models:
            LDA (Linear Discriminant Analysis)

        Latent-Variable Covariance-Maximization Methods:
            PLS Regression

        Linear Metric Learning Methods:
            LMNN (Large Margin Nearest Neighbor)
            NCA (Neighborhood Components Analysis)

    Nonlinear and Graph-Based Methods:
        Stochastic Manifold Learning:
            UMAP / Supervised UMAP

        Graph-Based Discriminant Manifold Methods:
            LFDA (Local Fisher Discriminant Analysis)
            SLPP (Supervised Locality Preserving Projection)
            LDE (Local Discriminant Embedding)
            MFA (Marginal Fisher Analysis)

        Geometry-Based Manifold Learning:
            ISOMAP

    Proposed Method:
        Geometry-Aware Supervised Graph-Based Metric Learning:
            Supervised K-ISOMAP

Scores:
    Neighborhood and Structure Preservation:
        Trustworthiness
        Continuity
        k-NN Preservation
        Sammon Stress

    Distribution-Based Scores:
        Hellinger Distance
        Jensen-Shannon Distance
        Distribution Scores

    Classification-Based Scores:
        Average Accuracy
        Average F1 Score

    Clustering-Based Scores:
        Silhouette Coefficient

Classifiers:
    KNN (k-Nearest Neighbors)
    SVM (Support Vector Machine)
    Naive Bayes
    Decision Tree
    Quadratic Discriminant Analysis
    MLP Classifier
    Gaussian Process Classifier
    Random Forest Classifier
"""

# Imports
import time
import warnings
import umap
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Force backend to save files, without graphical interface
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from numpy import sqrt
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import datetime
from metric_learn import NCA, LMNN, LFDA # pip install metric-learn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score
from collections import Counter

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    return novos_dados

# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
def SupervisedPCA(dados, labels, d):
    dados = dados.T
    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features
    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1
    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    return novos_dados

def SupervisedLPP(X, y, n_components=2, k=5):
    """
    Supervised Locality Preserving Projection
    """
    n = X.shape[0]

    # Build supervised k-NN graph (same-class neighbors only)
    knn = sknn.NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    W = np.zeros((n, n))

    for i in range(n):
        for j in indices[i]:
            if y[i] == y[j]:
                W[i, j] = 1
                W[j, i] = 1

    D = np.diag(W.sum(axis=1))
    L = D - W

    eps = 1e-6
    XTX = X.T @ D @ X + eps * np.eye(X.shape[1])
    XTLX = X.T @ L @ X

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = sp.linalg.eigh(XTLX, XTX)

    idx = np.argsort(eigvals)
    Wproj = eigvecs[:, idx[:n_components]]

    return sanitize_output(X @ Wproj)

def LDE(X, y, n_components=2, k=5):
    """
    Local Discriminant Embedding (LDE)
    """

    n = X.shape[0]

    # kNN global
    knn = sknn.NearestNeighbors(n_neighbors=k+1)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    Ww = np.zeros((n, n))  # within-class graph
    Wb = np.zeros((n, n))  # between-class graph

    for i in range(n):
        #for j in indices[i]:
        for j in indices[i][1:]:
            if y[i] == y[j]:
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
    A = X.T @ Lb @ X
    B = X.T @ Lw @ X + eps * np.eye(X.shape[1])

    eigvals, eigvecs = sp.linalg.eigh(A, B)

    idx = np.argsort(eigvals)[::-1]
    Wproj = eigvecs[:, idx[:n_components]]

    return sanitize_output(X @ Wproj)

def MFA(X, y, n_components=2, k1=5, k2=5):
    """
    Marginal Fisher Analysis (MFA)
    """

    n = X.shape[0]

    knn = NearestNeighbors(n_neighbors=max(k1, k2) + 1)
    
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    Ww = np.zeros((n, n))
    Wb = np.zeros((n, n))

    for i in range(n):
        count_w = 0
        count_b = 0

        for j in indices[i]:
            if y[i] == y[j] and count_w < k1:
                Ww[i, j] = 1
                Ww[j, i] = 1
                count_w += 1

            if y[i] != y[j] and count_b < k2:
                Wb[i, j] = 1
                Wb[j, i] = 1
                count_b += 1

    Dw = np.diag(Ww.sum(axis=1))
    Db = np.diag(Wb.sum(axis=1))

    Lw = Dw - Ww
    Lb = Db - Wb

    eps = 1e-6
    A = X.T @ Lb @ X
    B = X.T @ Lw @ X + eps * np.eye(X.shape[1])

    eigvals, eigvecs = sp.linalg.eigh(A, B)

    idx = np.argsort(eigvals)[::-1]
    Wproj = eigvecs[:, idx[:n_components]]

    return sanitize_output(X @ Wproj)

def sanitize_output(X_projected):
    """remove NaNs/Infs"""
    X_projected = np.asanyarray(X_projected).real
    X_projected = np.nan_to_num(X_projected, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if X_projected.shape[0] > 1:
        scaler = StandardScaler()
        X_projected = scaler.fit_transform(X_projected)
    return X_projected

def add_gaussian_noise(data, noise_level=0.01, random_state=None):
    """
    Adds Gaussian noise to the data.

    Parameters
    ----------
    data : ndarray
        Input data matrix (samples × features).

    noise_level : float, optional (default=0.01)
        Noise intensity defined as a fraction of the feature-wise standard
        deviation (e.g., 0.05 corresponds to 5% of the data standard deviation).

    random_state : int or None, optional (default=None)
        Seed for the random number generator. Ensures reproducibility of the
        generated noise when provided. If None, randomness is not fixed.

    Returns
    -------
    ndarray
        Data with additive Gaussian noise.
    """
    if noise_level <= 0:
        return data
    
    rng = np.random.default_rng(random_state)
    sigma = noise_level * np.std(data, axis=0)
    noise = rng.normal(0, sigma, data.shape)
    return data + noise

# Supervised K-ISOMAP implementation
def SK_Isomap(dados, k, d_variation, target):
    """ 
        Optimized version that returns multiple embeddings for different values ​​of d. 
        d_variation: Variation of integers to parameter d
    """
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the tangent spaces
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Eigenvectors in the columns
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     
            # Projection matrix
            Wpca = maiores_autovetores  # Eigenvectors in the columns
            matriz_pcs[i, :, :] = Wpca
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                if target[i] == target[j]:
                    B[i, j] = min(delta)
                else:
                    B[i, j] = sum(delta)
    # Computes geodesic distances in B
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])  
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas_all = lambdas[indices]
    alphas_all = alphas[:, indices]
    
    outputs = {}
    for d_item in d_variation:
        # Ensure that we don't try to encompass more dimensions than exist
        d = min(d_item, len(lambdas_all))
        
        # Select the d largest eigenvectors
        lambdas = lambdas_all[0:d]
        alphas = alphas_all[:, 0:d]
        # Computes the intrinsic coordinates
        outputs[d] = alphas*np.sqrt(lambdas)
        
    return outputs

def sammon_stress(X_high, X_low):
    d_high = pairwise_distances(X_high)
    d_low = pairwise_distances(X_low)
    
    # Avoid division by zero on the diagonal
    mask = ~np.eye(d_high.shape[0], dtype=bool)
    
    eps = 1e-12
    numerator = ((d_high[mask] - d_low[mask])**2 / (d_high[mask] + eps)).sum()
    denominator = d_high[mask].sum()
    return numerator / denominator

def knn_preservation(X_high, X_low, k=15):
    neigh_high = NearestNeighbors(n_neighbors=k+1).fit(X_high).kneighbors(return_distance=False)
    neigh_low = NearestNeighbors(n_neighbors=k+1).fit(X_low).kneighbors(return_distance=False)
    
    intersect = [len(np.intersect1d(h, l)) for h, l in zip(neigh_high, neigh_low)]
    return np.mean(intersect) / k

def myContinuity(X_high, X_low, k=15):
    n = X_high.shape[0]

    neigh_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    neigh_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)

    rank_high = neigh_high.kneighbors(return_distance=False)
    rank_low = neigh_low.kneighbors(return_distance=False)

    score = 0
    for i in range(n):
        U = set(rank_high[i][1:]) - set(rank_low[i][1:])
        for j in U:
            r = np.where(rank_low[i]==j)[0]
            if len(r)>0:
                score += r[0] - k

    norm = 2/(n*k*(2*n-3*k-1))
    return 1 - norm*score

def continuity(X_high, X_low, k=15):

    n = X_high.shape[0]

    D_high = pairwise_distances(X_high)
    D_low  = pairwise_distances(X_low)

    order_high = np.argsort(D_high, axis=1)
    order_low  = np.argsort(D_low, axis=1)

    rank_high = np.empty_like(order_high)
    rank_low  = np.empty_like(order_low)

    rank_high[np.arange(n)[:, None], order_high] = np.arange(n)
    rank_low[np.arange(n)[:, None], order_low] = np.arange(n)

    penalty = 0.0

    for i in range(n):

        neighbors_high = order_high[i, 1:k+1]  # exclude self

        for j in neighbors_high:
            if rank_low[i, j] > k:
                penalty += rank_low[i, j] - k

    norm = 2.0 / (n * k * (2*n - 3*k - 1))

    return 1.0 - norm * penalty

def neighborhood_distribution(X, k=15):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, idx = nbrs.kneighbors(X)

    # remove self neighbor
    dist = dist[:,1:]

    # convert distances → similarity
    sigma = np.mean(dist, axis=1, keepdims=True)
    sim = np.exp(-dist**2/(2*sigma**2))

    # normalize rows (probability distribution)
    prob = sim / np.sum(sim, axis=1, keepdims=True)

    return prob

def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)

def distribution_scores(X_high, X_low, k=15):

    P = neighborhood_distribution(X_high, k)
    Q = neighborhood_distribution(X_low, k)

    js_scores = []
    hell_scores = []

    for i in range(P.shape[0]):
        js_scores.append(jensenshannon(P[i], Q[i])**2)
        hell_scores.append(hellinger_distance(P[i], Q[i]))

    return np.mean(js_scores), np.mean(hell_scores)

# Train and test eight different supervised classifiers
def Classification(dados, target, method):
    acc_list = []
    f1_score_list = []
    
    counts = Counter(target)
    strat = target if min(counts.values()) >= 2 else None

    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real, target, test_size=0.5, random_state=42, stratify=strat)
    
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train) 
    pred = neigh.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    pred = svm.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Quadratic Discriminant 
    counts = pd.Series(y_train).value_counts()
    valid_classes = counts[counts > 1].index  # mantém classes com >1 amostra
    mask = np.isin(y_train, valid_classes)
    X_train_filtered = X_train[mask]
    y_train_filtered = y_train[mask]
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_filtered, y_train_filtered)
    pred = qda.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    mpl.fit(X_train, y_train)
    pred = mpl.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    pred = gpc.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    acc_list.append(acc)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_score_list.append(f1_macro)
    
    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados, target, metric='euclidean')
    
    # Computes the average accuracy
    acc_average = sum(acc_list)/len(acc_list)
    acc_max = max(acc_list)
    
    # Computes the average f1 score
    f1_average = sum(f1_score_list)/len(f1_score_list)
    f1_max = max(f1_score_list)
    
    print()
    print('Maximum balanced accuracy for %s features: %f' %(method, acc_max))
    print('Maximum macro F1-score for %s features: %f' % (method, f1_max))
    print()
    return [sc, acc_average, acc_max, f1_average, f1_max]

# Plot the scatterplots dor the 2D output data
def PlotaDados(dados, labels, metodo, img_title):
    nclass = len(np.unique(labels))
    if metodo == 'LDA':
        if nclass == 2:
            return -1
    # Encode the labels as integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to numbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    rotulos = np.array(rotulos)
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'green', 'black', 'cyan', 'magenta', 'orange', 'darkkhaki', 'brown', 'purple', 'salmon']
    
    
    # --------------------------------------------------
    # R^d -> R^2 projection (visualization step)
    # --------------------------------------------------
    if dados.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        dados_centered = dados - np.mean(dados, axis=0)
        dados_plot = pca.fit_transform(dados_centered)

    elif dados.shape[1] == 2:
        dados_plot = dados

    elif dados.shape[1] == 1:
        # artificial 2D embedding
        dados_plot = np.column_stack(
            (np.arange(dados.shape[0]), dados[:, 0])
        )
    else:
        raise ValueError("Input data must have at least 1 dimension.")
    
    if dados.shape[1] > 2:
        var_explained = pca.explained_variance_ratio_.sum()
    else:
        var_explained = 1.0
    
    
    plt.figure()
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        #plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, alpha=0.4, marker='.')
        if dados_plot.shape[1] >= 2:  # tem pelo menos 2 colunas
            plt.scatter(dados_plot[indices, 0], dados_plot[indices, 1], c=cor, alpha=0.4, marker='.')
        elif dados_plot.shape[1] == 1:  # só 1 coluna
            plt.scatter(range(len(indices)), dados_plot[indices, 0], c=cor, alpha=0.4, marker='.')
        else:
            raise ValueError("O array 'dados_plot' não tem colunas suficientes para o scatter.")
    
    nome_arquivo = metodo + '.png'
    save_path = os.path.join(img_dir, f"{nome_arquivo}")
    
    plt.title(img_title +' clusters')
    #plt.title(f"{img_title} clusters (PCA var={var_explained:.2f})")
    #plt.axis('equal')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')

def min_class_size(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.min()

def plot_comparison_image(embeddings_dict, labels, dataset_name, save_dir,
                          filename=None, dpi=600):
    """
    Generate a high-resolution comparison figure with 3 columns and 2 rows.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary with 2D embeddings. Required keys:
        "Supervised K-ISOMAP", "Supervised UMAP", "NCA",
        "LFDA", "LMNN", "ISOMAP"

    labels : ndarray
        Class labels.

    dataset_name : str
        Dataset name to appear in subplot titles.

    save_dir : str
        Directory to save the image.

    filename : str or None, optional
        Output PNG filename. If None, a default name is used.

    dpi : int, optional
        Resolution for saving the PNG figure.
    """
    method_order = [
        "Supervised K-ISOMAP",
        "Supervised UMAP",
        "NCA",
        "LFDA",
        "LMNN",
        "ISOMAP"
    ]

    nclass = len(np.unique(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cmap = plt.cm.get_cmap("tab10", max(nclass, 10))
    unique_labels = np.unique(labels)

    for ax, method_name in zip(axes, method_order):
        dados = np.asarray(embeddings_dict[method_name])

        # Guarantee a 2D visualization
        if dados.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            dados_centered = dados - np.mean(dados, axis=0)
            dados_plot = pca.fit_transform(dados_centered)
        elif dados.shape[1] == 2:
            dados_plot = dados
        elif dados.shape[1] == 1:
            dados_plot = np.column_stack((np.arange(dados.shape[0]), dados[:, 0]))
        else:
            raise ValueError(f"{method_name} embedding must have at least 1 dimension.")

        for i, cls in enumerate(unique_labels):
            idx = np.where(labels == cls)[0]
            ax.scatter(
                dados_plot[idx, 0],
                dados_plot[idx, 1],
                s=12,
                alpha=0.5,
                color=cmap(i),
                marker="."
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{method_name} {dataset_name} clusters", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    if filename is None:
        filename = f"{dataset_name}_comparison_6methods.png"

    png_path = os.path.join(save_dir, filename)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
def plot_comparison_image_fixed(embeddings_dict, labels, dataset_name, save_dir,
                               filename=None, dpi=300):
    """
    Improved visualization with consistent axis limits and visible coordinates.
    """

    method_order = [
        "Supervised K-ISOMAP",
        "Supervised UMAP",
        "NCA",
        "LFDA",
        "LMNN",
        "ISOMAP"
    ]

    nclass = len(np.unique(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cmap = plt.cm.get_cmap("tab10", max(nclass, 10))
    unique_labels = np.unique(labels)

    all_points = []

    processed_embeddings = {}

    # ===============================
    # First pass → preprocess + store
    # ===============================
    for method_name in method_order:
        dados = np.asarray(embeddings_dict[method_name])

        if dados.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            dados_centered = dados - np.mean(dados, axis=0)
            dados_plot = pca.fit_transform(dados_centered)
        elif dados.shape[1] == 2:
            dados_plot = dados
        elif dados.shape[1] == 1:
            dados_plot = np.column_stack((np.arange(dados.shape[0]), dados[:, 0]))
        else:
            raise ValueError(f"{method_name} embedding must have at least 1 dimension.")

        processed_embeddings[method_name] = dados_plot
        all_points.append(dados_plot)

    # ===============================
    # Global axis limits
    # ===============================
    all_points = np.vstack(all_points)
    x_min, x_max = np.percentile(all_points[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_points[:, 1], [1, 99])

    # small padding
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)

    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    # ===============================
    # Plot
    # ===============================
    for ax, method_name in zip(axes, method_order):
        dados_plot = processed_embeddings[method_name]

        for i, cls in enumerate(unique_labels):
            idx = np.where(labels == cls)[0]
            ax.scatter(
                dados_plot[idx, 0],
                dados_plot[idx, 1],
                s=10,
                alpha=0.7,
                color=cmap(i),
                marker="."
            )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_title(method_name, fontsize=12)
        #ax.set_xlabel("Component 1", fontsize=10)
        #ax.set_ylabel("Component 2", fontsize=10)

        #ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"{dataset_name} - Embedding Comparison", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if filename is None:
        filename = f"{dataset_name}_comparison.png"

    png_path = os.path.join(save_dir, filename)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
datasets = [
    ("zoo", 1),
    # ("glass", 1),
    # ("ecoli", 1),
    # ("balance-scale", 1),
    # ("energy-efficiency", 1),
    # ("vehicle", 1),
    # ("vowel", 1),
    # ("collins", 1),
    # ("cnae-9", 1),
    # ("wine-quality-red", 1),
    # ("one-hundred-plants-texture", 1),
    # ("one-hundred-plants-shape", 1),
    # ("car-evaluation", 1),
    # ("digits", 1),
    # ("mfeat-karhunen", 1),
    # ("mfeat-pixel", 1),
    # ("pendigits", 1),
    # ("Indian_pines", 1),
    # ("GesturePhaseSegmentationProcessed", 1),
    # ("artificial-characters", 1),
    # ("thyroid-dis", 1),
    # ("led24", 1),
    # ("nursery", 1),
    # ("eye_movements", 1),
    # ("MNIST_784", 1),
    # ("CIFAR_10_small", 1),
    # ("wine-quality-white", 1),
    # ("waveform-5000", 1),
    # ("wall-robot-navigation", 1),
    # ("optdigits", 1),
    # ("satimage", 1),
    # ("tic-tac-toe", 1),
    # ("diabetes", 1),
    # ("grub-damage", 2),
    # ("banknote-authentication", 1),
    # ("ionosphere", 1)
]

all_results = []
#num_runs = 10
#d_variation = [2, 3, 4, 5, 10, 20, 30, 50]
num_runs = 1
d_variation = [4]
noise_levels = [0]

base_dir = os.path.dirname(os.path.abspath(__file__))
files_dir = os.path.join(base_dir, "files")
os.makedirs(files_dir, exist_ok=True)
csv_file = os.path.join(files_dir, "results_all_datasets.csv")

img_dir = os.path.join(base_dir, "img")
os.makedirs(img_dir, exist_ok=True)

for current_noise in noise_levels:
    for run in range(1, num_runs + 1):
        
        for dataset_name, v in datasets:
            try:
                print(f"\nProcessing {dataset_name}...")
                
                # data base file init
                base_line_execution = {}
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if dataset_name == 'digits':
                    X = skdata.load_digits()
                else:
                    X = skdata.fetch_openml(name=dataset_name, version=v)
            
                dados = X['data']
                target = X['target']
                
                if 'details' in X.keys():    
                    if X['details']['name'] == 'GesturePhaseSegmentationProcessed':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
                    if X['details']['name'] == 'Indian_pines':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
                    if X['details']['name'] == 'CIFAR_10_small':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
                        dados = myPCA(dados, 30).real.T
                    elif X['details']['name'] == 'pendigits':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
                    elif X['details']['name'] == 'artificial-characters':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
                    elif X['details']['name'] == 'nursery':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)            
                    elif X['details']['name'] == 'eye_movements':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.3, random_state=42)
                    elif X['details']['name'] == 'mnist_784':
                        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
                        dados = myPCA(dados, 50).real.T
                
                n = dados.shape[0]
                m = dados.shape[1]
                c = len(np.unique(target))

                # Treat categorical features
                if not isinstance(dados, np.ndarray):
                    cat_cols = dados.select_dtypes(['category']).columns
                    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
                    dados = dados.to_numpy()
                le = LabelEncoder()
                le.fit(target)
                target = le.transform(target)

                # Number of neighbors
                nn = round(sqrt(n))     # Number of neighbors = square root of n
                print()
                print('Number of samples (n): ', n)
                print('Number of features (m): ', m)
                print('Number of classes (c): ', c)
                print('Number of Neighbors in k-NN graph (k): ', nn)
                print()
                # print('Press enter to continue...')
                # input()
                
                # fill data base file
                base_line_execution["Timestamp"] = timestamp
                base_line_execution["dataset_name"] = dataset_name
                base_line_execution["run_time"] = run
                base_line_execution["n_samples"] = n
                base_line_execution["n_features"] = m
                base_line_execution["n_classes"] = c
                base_line_execution["nn"] = nn
                base_line_execution["noise_level"] = current_noise

                # Data standardization (to deal with variables having different units/scales)
                dados_clean = preprocessing.scale(dados)
                
                dados_noisy = add_gaussian_noise(dados_clean, current_noise, run)
                print(f"Run {run}: Processando {dataset_name} com {current_noise*100:.0f}% de ruído gaussiano...")

                
                #############
                ############# PLS
                #############
                start = time.time()
                model = PLSRegression(n_components=2)
                dados_pls, _ = model.fit_transform(dados_noisy, y=target)
                end = time.time()
                base_line_execution["PLS"] = end - start
                print(f'PLS time: {base_line_execution["PLS"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_pls, n_neighbors=15)
                c_score = continuity(dados_clean, dados_pls)
                my_c_score = myContinuity(dados_clean, dados_pls)
                knn_preserv = knn_preservation(dados_clean, dados_pls)
                strees = sammon_stress(dados_clean, dados_pls)
                js, hell = distribution_scores(dados_clean, dados_pls)
                base_line_execution["PLS_t_score"] = t_score
                base_line_execution["PLS_c_score"] = c_score
                base_line_execution["PLS_my_c_score"] = my_c_score
                base_line_execution["PLS_knn_preserv"] = knn_preserv
                base_line_execution["PLS_strees"] = strees
                base_line_execution["PLS_js"] = js
                base_line_execution["PLS_hell"] = hell
                base_line_execution["PLS_js_sim"] = 1 - js
                base_line_execution["PLS_hell_sim"] = 1 - hell
                
                
                #############
                ############# UMAP
                #############
                start = time.time()
                model = umap.UMAP(n_components=2, random_state=42)
                dados_umap = model.fit_transform(dados_noisy, y=target)
                end = time.time()
                base_line_execution["UMAP"] = end - start
                print(f'UMAP time: {base_line_execution["UMAP"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_umap, n_neighbors=15)
                c_score = continuity(dados_clean, dados_umap)
                my_c_score = myContinuity(dados_clean, dados_umap)
                knn_preserv = knn_preservation(dados_clean, dados_umap)
                strees = sammon_stress(dados_clean, dados_umap)
                js, hell = distribution_scores(dados_clean, dados_umap)
                base_line_execution["UMAP_t_score"] = t_score
                base_line_execution["UMAP_c_score"] = c_score
                base_line_execution["UMAP_my_c_score"] = my_c_score
                base_line_execution["UMAP_knn_preserv"] = knn_preserv
                base_line_execution["UMAP_strees"] = strees
                base_line_execution["UMAP_js"] = js
                base_line_execution["UMAP_hell"] = hell
                base_line_execution["UMAP_js_sim"] = 1 - js
                base_line_execution["UMAP_hell_sim"] = 1 - hell

                #############
                ############# Supervised PCA
                #############
                start = time.time()
                dados_suppca = SupervisedPCA(dados_noisy, target, 2)
                dados_suppca = dados_suppca.T
                dados_suppca = np.real(dados_suppca)
                end = time.time()
                base_line_execution["PCA"] = end - start
                print(f'PCA time: {base_line_execution["PCA"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_suppca, n_neighbors=15)
                c_score = continuity(dados_clean, dados_suppca)
                my_c_score = myContinuity(dados_clean, dados_suppca)
                knn_preserv = knn_preservation(dados_clean, dados_suppca)
                strees = sammon_stress(dados_clean, dados_suppca)
                js, hell = distribution_scores(dados_clean, dados_suppca)
                base_line_execution["PCA_t_score"] = t_score
                base_line_execution["PCA_c_score"] = c_score
                base_line_execution["PCA_my_c_score"] = my_c_score
                base_line_execution["PCA_knn_preserv"] = knn_preserv
                base_line_execution["PCA_strees"] = strees
                base_line_execution["PCA_js"] = js
                base_line_execution["PCA_hell"] = hell
                base_line_execution["PCA_js_sim"] = 1 - js
                base_line_execution["PCA_hell_sim"] = 1 - hell

                #############
                ############# LDA
                #############
                start = time.time()
                if c > 2:
                    model = LinearDiscriminantAnalysis(n_components=2)
                else:
                    model = LinearDiscriminantAnalysis(n_components=1)
                dados_lda = model.fit_transform(dados_noisy, target)
                end = time.time()
                base_line_execution["LDA"] = end - start
                print(f'LDA time: {base_line_execution["LDA"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_lda, n_neighbors=15)
                c_score = continuity(dados_clean, dados_lda)
                my_c_score = myContinuity(dados_clean, dados_lda)
                knn_preserv = knn_preservation(dados_clean, dados_lda)
                strees = sammon_stress(dados_clean, dados_lda)
                js, hell = distribution_scores(dados_clean, dados_lda)
                base_line_execution["LDA_t_score"] = t_score
                base_line_execution["LDA_c_score"] = c_score
                base_line_execution["LDA_my_c_score"] = my_c_score
                base_line_execution["LDA_knn_preserv"] = knn_preserv
                base_line_execution["LDA_strees"] = strees
                base_line_execution["LDA_js"] = js
                base_line_execution["LDA_hell"] = hell
                base_line_execution["LDA_js_sim"] = 1 - js
                base_line_execution["LDA_hell_sim"] = 1 - hell
                
                #############
                ############# NCA
                #############
                start = time.time()
                model_nca = NCA(n_components=2, random_state=42)
                dados_nca = model_nca.fit_transform(dados_noisy, target)
                dados_nca = sanitize_output(dados_nca)
                base_line_execution["NCA"] = time.time() - start
                print(f'NCA time: {base_line_execution["NCA"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_nca, n_neighbors=15)
                c_score = continuity(dados_clean, dados_nca)
                my_c_score = myContinuity(dados_clean, dados_nca)
                knn_preserv = knn_preservation(dados_clean, dados_nca)
                strees = sammon_stress(dados_clean, dados_nca)
                js, hell = distribution_scores(dados_clean, dados_nca)
                base_line_execution["NCA_t_score"] = t_score
                base_line_execution["NCA_c_score"] = c_score
                base_line_execution["NCA_my_c_score"] = my_c_score
                base_line_execution["NCA_knn_preserv"] = knn_preserv
                base_line_execution["NCA_strees"] = strees
                base_line_execution["NCA_js"] = js
                base_line_execution["NCA_hell"] = hell
                base_line_execution["NCA_js_sim"] = 1 - js
                base_line_execution["NCA_hell_sim"] = 1 - hell
                
                #############
                ############# LMNN
                #############
                min_size = min_class_size(target)
                safe_k = max(1, min(3, min_size - 1))

                start = time.time()
                model_lmnn = LMNN(n_neighbors=safe_k)
                model_lmnn.fit(dados_noisy, target)
                dados_lmnn = model_lmnn.transform(dados_noisy)
                dados_lmnn_2d = myPCA(dados_lmnn, 2).real.T 
                dados_lmnn_2d = sanitize_output(dados_lmnn_2d)
                base_line_execution["LMNN"] = time.time() - start
                print(f'LMNN time: {base_line_execution["LMNN"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_lmnn_2d, n_neighbors=15)
                c_score = continuity(dados_clean, dados_lmnn_2d)
                my_c_score = myContinuity(dados_clean, dados_lmnn_2d)
                knn_preserv = knn_preservation(dados_clean, dados_lmnn_2d)
                strees = sammon_stress(dados_clean, dados_lmnn_2d)
                js, hell = distribution_scores(dados_clean, dados_lmnn_2d)
                base_line_execution["LMNN_t_score"] = t_score
                base_line_execution["LMNN_c_score"] = c_score
                base_line_execution["LMNN_my_c_score"] = my_c_score
                base_line_execution["LMNN_knn_preserv"] = knn_preserv
                base_line_execution["LMNN_strees"] = strees
                base_line_execution["LMNN_js"] = js
                base_line_execution["LMNN_hell"] = hell
                base_line_execution["LMNN_js_sim"] = 1 - js
                base_line_execution["LMNN_hell_sim"] = 1 - hell
                
                #############
                ############# LFDA (Local Fisher Discriminant Analysis)
                #############
                start = time.time()
                model_lfda = LFDA(n_components=2, k=nn)
                dados_lfda = model_lfda.fit_transform(dados_noisy, target)
                dados_lfda = sanitize_output(dados_lfda)
                base_line_execution["LFDA"] = time.time() - start
                print(f'LFDA time: {base_line_execution["LFDA"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_lfda, n_neighbors=15)
                c_score = continuity(dados_clean, dados_lfda)
                my_c_score = myContinuity(dados_clean, dados_lfda)
                knn_preserv = knn_preservation(dados_clean, dados_lfda)
                strees = sammon_stress(dados_clean, dados_lfda)
                js, hell = distribution_scores(dados_clean, dados_lfda)
                base_line_execution["LFDA_t_score"] = t_score
                base_line_execution["LFDA_c_score"] = c_score
                base_line_execution["LFDA_my_c_score"] = my_c_score
                base_line_execution["LFDA_knn_preserv"] = knn_preserv
                base_line_execution["LFDA_strees"] = strees
                base_line_execution["LFDA_js"] = js
                base_line_execution["LFDA_hell"] = hell
                base_line_execution["LFDA_js_sim"] = 1 - js
                base_line_execution["LFDA_hell_sim"] = 1 - hell
                
                #############
                ############# Supervised LPP
                #############
                start = time.time()
                dados_slpp = SupervisedLPP(dados_noisy, target, n_components=2, k=nn)
                base_line_execution["SLPP"] = time.time() - start
                print(f'SLPP time: {base_line_execution["SLPP"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_slpp, n_neighbors=15)
                c_score = continuity(dados_clean, dados_slpp)
                my_c_score = myContinuity(dados_clean, dados_slpp)
                knn_preserv = knn_preservation(dados_clean, dados_slpp)
                strees = sammon_stress(dados_clean, dados_slpp)
                js, hell = distribution_scores(dados_clean, dados_slpp)
                base_line_execution["SLPP_t_score"] = t_score
                base_line_execution["SLPP_c_score"] = c_score
                base_line_execution["SLPP_my_c_score"] = my_c_score
                base_line_execution["SLPP_knn_preserv"] = knn_preserv
                base_line_execution["SLPP_strees"] = strees
                base_line_execution["SLPP_js"] = js
                base_line_execution["SLPP_hell"] = hell
                base_line_execution["SLPP_js_sim"] = 1 - js
                base_line_execution["SLPP_hell_sim"] = 1 - hell

                #############
                ############# LDE
                #############
                start = time.time()
                dados_lde = LDE(dados_noisy, target, n_components=2, k=nn)
                base_line_execution["LDE"] = time.time() - start
                print(f'LDE time: {base_line_execution["LDE"]}')

                #------ scores
                t_score = trustworthiness(dados_clean, dados_lde, n_neighbors=15)
                c_score = continuity(dados_clean, dados_lde)
                my_c_score = myContinuity(dados_clean, dados_lde)
                knn_preserv = knn_preservation(dados_clean, dados_lde)
                strees = sammon_stress(dados_clean, dados_lde)
                js, hell = distribution_scores(dados_clean, dados_lde)
                base_line_execution["LDE_t_score"] = t_score
                base_line_execution["LDE_c_score"] = c_score
                base_line_execution["LDE_my_c_score"] = my_c_score
                base_line_execution["LDE_knn_preserv"] = knn_preserv
                base_line_execution["LDE_strees"] = strees
                base_line_execution["LDE_js"] = js
                base_line_execution["LDE_hell"] = hell
                base_line_execution["LDE_js_sim"] = 1 - js
                base_line_execution["LDE_hell_sim"] = 1 - hell

                #############
                ############# MFA
                #############
                start = time.time()
                dados_mfa = MFA(dados_noisy, target, n_components=2, k1=nn, k2=nn)
                base_line_execution["MFA"] = time.time() - start
                print(f'MFA time: {base_line_execution["MFA"]}')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_mfa, n_neighbors=15)
                c_score = continuity(dados_clean, dados_mfa)
                my_c_score = myContinuity(dados_clean, dados_mfa)
                knn_preserv = knn_preservation(dados_clean, dados_mfa)
                strees = sammon_stress(dados_clean, dados_mfa)
                js, hell = distribution_scores(dados_clean, dados_mfa)
                base_line_execution["MFA_t_score"] = t_score
                base_line_execution["MFA_c_score"] = c_score
                base_line_execution["MFA_my_c_score"] = my_c_score
                base_line_execution["MFA_knn_preserv"] = knn_preserv
                base_line_execution["MFA_strees"] = strees
                base_line_execution["MFA_js"] = js
                base_line_execution["MFA_hell"] = hell
                base_line_execution["MFA_js_sim"] = 1 - js
                base_line_execution["MFA_hell_sim"] = 1 - hell

                #############
                ############# ISOMAP
                #############
                start = time.time()
                #iso = Isomap(n_neighbors=nn, n_components=2, metric="euclidean", path_method="auto", neighbors_algorithm="auto")
                iso = Isomap(n_neighbors=nn)
                dados_iso = iso.fit_transform(np.asarray(dados_noisy, dtype=float))
                end = time.time()
                base_line_execution["ISO"] = end - start
                print(f'ISO time: {base_line_execution["ISO"]:.4f}s')
                
                #------ scores
                t_score = trustworthiness(dados_clean, dados_iso, n_neighbors=15)
                c_score = continuity(dados_clean, dados_iso)
                my_c_score = myContinuity(dados_clean, dados_iso)
                knn_preserv = knn_preservation(dados_clean, dados_iso)
                js, hell = distribution_scores(dados_clean, dados_iso)
                strees = sammon_stress(dados_clean, dados_iso)
                base_line_execution["ISO_t_score"] = t_score
                base_line_execution["ISO_c_score"] = c_score
                base_line_execution["ISO_my_c_score"] = my_c_score
                base_line_execution["ISO_knn_preserv"] = knn_preserv
                base_line_execution["ISO_strees"] = strees
                base_line_execution["ISO_js"] = js
                base_line_execution["ISO_hell"] = hell
                base_line_execution["ISO_js_sim"] = 1 - js
                base_line_execution["ISO_hell_sim"] = 1 - hell
                
                #############
                ############# SK-ISOMAP
                #############
                start = time.time()
                dados_skiso = SK_Isomap(dados_noisy, nn, d_variation, target)
                end = time.time()
                base_line_execution["SUP K-ISO"] = end - start
                print(f'SUP K-ISO time: {base_line_execution["SUP K-ISO"]}')

                #------ scores
                # on plot with d variation
                

                #--------------
                #-------------- Supervised classification
                #--------------
                
                ############# PLS
                start = time.time()
                L_pls = Classification(dados_pls, target, 'PLS')
                end = time.time()
                base_line_execution["cls_PLS"] = end - start
                base_line_execution["cls_PLS_sc"] = L_pls[0]
                base_line_execution["cls_PLS_acc_avg"] = L_pls[1]
                base_line_execution["cls_PLS_acc_max"] = L_pls[2]
                base_line_execution["cls_PLS_f1_avg"] = L_pls[3]
                base_line_execution["cls_PLS_f1_max"] = L_pls[4]
                print(f'cls_PLS time: {base_line_execution["cls_PLS"]}')
                
                ############# UMAP
                start = time.time()
                L_umap = Classification(dados_umap, target, 'S-UMAP')
                end = time.time()
                base_line_execution["cls_S-UMAP"] = end - start
                base_line_execution["cls_S-UMAP_sc"] = L_umap[0]
                base_line_execution["cls_S-UMAP_acc_avg"] = L_umap[1]
                base_line_execution["cls_S-UMAP_acc_max"] = L_umap[2]
                base_line_execution["cls_S-UMAP_f1_avg"] = L_umap[3]
                base_line_execution["cls_S-UMAP_f1_max"] = L_umap[4]
                print(f'cls_S-UMAP time: {base_line_execution["cls_S-UMAP"]}')
                
                ############# SUP PCA
                start = time.time()
                L_suppca = Classification(dados_suppca.real, target, 'SUP PCA')
                end = time.time()
                base_line_execution["cls_SUP PCA"] = end - start
                base_line_execution["cls_SUP PCA_sc"] = L_suppca[0]
                base_line_execution["cls_SUP PCA_acc_avg"] = L_suppca[1]
                base_line_execution["cls_SUP PCA_acc_max"] = L_suppca[2]
                base_line_execution["cls_SUP PCA_f1_avg"] = L_suppca[3]
                base_line_execution["cls_SUP PCA_f1_max"] = L_suppca[4]
                print(f'cls_SUP PCA time: {base_line_execution["cls_SUP PCA"]}')
                
                ############# LDA
                start = time.time()
                L_lda = Classification(dados_lda, target, 'LDA')
                end = time.time()
                base_line_execution["cls_LDA"] = end - start
                base_line_execution["cls_LDA_sc"] = L_lda[0]
                base_line_execution["cls_LDA_acc_avg"] = L_lda[1]
                base_line_execution["cls_LDA_acc_max"] = L_lda[2]
                base_line_execution["cls_LDA_f1_avg"] = L_lda[3]
                base_line_execution["cls_LDA_f1_max"] = L_lda[4]
                print(f'cls_LDA time: {base_line_execution["cls_LDA"]}')
        
                ############# NCA
                start = time.time()
                L_nca = Classification(dados_nca, target, 'NCA')
                end = time.time()
                base_line_execution["cls_NCA"] = end - start
                base_line_execution["cls_NCA_sc"] = L_nca[0]
                base_line_execution["cls_NCA_acc_avg"] = L_nca[1]
                base_line_execution["cls_NCA_acc_max"] = L_nca[2]
                base_line_execution["cls_NCA_f1_avg"] = L_nca[3]
                base_line_execution["cls_NCA_f1_max"] = L_nca[4]
                print(f'cls_NCA time: {base_line_execution["cls_NCA"]}')
                
                ############# LMNN
                start = time.time()
                L_lmnn = Classification(dados_lmnn_2d, target, 'NCA')
                end = time.time()
                base_line_execution["cls_LMNN"] = end - start
                base_line_execution["cls_LMNN_sc"] = L_lmnn[0]
                base_line_execution["cls_LMNN_acc_avg"] = L_lmnn[1]
                base_line_execution["cls_LMNN_acc_max"] = L_lmnn[2]
                base_line_execution["cls_LMNN_f1_avg"] = L_lmnn[3]
                base_line_execution["cls_LMNN_f1_max"] = L_lmnn[4]
                print(f'cls_LMNN time: {base_line_execution["cls_LMNN"]}')

                ############# LFDA
                start = time.time()
                L_lfda = Classification(dados_lfda, target, 'LFDA')
                end = time.time()
                base_line_execution["cls_LFDA"] = end - start
                base_line_execution["cls_LFDA_sc"] = L_lfda[0]
                base_line_execution["cls_LFDA_acc_avg"] = L_lfda[1]
                base_line_execution["cls_LFDA_acc_max"] = L_lfda[2]
                base_line_execution["cls_LFDA_f1_avg"] = L_lfda[3]
                base_line_execution["cls_LFDA_f1_max"] = L_lfda[4]
                print(f'cls_LFDA time: {base_line_execution["cls_LFDA"]}')
                
                ############# Supervised LPP
                start = time.time()
                L_slpp = Classification(dados_slpp, target, 'SLPP')
                end = time.time()
                base_line_execution["cls_SLPP"] = end - start
                base_line_execution["cls_SLPP_sc"] = L_slpp[0]
                base_line_execution["cls_SLPP_acc_avg"] = L_slpp[1]
                base_line_execution["cls_SLPP_acc_max"] = L_slpp[2]
                base_line_execution["cls_SLPP_f1_avg"] = L_slpp[3]
                base_line_execution["cls_SLPP_f1_max"] = L_slpp[4]
                print(f'cls_SLPP time: {base_line_execution["cls_SLPP"]}')
                
                ############# LDE
                start = time.time()
                L_lde = Classification(dados_lde, target, 'LDE')
                end = time.time()
                base_line_execution["cls_LDE"] = end - start
                base_line_execution["cls_LDE_sc"] = L_lde[0]
                base_line_execution["cls_LDE_acc_avg"] = L_lde[1]
                base_line_execution["cls_LDE_acc_max"] = L_lde[2]
                base_line_execution["cls_LDE_f1_avg"] = L_lde[3]
                base_line_execution["cls_LDE_f1_max"] = L_lde[4]
                print(f'cls_LDE time: {base_line_execution["cls_LDE"]}')
                
                ############# MFA
                start = time.time()
                L_mfa = Classification(dados_mfa, target, 'MFA')
                end = time.time()
                base_line_execution["cls_MFA"] = end - start
                base_line_execution["cls_MFA_sc"] = L_mfa[0]
                base_line_execution["cls_MFA_acc_avg"] = L_mfa[1]
                base_line_execution["cls_MFA_acc_max"] = L_mfa[2]
                base_line_execution["cls_MFA_f1_avg"] = L_mfa[3]
                base_line_execution["cls_MFA_f1_max"] = L_mfa[4]
                print(f'cls_MFA time: {base_line_execution["cls_MFA"]}')
                
                ############# ISOMAP
                start = time.time()
                L_iso = Classification(dados_iso, target, 'ISO')
                end = time.time()
                base_line_execution["cls_ISO time"] = end - start
                base_line_execution["cls_ISO_sc"] = L_iso[0]
                base_line_execution["cls_ISO_acc_avg"] = L_iso[1]
                base_line_execution["cls_ISO_acc_max"] = L_iso[2]
                base_line_execution["cls_ISO_f1_avg"] = L_iso[3]
                base_line_execution["cls_ISO_f1_max"] = L_iso[4]
                print(f'cls_ISO time: {base_line_execution["cls_ISO time"]}')
                
                ############# Supervised K-ISOMAP with d variation
                for d in d_variation:
                    start = time.time()
                    L_kiso = Classification(dados_skiso[d], target, 'SUP K-ISO')
                    end = time.time()
                    base_line_execution[f"cls_SUP K-ISO_d{d}"] = end - start
                    base_line_execution[f"cls_SUP K-ISO_d{d}_sc"] = L_kiso[0]
                    base_line_execution[f"cls_SUP K-ISO_d{d}_acc_avg"] = L_kiso[1]
                    base_line_execution[f"cls_SUP K-ISO_d{d}_acc_max"] = L_kiso[2]
                    base_line_execution[f"cls_SUP K-ISO_d{d}_f1_avg"] = L_kiso[3]
                    base_line_execution[f"cls_SUP K-ISO_d{d}_f1_max"] = L_kiso[4]
                    print(f'cls_SUP K-ISO time: {base_line_execution[f"cls_SUP K-ISO_d{d}"]}')
                    
                    #------ scores
                    t_score = trustworthiness(dados_clean, dados_skiso[d], n_neighbors=15)
                    c_score = continuity(dados_clean, dados_skiso[d])
                    my_c_score = myContinuity(dados_clean, dados_skiso[d])
                    knn_preserv = knn_preservation(dados_clean, dados_skiso[d])
                    strees = sammon_stress(dados_clean, dados_skiso[d])
                    js, hell = distribution_scores(dados_clean, dados_skiso[d])
                    base_line_execution[f"SUP K-ISO_d{d}_t_score"] = t_score
                    base_line_execution[f"SUP K-ISO_d{d}_c_score"] = c_score
                    base_line_execution[f"SUP K-ISO_d{d}_my_c_score"] = my_c_score
                    base_line_execution[f"SUP K-ISO_d{d}_knn_preserv"] = knn_preserv
                    base_line_execution[f"SUP K-ISO_d{d}_strees"] = strees
                    base_line_execution[f"SUP K-ISO_d{d}_js"] = js
                    base_line_execution[f"SUP K-ISO_d{d}_hell"] = hell
                    base_line_execution[f"SUP K-ISO_d{d}_js_sim"] = 1 - js
                    base_line_execution[f"SUP K-ISO_d{d}_hell_sim"] = 1 - hell
                
                    PlotaDados(dados_skiso[d], target, f"SUP K-ISO_{dataset_name}_run{run}_k{nn}_d{d}", f"Supervised K-ISOMAP {dataset_name}")
                
                PlotaDados(dados_pls, target, f"PLS_{dataset_name}_run{run}_k{nn}", f"PLS {dataset_name}")
                PlotaDados(dados_umap, target, f"S-UMAP_{dataset_name}_run{run}_k{nn}", f"Supervised UMAP {dataset_name}")
                PlotaDados(dados_suppca, target, f"SUP PCA_{dataset_name}_run{run}_k{nn}", f"Supervised PCA {dataset_name}")
                PlotaDados(dados_lda, target, f"LDA_{dataset_name}_run{run}_k{nn}", f"LDA {dataset_name}")
                PlotaDados(dados_nca, target, f"NCA_{dataset_name}_run{run}_k{nn}", f"NCA {dataset_name}")
                PlotaDados(dados_lmnn_2d, target, f"LMNN_{dataset_name}_run{run}_k{nn}", f"LMNN {dataset_name}")
                PlotaDados(dados_lfda, target, f"LFDA_{dataset_name}_run{run}_k{nn}", f"LFDA {dataset_name}")
                PlotaDados(dados_slpp, target, f"SLPP_{dataset_name}_run{run}_k{nn}", f"SLPP {dataset_name}")
                PlotaDados(dados_lde, target, f"LDE_{dataset_name}_run{run}_k{nn}", f"LDE {dataset_name}")
                PlotaDados(dados_mfa, target, f"MFA_{dataset_name}_run{run}_k{nn}", f"MFA {dataset_name}")
                PlotaDados(dados_iso, target, f"ISO_{dataset_name}_run{run}_k{nn}_iso", f"ISOMAP {dataset_name}")
                
                all_results.append(base_line_execution)

                # --------------------------------------------------
                # Paper comparison figure (6 methods, 3x2)
                # --------------------------------------------------
                comparison_embeddings = {
                    "Supervised K-ISOMAP": dados_skiso[2],
                    "Supervised UMAP": dados_umap,
                    "NCA": dados_nca,
                    "LFDA": dados_lfda,
                    "LMNN": dados_lmnn_2d,
                    "ISOMAP": dados_iso
                }

                plot_comparison_image_fixed(
                    embeddings_dict=comparison_embeddings,
                    labels=target,
                    dataset_name=dataset_name,
                    save_dir=img_dir,
                    filename=f"{dataset_name}_comparison_run{run}_k{nn}.png",
                    dpi=600
                )


                # Convert dict to DataFrame
                df_current = pd.DataFrame([base_line_execution])
                
                if os.path.exists(csv_file):
                    # Append without headers
                    df_current.to_csv(csv_file, mode='a', header=False, index=False, sep=';', decimal=',')
                else:
                    # Create file with headers
                    df_current.to_csv(csv_file, index=False, sep=';', decimal=',')

            except Exception as e:
                error_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                base_line_execution["erro"] = f"Exception{e} - Run:{run} - Dataset:{dataset_name} - Timestamp{error_timestamp}"
                
                df_current = pd.DataFrame([base_line_execution])
                if os.path.exists(csv_file):
                    df_current.to_csv(csv_file, mode='a', header=False, index=False, sep=';', decimal=',')
                else:
                    df_current.to_csv(csv_file, index=False, sep=';', decimal=',')
                
                print(f"Error processing {dataset_name}: {e}")
