# Repository for "Supervised K-ISOMAP: Enhancing Metric Learning via Curvature-Aware Geometry"
Dimensionality reduction is crucial for transforming high-dimensional data into meaningful representations. We propose Supervised K-ISOMAP, a novel method that integrates curvature-aware geometric information, induced by tangent-space variations along graph paths, and class labels to enhance class separability, offering a deterministic and conceptually simple alternative to existing methods such as Supervised UMAP.

A distinctive feature of Supervised K-ISOMAP is that it blends manifold learning with graph-based metric learning. Specifically, it uses tangent-space variations and class labels to guide a supervised reweighting of neighborhood edges, contracting intra-class connections and expanding inter-class ones. This strategy captures the global structure of the data while improving class separability through the learned graph representation. 
Unlike traditional methods, Supervised K-ISOMAP leverages tangent-space variations along graph paths to define a discrete notion of curvature for supervised edge reweighting.

What sets Supervised K-ISOMAP apart is its conceptual simplicity, determinism, and discriminative power. Given input data and corresponding labels, it requires only two hyperparameters, the number of neighbors and the output dimensionality, and avoids the randomness and heavy parameter tuning typical of methods like Supervised UMAP. 

Extensive experiments on 36 benchmark datasets show that Supervised K-ISOMAP delivers competitive or superior classification performance relative to representative baseline methods, while maintaining interpretability, determinism, and robustness.

Our findings demonstrate that incorporating curvature-aware geometry into supervised graph-based representations can significantly improve supervised dimensionality reduction, making Supervised K-ISOMAP a powerful and interpretable alternative in the landscape of metric learning methods.
