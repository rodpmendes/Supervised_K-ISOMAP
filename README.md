# Supervised K-ISOMAP

Code associated with the manuscript **“Supervised K-ISOMAP: Enhancing Metric
Learning via Curvature-Aware Geometry”**, by Rodrigo de P. Mendes and Alexandre
L. M. Levada.

Supervised K-ISOMAP is a deterministic, graph-based dimensionality-reduction
method that combines class information with tangent-space variations to
construct curvature-aware embeddings.

The native method is non-parametric and transductive. The original experiments
therefore measure class separability within embeddings constructed from the 36
benchmark datasets.

## Leakage-free evaluation

The additional out-of-sample implementation is organized as:

```text
experiments/leakage_free/
├── sup_kiso_leakage_free.py
└── supervised_k_isomap.py
```

It uses ten stratified 50/50 training--test splits. All feature preprocessing,
embedding construction, external mapping and classifier fitting are performed
using the training partition only. Test labels are used only for evaluation.

One dataset is excluded because of a singleton class, leaving 35 valid
datasets. Under the same external RBF protocol, Supervised K-ISOMAP obtains
57.08% mean balanced accuracy, compared with 47.82% for standard ISOMAP.

The external mapper is used only for this evaluation and is not a native
component of Supervised K-ISOMAP.

## Requirements

The experiment requires Python 3, NumPy, pandas, SciPy, scikit-learn,
UMAP-learn and metric-learn. It also requires the project implementation of
`SupervisedKIsomap` to be available to the script.

From the repository root, run:

```bash
python experiments/leakage_free/sup_kiso_leakage_free.py
```

## Citation

If you use this code, please cite the manuscript **“Supervised K-ISOMAP:
Enhancing Metric Learning via Curvature-Aware Geometry.”**
