# Unbiased Feature Importance

This repository implements unbiased measurement of feature importance in Random Forests, as described in:

> Mentch, L. & Zhou, Z. (2020). Randomization as Regularization: A Degrees of Freedom Explanation for Random Forest Success. *Journal of Machine Learning Research*, 21(171), 1-36. ([arXiv:1903.05179](https://arxiv.org/abs/1903.05179))

## Installation

```sh
pip install .
```

Or in development mode:

```sh
pip install -e .
```

## Usage

### Basic usage

```python
from unbiased_feature_importance import UnbiasedRandomForestClassifier, UnbiasedRandomForestRegressor
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

clf = UnbiasedRandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

print(clf.feature_importances_)
```

### SelectFromModel

`feature_importances_` is available after fitting, so `SelectFromModel` works out of the box:

```python
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(UnbiasedRandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X, y)
X_selected = selector.transform(X)
print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
```

### Pipeline

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('select', SelectFromModel(UnbiasedRandomForestClassifier(n_estimators=50, random_state=42))),
    ('clf', UnbiasedRandomForestClassifier(n_estimators=100, random_state=42)),
])
pipe.fit(X, y)
print(pipe.score(X, y))
```

### Regression

```python
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

reg = UnbiasedRandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y)
print(reg.feature_importances_)
```

## Background

Standard Random Forest feature importance (MDI) is computed using in-bag samples, which introduces a systematic upward bias for features with many splits. This implementation uses out-of-bag (OOB) samples to evaluate node impurity, producing an unbiased estimate of each feature's true contribution.
