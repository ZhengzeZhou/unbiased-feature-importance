from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from . import _ufi

__all__ = ["UnbiasedRandomForestClassifier", "UnbiasedRandomForestRegressor"]


class UnbiasedRandomForestClassifier(RandomForestClassifier):
    """Random Forest Classifier with unbiased feature importances.

    Computes feature importances using out-of-bag samples to remove the
    in-bag bias present in the standard MDI (mean decrease in impurity)
    measure. See arXiv:1903.05179 for details.

    All parameters are identical to sklearn's RandomForestClassifier.
    Note: oob_score is not required to be True for UFI computation.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self._ufi_importances = _ufi.cls(self, X, y)
        return self

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        return self._ufi_importances


class UnbiasedRandomForestRegressor(RandomForestRegressor):
    """Random Forest Regressor with unbiased feature importances.

    Computes feature importances using out-of-bag samples to remove the
    in-bag bias present in the standard MDI (mean decrease in impurity)
    measure. See arXiv:1903.05179 for details.

    All parameters are identical to sklearn's RandomForestRegressor.
    Note: oob_score is not required to be True for UFI computation.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self._ufi_importances = _ufi.regr(self, X, y)
        return self

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        return self._ufi_importances
