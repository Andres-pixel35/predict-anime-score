import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _count_values(series: pd.Series) -> pd.Series:
    return series.str.split(";").explode().value_counts()


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encode a ';'-separated column, keeping values seen >= min_count times in fit."""

    def __init__(self, column, prefix, min_count, add_other=False, add_size=False):
        self.column = column
        self.prefix = prefix
        self.min_count = min_count
        self.add_other = add_other
        self.add_size = add_size

    def fit(self, X, y=None):
        self.counts_ = _count_values(X[self.column])
        self.keep_ = self.counts_[self.counts_ >= self.min_count].index.tolist()
        return self

    def transform(self, X):
        col = X[self.column]
        lists = col.str.split(";").apply(lambda x: x if isinstance(x, list) else [])
        keep = set(self.keep_)

        encoded = (
            col.str.get_dummies(sep=";")
            .reindex(columns=self.keep_, fill_value=0)
            .add_prefix(f"{self.prefix}_")
        )
        if self.add_other:
            encoded[f"{self.prefix}_other"] = lists.apply(lambda l: int(any(s not in keep for s in l)))
        if self.add_size:
            # largest (train) count among the item's values; unseen values count 0
            encoded[f"{self.prefix}_size"] = lists.apply(
                lambda l: max((self.counts_.get(s, 0) for s in l), default=0)
            )
        return pd.concat([X, encoded], axis=1).drop(columns=[self.column])


class StaffEncoder(BaseEstimator, TransformerMixin):
    """Staff -> staff_wellknown / staff_other / staff_unknown (missing)."""

    CATEGORIES = ["wellknown", "other", "unknown"]

    def __init__(self, column, prefix, min_count=50):
        self.column = column
        self.prefix = prefix
        self.min_count = min_count

    def fit(self, X, y=None):
        counts = _count_values(X[self.column])
        self.keep_ = set(counts[counts >= self.min_count].index)
        return self

    def _label(self, value):
        if not isinstance(value, str):
            return "unknown"
        return "wellknown" if self.keep_ & set(value.split(";")) else "other"

    def transform(self, X):
        labels = pd.Categorical(X[self.column].map(self._label), categories=self.CATEGORIES)
        dummies = pd.get_dummies(labels, dtype=int, prefix=self.prefix, prefix_sep="_")
        dummies.index = X.index
        return pd.concat([X, dummies], axis=1).drop(columns=[self.column])


class MedianByTypeImputer(BaseEstimator, TransformerMixin):
    """Fill NaN in `column` with the train median of the row's `type_column` group.

    Types unseen in fit fall back to the overall train median.
    """

    def __init__(self, column="episode_duration_min", type_column="type"):
        self.column = column
        self.type_column = type_column

    def fit(self, X, y=None):
        self.global_median_ = X[self.column].median()
        self.medians_ = X.groupby(self.type_column)[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.type_column].map(self.medians_).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        return X
