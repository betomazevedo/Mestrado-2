""" definitions for experiment 3 module
    Multiclass classification
    Statistical+Wavelet features
    Most-Recent and tranient Label strategy
    Drop windows with NaN label
    Used in experiment 2 of the master's dissertation.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from processing.feature_mappers import (
    TorchWaveletFeatureMapper,
    TorchStatisticalFeatureMapper,
    MixedMapper,
)
from processing.label_mappers import  TorchMulticlassTransientMRL

from experiments.base_experiment import BaseExperiment   
from dataset.dataset import MAEDataset


MAX_LEVEL = 10


def sample(trial, *args, **kwargs):
    return Experiment(
        level=trial.suggest_int("level", 4, MAX_LEVEL, step=1),
        stride=trial.suggest_int("stride", 10, 10),  # stride ís the Number of samples between consecutive windows
        n_components=trial.suggest_float("n_components", 0.9, 1.0),
        normal_balance=trial.suggest_int("normal_balance", 1, 10, step=1),
    )


class Experiment(BaseExperiment):
    """the docstring"""

    def __init__(
        self,
        level,
        stride,
        n_components,
        normal_balance,
        *args,
        **kwargs,
    ):
        super().__init__()

        # save params
        self.window_size = 2**level
        self.level = level
        self.n_components = n_components
        self.stride = stride
        self.normal_balance = normal_balance

        self._init_raw_mappers()
        self._init_preprocessor()

    def raw_transform(self, event, transient_only=True, no_nans=True):
        # filter tags and set zeros to nans
        tags = event["tags"][self.selected_tags].replace(0, np.nan)
        labels = event["labels"]
        event_type = event["event_type"]

        # to trim stablished fault if has transient
        if transient_only and MAEDataset.TRANSIENT_CLASS[event_type]:   # Seleciona somente os eventos que possuem fase transiente (1,2,5,6,7,8)
            transients = labels.values != event_type
            tags = tags[transients]
            labels = labels[transients]

        features = self._feature_mapper(tags, event_type)
        labels = self._label_mapper(labels, event_type)
        # print(labels)
        
        # drop windows with NaN label
        if no_nans:
            notnan = labels.notna()
            features = features[notnan]
            labels = labels[notnan]

        return features, labels, event_type

    def metric_name(self):
        return "balanced_accuracy"  # alterado em 11/12/23 para balanced_accuracy

    def metric_rf(self):
        return get_scorer("balanced_accuracy")   # alterado em 11/12/23 para balanced_accuracy

    def metric_lgbm(self):
        def acc(preds, train_data):
            preds_ = np.argmax(np.reshape(preds, (self.num_classes, -1)), axis=0)
            return "balanced_accuracy", balanced_accuracy_score(train_data.get_label(), preds_), True  # alterado em 11/12/23 para balanced_accuracy

        return acc

    def fit(self, X, y):
        y = self._label_encoder.fit_transform(y)   
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._pca.fit(X, y)  # Comentar, Para treinar sem PCA

    def transform(self, X, y):
        y = self._label_encoder.transform(y)
        X = self._scaler.transform(X)
        X = self._imputer.transform(X)
        X = self._pca.transform(X) # COMENTAR, Para treinar sem PCA
        # print("explained_variance_ratio:", self._pca.explained_variance_ratio_(X)) # Percentage of variance explained by each of the selected components.
        return X, y

    def _init_raw_mappers(self):
        offset = 2**MAX_LEVEL - self.window_size
        wavelet_mapper = TorchWaveletFeatureMapper(
            level=self.level, stride=self.stride, offset=offset
        )
        stats_mapper = TorchStatisticalFeatureMapper(
            window_size=2**self.level, stride=self.stride, offset=offset
        )

        self._feature_mapper = MixedMapper(stats_mapper, wavelet_mapper)

        self._label_mapper =   TorchMulticlassTransientMRL(
            window_size=self.window_size,
            stride=self.stride,
            offset=offset,
        )

    def _init_preprocessor(self):
        # z-score
        self._scaler = StandardScaler()
        # remove nans
        self._imputer = SimpleImputer(strategy="mean")
        # pca
        self._pca = PCA(n_components=self.n_components, whiten=True)

        # label encoder for multiclass
        self._label_encoder = LabelEncoder()   # Encode target labels with value between 0 and n_classes-1.
