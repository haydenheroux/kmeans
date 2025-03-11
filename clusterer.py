import abc
import sklearn
import numpy as np


class Clusterer(abc.ABC):
    """
    Define a clusterer interface so that different libraries use a common API.
    """

    @abc.abstractmethod
    def __init__(self, clusters):
        pass

    @abc.abstractmethod
    def fit(self, pixels) -> np.typing.NDArray:
        pass

    @abc.abstractmethod
    def predict(self, pixels) -> np.typing.NDArray:
        pass


class MiniBatchKMeans(Clusterer):
    """
    Defines the clusterer for scikit-learn's MiniBatchKMeans clusterer.
    """

    def __init__(self, clusters) -> None:
        self.mini_batch_kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=clusters)
        self.fitted = False

    def fit(self, pixels):
        self.fitted = True
        self.mini_batch_kmeans.fit(pixels)
        return self.mini_batch_kmeans.cluster_centers_

    def predict(self, pixels):
        if not self.fitted:
            self.fit(pixels)

        return self.mini_batch_kmeans.predict(pixels)


class KMeans(Clusterer):
    """
    Defines the clusterer for scikit-learn's KMeans clusterer.
    """

    def __init__(self, clusters):
        self.kmeans = sklearn.cluster.KMeans(n_clusters=clusters)
        self.fitted = False

    def fit(self, pixels):
        self.fitted = True
        self.kmeans.fit(pixels)
        return self.kmeans.cluster_centers_

    def predict(self, pixels):
        if not self.fitted:
            self.fit(pixels)

        return self.kmeans.predict(pixels)
