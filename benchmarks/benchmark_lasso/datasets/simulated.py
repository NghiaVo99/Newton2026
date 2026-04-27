import numpy as np

from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples": [100, 500, 10_000],
        "n_features": [10_000, 600, 100],
        "rho": [0, 0.6],
    }

    def __init__(self, n_samples=10, n_features=50, rho=0, random_state=27):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.rho = rho

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            random_state=rng,
        )
        return dict(X=X, y=y)
