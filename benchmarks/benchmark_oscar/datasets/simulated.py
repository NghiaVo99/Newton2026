from benchopt import BaseDataset
from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        "n_samples": [20_000, 500, 200],
        "n_features": [1_000, 20_000, 2_000_000],
        "n_signals": [40, 20],
        "X_density": [1.0, 0.001],
        "rho": [0, 0.8],
    }

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def __init__(
        self,
        n_samples=10,
        n_features=50,
        n_signals=5,
        X_density=1.0,
        rho=0,
        standardize=True,
        random_state=27,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_signals = n_signals
        self.X_density = X_density
        self.rho = rho
        self.standardize = standardize
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            density=self.n_signals / self.n_features,
            random_state=self.random_state,
            X_density=self.X_density,
        )
        return dict(X=X, y=y)
