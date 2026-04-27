from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import rand as sprand
    from scipy.sparse.linalg import LinearOperator


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        "n_samples": [400],
        "n_features": [250],
        "n_blocks": [10],
        "loc, scale": [(0, 0.1)],
        "type_A": ["identity", "random", "conv"],
        "type_x": ["block", "sin"],
        "type_n": ["gaussian", "laplace"],
        "random_state": [27],
    }

    test_parameters = {
        "type_A": ["random", "conv"],
        "n_samples, n_features": [(10, 5)],
    }

    def __init__(
        self,
        n_samples=5,
        n_features=5,
        n_blocks=1,
        loc=0,
        scale=0.01,
        type_A="identity",
        type_x="block",
        type_n="gaussian",
        random_state=27,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.loc = loc
        self.scale = scale
        self.type_A = type_A
        self.type_x = type_x
        self.type_n = type_n
        self.random_state = random_state

    def get_A(self, rng):
        if self.type_A == "identity":
            width = min(self.n_samples, self.n_features)

            def matvec(x):
                out = np.zeros(self.n_samples)
                out[:width] = x[:width]
                return out

            def matmat(X):
                out = np.zeros((self.n_samples, X.shape[1]))
                out[:width, :] = X[:width, :]
                return out

            def rmatvec(x):
                out = np.zeros(self.n_features)
                out[:width] = x[:width]
                return out

            def rmatmat(X):
                out = np.zeros((self.n_features, X.shape[1]))
                out[:width, :] = X[:width, :]
                return out

            return LinearOperator(
                dtype=np.float64,
                matvec=matvec,
                matmat=matmat,
                rmatvec=rmatvec,
                rmatmat=rmatmat,
                shape=(self.n_samples, self.n_features),
            )
        if self.type_A == "random":
            return rng.randn(self.n_samples, self.n_features)
        if self.type_A == "conv":
            if self.n_samples < self.n_features:
                raise ValueError("type_A='conv' requires n_samples >= n_features.")
            len_A = self.n_samples - self.n_features + 1
            filt = rng.randn(len_A)
            return LinearOperator(
                dtype=np.float64,
                matvec=lambda x: np.convolve(x, filt, mode="full"),
                matmat=lambda X: np.array(
                    [np.convolve(x, filt, mode="full") for x in X.T]
                ).T,
                rmatvec=lambda x: np.correlate(x, filt, mode="valid"),
                rmatmat=lambda X: np.array(
                    [np.correlate(x, filt, mode="valid") for x in X.T]
                ).T,
                shape=(self.n_samples, self.n_features),
            )
        raise ValueError(f"Unknown type_A={self.type_A!r}.")

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        if self.type_x == "sin":
            t = np.arange(self.n_features)
            x = np.cos(np.pi * t / self.n_features * self.n_blocks)
        elif self.type_x == "block":
            z = sprand(
                1,
                self.n_features,
                density=self.n_blocks / self.n_features,
                random_state=rng,
            ).toarray()[0]
            x = np.cumsum(rng.randn(self.n_features) * z)
        else:
            raise ValueError(f"Unknown type_x={self.type_x!r}.")

        if self.type_n == "gaussian":
            noise = rng.normal(self.loc, self.scale, self.n_samples)
        elif self.type_n == "laplace":
            noise = rng.laplace(self.loc, self.scale, self.n_samples)
        else:
            raise ValueError(f"Unknown type_n={self.type_n!r}.")

        A = self.get_A(rng)
        y = A @ x + noise
        return dict(A=A, y=y, x=x)
