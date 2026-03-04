import unittest

from parameterized import parameterized
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from experiments.constants import HAMMING, EXPONENTIAL, BINARY, TERNARY
from qnn import QNN
from qnn.constants import GOLOMB, SEQUENTIAL, PARALLEL, TURNPIKE, BINARY_CROSS_ENTROPY
from test.utils import timeit


class TestTrainingClassificationUnivariateQNN(unittest.TestCase):
    """
    Test the training of a QNN for a simple classification task with a single feature.
    """

    @parameterized.expand([
        (HAMMING, 1, 6),
        (HAMMING, 2, 3),
        (HAMMING, 3, 2),
        (HAMMING, 6, 1),
        (EXPONENTIAL, 1, 6),
        (EXPONENTIAL, 2, 3),
        (EXPONENTIAL, 3, 2),
        (EXPONENTIAL, 6, 1),
        (BINARY, 1, 6),
        (BINARY, 2, 3),
        (BINARY, 3, 2),
        (BINARY, 6, 1),
        (TERNARY, 1, 6),
        (TERNARY, 2, 3),
        (TERNARY, 3, 2),
        (TERNARY, 6, 1),
    ])
    @timeit
    def test_simple_classification(self, encoding: str, R: int, L: int):
        x, y = make_classification(n_samples=100,
                                   n_features=1,
                                   n_classes=2,
                                   random_state=13,
                                   n_informative=1,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_clusters_per_class=1)
        for ansatz in (SEQUENTIAL, PARALLEL):
            print(f"Test {encoding=} with {(R, L)=}, {ansatz=}")
            qnn = QNN(R=R,
                      L=L,
                      N=1,
                      ansatz=ansatz,
                      encoding=encoding,
                      loss_fn=BINARY_CROSS_ENTROPY,
                      trainable_block_layers=5,
                      save_weights=False,
                      seed=13,
                      max_iter=100 if R > 2 else 500,
                      step_size=0.01,
                      verbose=False)
            qnn.fit(x.reshape(-1, 1), y)
            y_pred = qnn.predict(x.reshape(-1, 1)) > 0.5
            accuracy = accuracy_score(y_true=y, y_pred=y_pred)
            print(f"Accuracy: {accuracy}")
            self.assertGreater(accuracy, 0.95)

    @parameterized.expand([
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
    ])
    @timeit
    def test_simple_classification_golomb_and_turnpike(self, R: int, L: int):
        x, y = make_classification(n_samples=100,
                                   n_features=1,
                                   n_classes=2,
                                   random_state=13,
                                   n_informative=1,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_clusters_per_class=1)
        for ansatz in (SEQUENTIAL, PARALLEL):
            for encoding in (GOLOMB, TURNPIKE):
                print(f"Test {encoding=}, {(R, L)=}, {ansatz=}")
                qnn = QNN(R=R,
                          L=L,
                          N=1,
                          ansatz=ansatz,
                          encoding=encoding,
                          loss_fn=BINARY_CROSS_ENTROPY,
                          s=[0, 1, 4, 6],
                          trainable_block_layers=5,
                          save_weights=False,
                          seed=13,
                          max_iter=300,
                          step_size=0.01,
                          verbose=False)
                qnn.fit(x.reshape(-1, 1), y)
                y_pred = qnn.predict(x.reshape(-1, 1)) > 0.5
                accuracy = accuracy_score(y_true=y, y_pred=y_pred)
                print(f"Accuracy: {accuracy}")
                self.assertGreater(accuracy, 0.95)


class TestTrainingClassificationMultivariateQNN(unittest.TestCase):
    """
    Test the training of a QNN for a simple classification task with multiple features.
    """

    @parameterized.expand([
        (HAMMING, 1, 6, 2),
        (HAMMING, 2, 3, 2),
        (HAMMING, 3, 2, 2),
        (HAMMING, 6, 1, 2),
        (HAMMING, 1, 6, 3),
        (HAMMING, 2, 3, 3),
        (HAMMING, 3, 2, 3),
        (HAMMING, 6, 1, 3),
        (EXPONENTIAL, 1, 6, 2),
        (EXPONENTIAL, 2, 3, 2),
        (EXPONENTIAL, 3, 2, 2),
        (EXPONENTIAL, 6, 1, 2),
        (EXPONENTIAL, 1, 6, 3),
        (EXPONENTIAL, 2, 3, 3),
        (EXPONENTIAL, 3, 2, 3),
        (EXPONENTIAL, 6, 1, 3),
        (BINARY, 1, 6, 2),
        (BINARY, 2, 3, 2),
        (BINARY, 3, 2, 2),
        (BINARY, 6, 1, 2),
        (BINARY, 1, 6, 3),
        (BINARY, 2, 3, 3),
        (BINARY, 3, 2, 3),
        (BINARY, 6, 1, 3),
        (TERNARY, 1, 6, 2),
        (TERNARY, 2, 3, 2),
        (TERNARY, 3, 2, 2),
        (TERNARY, 6, 1, 2),
        (TERNARY, 1, 6, 3),
        (TERNARY, 2, 3, 3),
        (TERNARY, 3, 2, 3),
        (TERNARY, 6, 1, 3),
    ])
    @timeit
    def test_simple_classification(self, encoding: str, R: int, L: int, N: int):
        x, y = make_classification(n_samples=100,
                                   n_features=N,
                                   n_classes=2,
                                   random_state=13,
                                   n_informative=N,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_clusters_per_class=1)
        for ansatz in (SEQUENTIAL, PARALLEL):
            print(f"Test {encoding=} with {(R, L, N)=}, {ansatz=}")
            if R * N > 10 and ansatz == PARALLEL:
                print("Skipped since more than 10 qubits required")
                continue
            qnn = QNN(R=R,
                      L=L,
                      N=N,
                      ansatz=ansatz,
                      encoding=encoding,
                      loss_fn=BINARY_CROSS_ENTROPY,
                      trainable_block_layers=5,
                      save_weights=False,
                      seed=13,
                      max_iter=200,
                      step_size=0.01,
                      verbose=False)
            qnn.fit(x, y)
            y_pred = qnn.predict(x) > 0.5
            accuracy = accuracy_score(y_true=y, y_pred=y_pred)
            print(f"Accuracy: {accuracy}")
            self.assertGreater(accuracy, 0.85)


    @parameterized.expand([
        (4, 1, 2),
        (4, 1, 3),
        (4, 2, 2),
        (4, 2, 3),
        (8, 1, 2),
        (8, 1, 3),
        (8, 2, 2),
        (8, 2, 3),
    ])
    @timeit
    def test_simple_classification_golomb_and_turnpike(self, R: int, L: int, N: int):
        x, y = make_classification(n_samples=100,
                                   n_features=N,
                                   n_classes=2,
                                   random_state=13,
                                   n_informative=N,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_clusters_per_class=1)
        for ansatz in (SEQUENTIAL, PARALLEL):
            for encoding in (GOLOMB, TURNPIKE):
                print(f"Test {encoding=}, {(R, L, N)=}, {ansatz=}")
                if R * N > 10 and ansatz == PARALLEL:
                    print("Skipped since more than 10 qubits required")
                    continue
                qnn = QNN(R=R,
                          L=L,
                          N=N,
                          ansatz=ansatz,
                          encoding=encoding,
                          loss_fn=BINARY_CROSS_ENTROPY,
                          s=[0, 1, 4, 6],
                          trainable_block_layers=5,
                          save_weights=False,
                          seed=13,
                          max_iter=100 if R > 1 else 300,
                          step_size=0.01,
                          verbose=False)
                qnn.fit(x, y)
                y_pred = qnn.predict(x) > 0.5
                accuracy = accuracy_score(y_true=y, y_pred=y_pred)
                print(f"Accuracy: {accuracy}")
                self.assertGreater(accuracy, 0.90)


if __name__ == '__main__':
    unittest.main()
