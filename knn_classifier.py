import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        preds = [self._predict_one(x) for x in X]
        return np.array(preds)

    def _predict_one(self, x):
        # Compute Euclidean distance
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_indices]
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        return most_common

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Use first 100 points for simplicity
X_train, y_train = X[:100], y[:100]
X_test, y_test = X[100:110], y[100:110]

knn = KNN(k=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("Predictions:", preds)
print("Actual:", y_test)
