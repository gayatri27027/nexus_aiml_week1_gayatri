import numpy as np
import pandas as pd

class LinearRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.m = None
        self.c = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        self.m = 0
        self.c = 0

        for _ in range(self.n_iters):
            y_pred = self.m * X + self.c
            # Gradients
            dm = (-2/n) * np.sum(X * (y - y_pred))
            dc = (-2/n) * np.sum(y - y_pred)
            # Update parameters
            self.m -= self.lr * dm
            self.c -= self.lr * dc

    def predict(self, X):
        return self.m * np.array(X) + self.c

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegressionScratch(lr=0.01, n_iters=1000)
model.fit(X, y)
preds = model.predict(X)

print(f"Slope: {model.m:.3f}, Intercept: {model.c:.3f}")
print("Predictions:", preds)