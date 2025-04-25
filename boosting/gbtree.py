import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None  # Feature used for split
        self.threshold = None      # Threshold for the split
        self.left_value = None     # Predicted value for left side
        self.right_value = None    # Predicted value for right side

    def fit(self, X, y):
        m, n = X.shape
        min_error = float('inf')

        # Try all features and thresholds to find best split
        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                left_output = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_output = np.mean(y[right_mask]) if np.any(right_mask) else 0

                prediction = np.where(left_mask, left_output, right_output)
                error = np.mean((y - prediction) ** 2)

                if error < min_error:
                    min_error = error
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_output
                    self.right_value = right_output

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.init_value = 0

    def fit(self, X, y):
        self.init_value = np.mean(y)
        F = np.full(y.shape, self.init_value)

        for _ in range(self.n_estimators):
            residual = y - F
            stump = DecisionStump()
            stump.fit(X, residual)
            update = stump.predict(X)
            F += self.learning_rate * update
            self.trees.append(stump)

    def predict(self, X):
        F = np.full((X.shape[0],), self.init_value)
        for stump in self.trees:
            F += self.learning_rate * stump.predict(X)
        return np.where(F >= 0.5, 1, 0)
