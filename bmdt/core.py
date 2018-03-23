from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class Bmdt(BaseEstimator, ClassifierMixin):
    """Binary Multiclass Decision Tree classification algorithm. It can be
    choice when there is a big majority class. There is fit and score
    methods like in Scikit."""

    def __init__(self, class_weight="balanced"):
        self.binary = DecisionTreeClassifier()

        self.multi = DecisionTreeClassifier()

        self.class_weight = class_weight
        self.binary.class_weight = self.class_weight
        self.majority_class = None
        self.classes = None

    def fit(self, X, y):
        """Training function. It takes a training vector features and a
        training class vector."""
        X = np.array(X)
        y = np.array(y)
        copy_y = y.copy()
        self.classes = np.unique(y)
        # we find the majority class
        self.majority_class = Counter(y).most_common()[0][0]
        # create a mask for the binary classification
        mask = copy_y == self.majority_class
        # apply the mask
        copy_y[mask] = self.majority_class
        copy_y[~mask] = 0
        # fit the binary classifier if the mask is enough
        if np.any(mask):
            self.binary.fit(X, copy_y)
            # get the predictions
            y_pred = self.binary.predict(X)
            # filter the non majority class
            mask = y_pred != self.majority_class
            if np.any(mask):
                # fit on it
                self.multi.fit(X[mask], y[mask])
            else:
                self.multi.fit(X, y)
        else:
            self.multi.fit(X, y)

    def predict(self, X):
        """Predict function. It predict the class, based on given features
        vector."""
        X = np.array(X)
        y_pred = self.binary.predict(X)
        mask = y_pred != self.majority_class
        # to avoid the case of empty array
        if np.any(mask):
            y_pred[mask] = self.multi.predict(X[mask])
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Score function. It computes the accuracy based on given features
        vector and class vector"""
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / y.shape[0]
