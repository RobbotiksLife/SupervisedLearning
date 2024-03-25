import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm  # Import tqdm for loading indicators

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # y_pred = [self._predict(x) for x in X]
        # return np.array(y_pred)
        y_pred = []
        # Use tqdm to show loading during prediction
        for x in tqdm(X, desc="Predicting", unit=" sample"):
            y_pred.append(self._predict(x))
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distances.append(np.sqrt(np.sum((x - x_train)**2)))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return int(np.argmax(np.bincount(k_nearest_labels)))

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'o')
    colors = ('blue', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=cl)

