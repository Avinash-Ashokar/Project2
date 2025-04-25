import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from boosting.gbtree import GradientBoostingClassifier
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Gradient Boosting Classifier Testing")
parser.add_argument('--n_estimators', type=int, default=20, help='Number of estimators for Gradient Boosting')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for Gradient Boosting')
parser.add_argument('--save_plots', action='store_true', help='Save plots instead of displaying them')
args = parser.parse_args()

# Ensure the results directory exists
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

# Helper function to plot decision boundary
def plot_decision_boundary(X, y, model, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if args.save_plots:
        plot_path = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(plot_path)
    plt.close()

# Define datasets for testing
datasets = {
    "Moons": make_moons(noise=0.2, random_state=42),
    "Circles": make_circles(noise=0.2, factor=0.5, random_state=42),
    "Linearly Separable": make_classification(n_features=2, n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1, random_state=42),
}

# Evaluate classifier and plot results
for name, (X, y) in datasets.items():
    logging.info(f"Training Gradient Boosting Classifier on {name} dataset")
    clf = GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    clf.fit(X, y)
    logging.info(f"Model training completed for {name} dataset")

    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    logging.info(f"Accuracy on {name} dataset: {acc:.2f}")

    if args.save_plots:
        logging.info(f"Saving plot for {name} dataset")
        plot_decision_boundary(X, y, clf, f"Gradient Boosting on {name} (Accuracy:_{acc:.2f})")
    else:
        logging.info(f"Displaying plot for {name} dataset")
        plot_decision_boundary(X, y, clf, f"Gradient Boosting on {name} (Accuracy:_{acc:.2f})")
        plt.show()
