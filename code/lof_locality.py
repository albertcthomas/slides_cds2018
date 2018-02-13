import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
lof = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0]

# Add outliers
# Generate numbers 3 times to get same random state as in sklearn example
rng = np.random.RandomState(42)
for i in range(3):
    _ = rng.uniform(low=-6, high=6, size=(n_outliers, 2))

X = np.concatenate([X, rng.uniform(low=-6, high=6,
                   size=(n_outliers, 2))], axis=0)

lof.fit(X)
y_pred = lof.fit_predict(X)

colors = np.array(['#377eb8', '#ff7f00'])

plt.figure()
plt.title('Local Outlier Factor', size=18)
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.savefig('locality_lof.pdf', bbox_inches='tight')
plt.show()
