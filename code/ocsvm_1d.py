"""Showing that the One-Class SVM scoring function is constant around the
modes of the distribution"""

import numpy as np

from scipy.stats import norm

from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
import matplotlib

font = {'weight': 'normal',
        'size': 15}

matplotlib.rc('font', **font)


# generating gaussian mixture samples
n_samples = 1000

weight_1 = 0.5
weight_2 = 0.5
mean_1 = 1
mean_2 = 9
cov_1 = 1
cov_2 = 1
weights = np.array([weight_1, weight_2])
means = np.array([mean_1, mean_2])
covars = np.array([cov_1, cov_2])


def density(X):
    """Gaussian Mixture density."""

    n_samples = len(X)
    density = np.zeros(n_samples)

    for (weight, mean, cov) in zip(weights, means, covars):
        density += weight * norm.pdf(X, loc=mean, scale=cov)

    return density


rng = np.random.RandomState(42)

n_samples_comp = rng.multinomial(n_samples, weights)

x = np.hstack([
    norm.rvs(loc=mean, scale=cov, size=int(sample), random_state=rng).ravel()
    for (mean, cov, sample) in zip(
        means, covars, n_samples_comp)])

# grid for plot
grid = np.linspace(x.min() - 1.5, x.max() + 1.5, 1000)

clf = OneClassSVM(nu=0.2, gamma=1.0)
clf.fit(x.reshape(-1, 1))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title('Gaussian mixture density')
ax1.plot(grid, density(grid), lw=2, color='blue')
ax2.set_title('OneClassSVM scoring function')
ax2.plot(grid, clf.score_samples(grid.reshape(-1, 1)), lw=2, color='green')
plt.tight_layout()
plt.show()
