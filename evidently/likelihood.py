import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

from sklearn.neighbors.kde import KernelDensity

# Stolen from
# https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html#bw_silverman
# to save a dependency.

def _select_sigma(X):
    normalize = 1.349
    IQR = (sap(X, 75) - sap(X, 25))/normalize
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)

def bw_silverman(x, kernel=None):
    A = _select_sigma(x)
    n = len(x)
    return .9 * A * n ** (-0.2)


def eval_with_kde(x, y):
    """Silverman Epanechnikov kernel"""
    try:
        bw = bw_silverman(x)
        kde = stats.gaussian_kde(x, bw_method='silverman')
        return np.log(kde.evaluate(y))
    except (ValueError, ZeroDivisionError):
        return -np.inf
#     kde_skl = KernelDensity(bandwidth=bw, kernel='epanechnikov')
#     kde_skl.fit(x[:, np.newaxis])
#     ll = kde.score_samples(y[:, np.newaxis])
#     return ll

def kde_loglik_1d(simulated_rts, observed_rts,
                  bw_method='silverman'):
    nas = np.isnan(simulated_rts)
    prop_na = np.mean(nas)
    simulated_rts = simulated_rts[~nas]
    ll = eval_with_kde(simulated_rts, observed_rts)
    return ll

def kde_loglik_multiresponse(simulated_responses, simulated_rts,
                             observed_responses, observed_rts):
    def ll_for_response(r):
        sim_rt = simulated_rts[simulated_responses == r]
        obs_rt = observed_rts[observed_responses == r]
        p_resp = np.mean(observed_responses == r)
        lls = eval_with_kde(sim_rt, obs_rt) + np.log(p_resp)
        return np.sum(lls)
    total_ll = np.sum([ll_for_response(r) for r in set(observed_responses)])
    return total_ll

def kde_sim_plot(simulated_responses, simulated_rts,
                 observed_responses, observed_rts,
                 max_time=5.):
    times = np.linspace(-max_time, max_time, 101)
    bins = np.linspace(-max_time, max_time, 51)
    signed_rts = observed_responses * observed_rts
    plt.hist(signed_rts, bins, density=True, alpha=.5)
    for r in [-1, 1]:
        _rts = simulated_rts[simulated_responses==r] * r
        p_resp = np.mean(simulated_responses==r)
        if len(_rts) > 2:
            kde = stats.gaussian_kde(_rts, bw_method='silverman')
            x = times[np.sign(times) == r]
            y = kde.pdf(x) * p_resp
            plt.plot(x, y, linewidth=2)
    plt.ylim(0, 1.5)
