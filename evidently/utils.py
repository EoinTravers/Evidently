import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from numba import jit, njit
from typing import Union, Optional

def do_dataset_no_stimuli(model, n: int, *args, **kwargs):
    '''Simulate a dataset for a model that does not take trial-by-trial inputs.

    Args:
        model: pycumulate.AccumulatorModel object
        n: number of repititions to simulate
        Additional arguments get passed to `model.do_trial()`.

    Returns:
        X: DataFrame containing accumulator traces for each repitition.
           If model.n_traces > 1, X uses a MultiIndex.
        R: Responses for each repitition.
        T: Response times for each repitition.
    '''
    n_traces = model.n_traces
    X, responses, rts = model.do_trial()
    trials = [model.do_trial(*args, **kwargs) for i in range(n)]
    _, R, T = zip(*trials)
    X = [pd.DataFrame(trial[0]).T for trial in trials]
    X = pd.concat(X)
    if n_traces > 1:
        X['accumulator'] = X.index
        X['rep'] = np.repeat(range(n), n_traces)
        X = X.set_index(['rep', 'accumulator'])
    R = np.array(R)
    T = np.array(T)
    return X, R, T


def do_dataset_with_stimuli(model, n, pars=None, *args, **kwargs):
    '''Simulate a dataset for a model that takes trial-by-trial inputs.

    Args:
        model: pycumulate.AccumulatorModel object
               model must have its `stimuli` attribute set.
               Stimuli should be of shape (n_trials, ...), e.g.
               (n_trials,) for scaler inputs,
               (n_trials, j) for 1-D inputs,
               (n_trials, j, k) for 2-D inputs, etc.
        n: number of repititions to simulate.
        Additional arguments get passed to `model.do_trial()`.

    Returns:
        X: DataFrame containing accumulator traces for each repitition of each trial..
        R: Responses for each repitition (Shape: (n_trials, n))
        T: Response times for each repitition (Shape: (n_trials, n)).
    '''
    if pars is None:
        pars = model.pars
    n_traces = model.n_traces
    stimuli = model.stimuli
    n_trials = stimuli.shape[0]
    trials = [  # ( (Repitions) x Trials )
        [model.do_trial(pars, stimuli[s], *args, **kwargs) for i in range(n)]
        for s in range(n_trials)
    ]
    if n_traces == 1:
        X = [[rep[0] for rep in trial] for trial in trials]
        X = pd.DataFrame(np.concatenate(X))
        X['trial'] = np.repeat(range(n_trials), n)
        X['rep'] = np.tile(range(n), n_trials)
        X = X.set_index(['trial', 'rep'])
    else:
        X = [[rep[0].T for rep in trial] for trial in trials]
        X = np.array(X)  # Stim x Reps x Accumulators x Time
        X = X.reshape(-1, X.shape[-1])  # (Stim x Reps x Accum) x Time
        X = pd.DataFrame(X)
        X['trial'] = np.repeat(range(n_trials), n * n_traces)
        X['rep'] = np.tile(np.repeat(range(n), n_traces), n_trials)
        X['accumulator'] = np.tile(range(n_traces), n * n_trials)
        X = X.set_index(['trial', 'rep', 'accumulator'])
    RESPONSE = np.array([[rep[1] for rep in trial] for trial in trials])
    RT = np.array([[rep[2] for rep in trial] for trial in trials])
    return X, RESPONSE, RT


def lock_to_movement(X, rts, duration = 2, min_rt=None):
    '''Realign simulated data to the time it crosses a threshold.
    Args:
        X: Simulated data (pd.DataFrame)
        rts: Simulated response times
        duration: Number of seconds prior to threshold to include (Default 2).
        min_rt: Shortest RT to include. Defaults to `duration`.
    '''
    if min_rt is None:
        min_rt = duration
    dt = np.diff(X.columns)[0]
    float_rts = np.copy(rts) # Avoid 'RuntimeWarning: invalid value encountered in greater'
    float_rts[np.isnan(rts)] = -1
    mask = float_rts > min_rt
    _X = X.loc[mask]
    _X.columns = _X.columns.astype(float)
    _rts = rts[mask]
    nt = int(duration / dt) + 1
    # This looks worse than it is
    mX = [_X.loc[trial, (rt - duration - .5*dt):rt].values[(-nt):]
          for trial, rt in zip(_X.index, _rts)]
    mX = pd.DataFrame(mX, columns=np.linspace(-duration, 0, nt))
    return mX

def silverman_bandwidth(x):
    iqr = np.diff(np.percentile(x, [25, 75]))[0]
    n = len(x)
    return .9 * min(np.std(x), iqr / 1.34) * np.power(n, -.2)

def epanechnikov_kde(x, h=None):
    if h is None:
        h = silverman_bandwidth(x)
    X = x.reshape(-1, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=h).fit(X)
    return kde

def unit_bind(x):
    return max(min(x, 1), 0)

def _flip(x):
    i = np.arange(len(x), 0, -1) - 1
    return x[i]
flip = jit(_flip)

def random_gamma(mean: float, sd: float, n=None):
    '''Gamma distribution parameterised by mean and standard deviation.'''
    var = sd ** 2
    mean, var = [float(x) for x in [mean, var]]
    shape = mean**2 / var
    scale = var / mean
    x = np.random.gamma(shape, scale, n)
    return x

def random_normal(mean: float, sd: float, n=None):
    '''np.random.normal(loc=mean, scale=sd, size=n)'''
    return np.random.normal(loc=mean, scale=sd, size=n)


def random_samples(mean: float, sd: float, dist: str, n=None):
    '''Random samples with a given mean, sd, from a given distribution.
    If sd = 0, returns np.repeat(mean, n).
    Like np.random, output is a float if n=None (default), array otherwise.
    '''
    funs = {
        'normal': random_normal,
        'gamma': random_gamma
    }
    if dist in funs.keys():
        f = funs[dist]
        if sd > 0:
            return f(mean, sd, n)
        else:
            return np.repeat(mean, n)
    else:
        raise NotImplementedError('dist %s not available' % dist)


def generate_randoms(means: list, sds: list, n=None, dist='normal'):
    '''Lists of random numbers with specified means, SDs, and distributions.

    Args:
        means: List of k mean values
        sds: List of k SD values
        n: How many values to sample (None or int)
        dist: Distribution to use. String, or list of length k.
              Accepted values so far are 'normal' and 'gamma'.

    Returns:
        List of np.ndarray: A list of randomly sampled values.
    '''
    k = len(means)
    assert len(sds) == k
    if type(dist) == str:
        dist = [dist] * k
    results = [random_samples(m, s, d, n=n) if s > 0
               else np.ones(n) * m
               for m, s, d in zip(means, sds, dist)]
    return results


@njit()
def clone_column(x: np.array, n: int):
    '''[a,b,c] -> [[a,b,c], [a,b,c], ..., [a,b,c]]'''
    return np.array([list(x), ]  * n).T

@jit()
def fill_matrix(dim: int, diagonal: float, off_diagonal: float) -> np.ndarray:
    matrix_I = np.identity(dim)
    return (matrix_I * -(off_diagonal - diagonal)) + off_diagonal


def split_by_accumulator(trace: pd.DataFrame):
    if trace.index.names[1] != 'accumulator':
        raise Exception('Model does not have multiple accumulators')
    _, accums = zip(*trace.index.values)
    accums = np.unique(accums)
    Xs = [trace.xs(accum, level=1) for accum in accums]
    return Xs


def _leaky_accumulation(x0: np.ndarray,
                        k: float,
                        V: np.array,
                        dt=.001):
    '''Vectorised leaky accumulation.

    Args:
        x0: Starting values (Shape: n)
        k: Decay parameter
        V: Input (including noise) (Shape (nt, n))
    '''
    nt = V.shape[0]
    n = x0.shape[0]
    x = x0
    results = np.zeros((n, nt))
    for i in range(nt):
        results[:, i] = x
        x = x + V[i] - .001 * k * x
    return results.T

leaky_accumulation = jit(_leaky_accumulation, nopython=True)


def _leaky_competition(x0: np.ndarray,
                       k: float,
                       b: float,
                       V: np.array,
                       n_trials: int,
                       n_accums: int,
                       n_times: int,
                       saturation: Optional[float] = None,
                       rectify: bool               = True,
                       dt: float                   = .001):
    '''Simulate multiple trials of leaky competition between multiple accumulators.
    See equation 4 of Usher & McClelland (2011)
    http://web.stanford.edu/~jlmcc/papers/UsherMcC01.pdf

    Args:
        x0: Starting values (Shape: n_trials, n_accums)
        k: Self-feedback (negative for decay)
        b: Lateral feedback (negative for lateral inhibition)
        V: Input (including noise), (Shape (n_trials, n_accums, n_times))
           Should already be scaled appropriately, eq N(v * dt, c * np.sqrt(dt))
        saturation: Prevent values higher than a threshold? Should be None, or a threshold.
                    Should be slightly higher than response threshold!
        rectify: Prevent negative values?
        dt: Time step

    Returns:
        np.ndarray of accumualtor values (nt, n)
    '''
    assert(V.shape == (n_trials, n_accums, n_times))
    assert(x0.shape == (n_trials, n_accums))
    # Can we do this without a loop?
    transition_matrix = fill_matrix(n_accums, diagonal=k, off_diagonal=b)
    results = np.zeros_like(V)
    for trial_ix in range(n_trials):
        accum_res = np.zeros((n_accums, n_times))
        x = x0[trial_ix]
        for i in range(n_times):
            accum_res[:, i] = x
            dx = V[trial_ix, :, i] + x.dot(transition_matrix) * dt
            x = x + dx
            if rectify:
                x[x < 0] = 0.
            if saturation is not None:
                x[x > saturation] = saturation
        results[trial_ix, :, :] = accum_res
    return results

leaky_competition = jit(_leaky_competition, nopython=True)

def weibull_bounds(times, a_initial: float, a_terminal: float, time_par: float, shape_par: float):
    '''Weibull collapsing bounds function.
    Args:
        times: Time points to calculate the bounds at
        a_initial: Starting height of bounds
        a_terminal: Final height of bounds.
                    May not be reached by the end of the time window
        time_par: Dictates how late the bounds begin to collapse.
        shape_par: Dictates how steeply the bounds collapse

    See https://pyddm.readthedocs.io/en/latest/cookbook/bounds.html#bound-weibull-cdf,
    and https://www.jneurosci.org/content/35/6/2476

    '''
    return a_initial - (1 - np.exp(-(times / time_par)**shape_par)) * (a_initial - a_terminal)
