import numpy as np
# import functools
import pandas as pd

from ..utils import clone_column, generate_randoms, leaky_accumulation
from ..base import BaseModel

def run_schurger(pars, n=50, dt=.001, nt=12000, a=1.):
    '''Single-accumulator model with leak,
    used as a model for self-initiated actions.

    Note that this implementation has t0 and z parameters that do not
    feature in the original version. These should be set to 0 to
    recover the original implementation.

    Args:
        pars: Model parameters
          - t0: Start time of input
          - v: Strength of input
          - k: Decay
          - z: Starting point
          - a: Threshold
          - c: Noise
        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt

    Returns:
        X: State of accumulator over time (np.array:  n x nt)
        response: 1 if threshold crossed, 0 otherwise.
        rt: Time of threshold crossing, in seconds, or np.NaN

    '''
    t0, v, k, z, a, c = pars
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=c, size=(nt, n))
    colV = np.repeat(v, nt)       # Input as column vector
    colV[times < (t0 / dt)] = 0.  # Input is 0 before t0
    V = clone_column(colV, n)     # Input as (nt x n) matrix
    Z = np.repeat(z * a, n)
    X = leaky_accumulation(Z, k, dt * V + np.sqrt(dt) * NOISE)
    rt = np.argmax(X > a, 0)  # Upper boundary only
    response = np.where(rt > 0, 1, 0)
    rt = np.where(rt > 0, rt, np.nan) * dt  # Convert to seconds
    return X.T, response, rt


class Schurger(BaseModel):
    '''Single-accumulator model with leak, used as a model for
    self-initiated actions.

    This model has 6 free parameters:

    - t0: Start time of input (seconds)
    - v: Strength of input
    - k: Decay
    - z: Starting point
    - a: Threshold
    - c: Noise

    Note that this implementation has t0 and z parameters that do no
    feature in the original version. These should be set to 0 to
    recover the original implementation.

    The model equation is

        | x_{t+1} = x_{t} + dt * (input_{t} - k * x{t-1}) * ε;
        | x_{0} = z * a
        | ε ~ Normal(0, c*sqrt(dt));
        | input_{t} = where(t > t0, v, 0)
        | Response:
        |  +1 if x > a;
        |  0 otherwise

    Notes
    =====
    The noise parameter `c=0.1` in the original paper.

    Schurger, A., Sitt, J. D., & Dehaene, S. (2012). An accumulator
    model for spontaneous neural activity prior to self-initiated
    movement. Proceedings of the National Academy of Sciences,
    109(42), E2904-E2913.
    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0', 'v', 'k', 'z', 'a', 'c'],
                 max_time = 12., dt=.001,
                 bounds=None):
        super(Schurger, self).__init__(trial_func=run_schurger,
                                       par_names=par_names,
                                       pars=pars,
                                       par_descriptions={
                                           't0': 'Start time',
                                           'v': 'Drift rate',
                                           'k': 'Decay',
                                           'z': 'Starting point',
                                           'a': 'Threshold',
                                           'c': 'Noise SD'
                                       },
                                       max_time = max_time, dt=dt,
                                       bounds=None,
                                       n_traces=1)
        self.name = 'Schurger Model'


def run_schurger_hierarchical(pars, n=50, dt=.001, nt=12000,
                              return_trial_pars=False):
    '''Schurger Single-accumulator model with
    trial-to-trial variance in all parameters.

    Args:
        pars: Model parameters
          - t0_m: Start time of input
          - v_m: Strength of input
          - k_m: Decay
          - z_m: Starting point
          - a_m: Threshold
          - c_m: Noise
          - t0_sd: Start time of input
          - v_sd: Strength of input
          - k_sd: Decay
          - z_sd: Starting point
          - a_sd: Threshold
          - c_sd: Noise
        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt
        return_trial_pars: Return DataFrame of parameters on each trial?
                           Default False.

    Returns:
        X: State of accumulator over time (np.array:  n x nt)
        response: 1 if threshold crossed, 0 otherwise.
        rt: Time of threshold crossing, in seconds, or np.NaN

    If return_trial_pars==True, returns X, response, rt, trial_pars,
    where trial_pars is a pandas DataFrame.
    '''
    (t0_m, v_m, k_m, z_m, a_m, c_m,
     t0_sd, v_sd, k_sd, z_sd, a_sd, c_sd) = pars
    means = t0_m, v_m, k_m, z_m, a_m, c_m
    sds = t0_sd, v_sd, k_sd, z_sd, a_sd, c_sd
    dists = ['gamma', 'normal', 'normal', 'normal', 'gamma', 'gamma']
    # Get vectors of parameter values for each trial
    t0, v, k, z, a, c = generate_randoms(means, sds, n=n, dist=dists)
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=1, size=(nt, n)) * c.T
    V = clone_column(v, nt).T  # Input as (nt, n) matrix
    for i, _t0 in enumerate(t0):
        V[times < (_t0 / dt), i] = 0.  # Input is 0 before t0
    Z = z * a  # Starting point is relative to threshold
    X = leaky_accumulation(Z, k, dt * V + np.sqrt(dt) * NOISE)
    CROSSED = X > a  # n×t > n
    rt = np.argmax(CROSSED, 0)
    response = np.where(rt > 0, 1, 0)
    rt = np.where(rt > 0, rt, np.nan) * dt  # Convert to seconds
    if return_trial_pars:
        trial_pars = pd.DataFrame([t0, v, k, z, a, c],
                                  index=['t0', 'v', 'k', 'z', 'a', 'c']).T
        return X.T, response, rt, trial_pars
    else:
        return X.T, response, rt


class HSchurger(BaseModel):
    '''Hierarchiacal Schurger model, where every parameter can vary
    between trials.

    See evidently.models.Schurger for details.

    This model has 6 parameters per trial:

    - t0: Start time of input (seconds)
    - v: Strength of input
    - k: Decay
    - z: Starting point
    - a: Threshold
    - c: Noise

    Each of these requirese a mean value, and a standard deviation.
    Values on individual trials are drawn from a normal distritribution
    for [v, k, z]. The remaining parameters are drawn from a  gamma
    distribution to ensure non-negative values.

    Notes
    =====

    Schurger, A., Sitt, J. D., & Dehaene, S. (2012). An accumulator
    model for spontaneous neural activity prior to self-initiated
    movement. Proceedings of the National Academy of Sciences,
    109(42), E2904-E2913.

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0_m', 'v_m', 'k_m', 'z_m', 'a_m', 'c_m',
                              't0_sd', 'v_sd', 'k_sd', 'z_sd', 'a_sd', 'c_sd'],
                 max_time = 12., dt=.001,
                 bounds=None):
        super(HSchurger, self).__init__(trial_func=run_schurger_hierarchical,
                                       par_names=par_names,
                                       pars=pars,
                                       par_descriptions={
                                           't0_m': 'Mean Start time',
                                           'v_m':  'Mean Drift rate',
                                           'k_m':  'Mean Decay',
                                           'z_m':  'Mean Starting point',
                                           'a_m':  'Mean Threshold',
                                           'c_m':  'Mean Noise variance',
                                           't0_sd': 'SD Start time',
                                           'v_sd':  'SD Drift rate',
                                           'k_sd':  'SD Decay',
                                           'z_sd':  'SD Starting point',
                                           'a_sd':  'SD Threshold',
                                           'c_sd':  'SD Noise variance'
                                       },
                                       max_time = max_time, dt=dt,
                                       bounds=None,
                                       n_traces=1)
        self.name = 'Hierarchical Schurger Model'
