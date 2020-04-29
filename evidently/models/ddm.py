import numpy as np
import functools

from ..utils import clone_column, generate_randoms
from ..base import BaseModel

def run_ddm(pars, lower_bound=True, n=50, dt=.001, nt=5000):
    '''Run Drift Diffusion trials, with/without a lower response bound.

    Args:
        pars: Model parameters

          - t0: Onset of evidence accumulation (seconds)
          - v: Drift rate after t0
          - z: Starting point
          - a: Threshold.
          - c: Noise Standard deviation

        lower_bound: Whether a response should occur if lower bound crossed.

            | If True (Default), this is a Diffusion model. If False, Wald.
            | See also accumulate.models.Wald().

        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt

    Returns:
        (array, array, array): tuple containing:

            - X (np.ndarray): State of accumulator over time (Shape n x nt)
            - responses (np.ndarray): +/-1 if upper/lower threshold crossed, 0 otherwise.
            - rts (np.ndarray): Time of threshold crossing, in seconds, or np.NaN
    '''
    t0, v, z, a, c = pars
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=c, size=(nt, n))
    colV = np.repeat(v, nt)       # Input as column vector
    colV[times < (t0 / dt)] = 0.  # Input is 0 before t0
    V = clone_column(colV, n)     # Input as (nt x n) matrix
    X = (z * a) + np.cumsum(V * dt + NOISE * np.sqrt(dt), 0)  # Accumulation
    if lower_bound:
        rt = np.argmax(np.abs(X) > a, 0)  # Either response boundary
        # Response is sign of accumulator when boundary reached
        response = np.array([np.sign(X[_t, i]) if _t > 0 else 0
                             for i, _t in enumerate(rt)])
    else:
        rt = np.argmax(X > a, 0)  # Upper boundary only
        response = np.where(rt > 0, 1, 0)
    rt = np.where(rt > 0, rt, np.nan) * dt  # Convert to seconds
    return X.T, response, rt

class Diffusion(BaseModel):
    '''The classic Drift Diffusion model.

    This model has five free parameters.

    - t0: Onset of evidence accumulation (seconds)
    - v: Drift rate after t0.
    - z: Starting point.
    - a: Threshold.
    - c: Noise standard devitation.

    The model equation is

        | x_{t+1} = x_{t} + dt * input_{t} + sqrt(dt) * ε;
        | x_{0} = z * a
        | ε ~ Normal(0, c);
        | input_{t} = where(t > t0, v, 0)
        | Response:
        |  +1 if x > a;
        |  -1 if x < -a;
        |  0 otherwise

    Notes
    =====

    Like most models in evidently, this model is overparameterised, so
    that if you were to multiply the parameters v, z, a, and c by the
    same value the model simulations do not change. This means if
    you're attempting to fit this model to real data you'll need to
    fix one of these values. By convention, c is usually fixed to
    `c=1`, although some older papers use `c=0.1`.

    The starting point parameter, `z`, is measured in units of the
    threshold `a`. This means that a starting point of `z = 0.5` means
    the accumulator starts halfway to the upper boundary, where ever
    that might be. This is the case for all models with a starting point parameter.

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0', 'v', 'z', 'a', 'c'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Diffusion, self).__init__(trial_func=run_ddm,
                                        par_names=par_names,
                                        pars=pars,
                                        par_descriptions={
                                            't0': 'Non-decision time',
                                            'v': 'Drift rate',
                                            'z': 'Starting point',
                                            'a': 'Threshold (±)',
                                            'c': 'Noise SD'
                                        },
                                        max_time = max_time, dt=dt,
                                        bounds=None,
                                        n_traces=1)
        self.name = 'Classic Drift Diffusion Model'


run_wald = functools.partial(run_ddm, lower_bound=False)
run_wald.__doc__ = run_wald.func.__doc__.replace('with/', '')


class Wald(BaseModel):
    '''The Wald Diffusion model.

    This model is identical to the Drift Diffusion model
    (`evidently.models.Diffusion`), except that there is no lower
    response boundary, and so only one response is possible.

    This model has five free parameters.

    - t0: Onset of evidence accumulation (seconds)
    - v: Drift rate after t0.
    - z: Starting point.
    - a: Threshold.
    - c: Noise standard devitation.

    The model equation is

    | x_{t+1} = x_{t} + dt * input_{t} + sqrt(dt) * ε;
    | x_{0} = z * a
    | ε ~ Normal(0, c);
    | input_{t} = v if t > t0, 0 otherwise;
    | Response:  +1 if x > a

    Notes
    =====

    See `evidently.models.Diffusion`.
    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0', 'v', 'z', 'a', 'c'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Wald, self).__init__(trial_func=run_wald,
                                   par_names=par_names,
                                   pars=pars,
                                   par_descriptions={
                                       't0': 'Non-decision time',
                                       'v': 'Drift rate',
                                       'z': 'Starting point',
                                       'a': 'Threshold (positive)',
                                       'c': 'Noise SD'
                                   },
                                   max_time = max_time, dt=dt,
                                   bounds=None,
                                   n_traces=1)
        self.name = 'Wald Drift Diffusion Model'


# Hierarchical version
def run_hddm(pars, lower_bound=True, n=50, dt=.001, nt=5000, c=.1):
    '''Run Hierarchical Drift Diffusion trials, with(out) a lower response bound.

    Args:
        pars: Model parameters

          - t0_mu: Mean of Onset of evidence accumulation (seconds)
          - v_mu: Mean of Drift rate after t0
          - z_mu: Mean of Starting point
          - a_mu: Mean of Threshold.
          - t0_sd: Standard deviation (SD) of t0 (Gamma distribution)
          - v_sd: SD of v (Gaussian)
          - z_sd: SD of z (Gaussian)
          - a_sd: SD of threshold a (Gaussian).

        lower_bound: Emit response if lower bound crossed?
        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt
        c: Noise standard deviation. Default = 0.1

    Returns:
        (array, array, array): tuple containing:

            - X (np.ndarray): State of accumulator over time (Shape n x nt)
            - responses (np.ndarray): +/-1 if upper/lower threshold crossed, 0 otherwise.
            - rts (np.ndarray): Time of threshold crossing, in seconds, or np.NaN
    '''

    m_t0, m_v, m_z, m_a, s_t0, s_v, s_z, s_a, c = pars
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=c, size=(nt, n))
    # Variance between trials
    T0, V, Z, A = generate_randoms(means=[m_t0, m_v, m_z, m_a],
                                   sds  =[s_t0, s_v, s_z, s_a],
                                   dist=['gamma', 'normal', 'normal', 'gamma'],
                                   n=n)
    Z = Z * A # Scale starting point by threshold
    V = clone_column(V, nt).T  # Input as (nt x n) matrix
    if s_t0 > 0:  # Non-decision times
        for _t0 in T0:
            V[times < (_t0 / dt)] = 0.  # No evidence before t0 on this trial
    else:
        V[times < (m_t0 / dt)] = 0.
    # Do accumulation
    X = Z + np.cumsum(V * dt + NOISE * np.sqrt(dt), 0)
    if lower_bound:
        rt = np.argmax(np.abs(X) > A, 0)  # Either response boundary
        # Response is sign of accumulator when boundary reached
        response = np.array([np.sign(X[_t, i]) if _t > 0 else 0
                             for i, _t in enumerate(rt)])
    else:
        rt = np.argmax(X > A, 0)  # Upper boundary only
        response = np.where(rt > 0, 1, 0)
    rt = np.where(rt > 0, rt, np.nan) * dt  # Convert to seconds
    return X.T, response, rt



def run_hwald(pars, n=50, dt=.001, nt=5000):
    '''
    See `evidently.models.run_hierarchical_ddm`.
    '''
    return run_hddm(pars=pars, lower_bound=False, n=n, dt=dt, nt=nt, c=c)


class HDiffusion(BaseModel):
    '''The Hierarchical Drift Diffusion model.

    This model is based on `evidently.models.Diffusion`,
    but instead of setting every parameter to a fixed value,
    it takes a mean and standard deviation for each parameter,
    and randomly varies parameters across trials.

    Parameters:

    - m_t0 : Mean of Onset of evidence accumulation (seconds)
    - m_v  : Mean of Drift rate after t0.
    - m_z  : Mean of Starting point.
    - m_a  : Mean of Threshold.
    - s_t0 : SD of t0 (Gamma distribution)
    - s_v  : SD of v  (Normal distribution)
    - s_z  : SD of z  (Normal distribution)
    - s_a  : SD of a  (Gamma distribution)
    - c    : Noise standard devitation.

    As in the standard drift diffusion, the model equation is

    | x_{t+1} = x_{t} + dt * input_{t} + sqrt(dt) * ε;
    | x_{0} = z * a
    | ε ~ Normal(0, c);
    | input_{t} = v if t > t0, 0 otherwise;
    | Response:
    |  +1 if x > a
    |  -1 ix x < -a
    |  0 otherwise

    The difference is that each parameter is now drawn randomly from a
    distribution:

    | t0 ~ Gamma*(m_t0, s_t0);
    | v ~ Normal(m_v, s_v);
    | z ~ Normal(m_z, s_z);
    | a ~ Gamma*(m_a, s_a);

    Gamma*(m, s) denotes random samples from a Gamma distribution with
    mean m and standard deviation s. See
    `evidently.utils.random_gamma` for details. We use the Gamma
    distribution for parameters that must be positive.

    Notes
    -----

    Although we do not allow the noise parameter c to vary across
    trials here, it might be interesting to do so.

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['m_t0', 'm_v', 'm_z', 'm_a',
                              's_t0', 's_v', 's_z', 's_a',
                              'c'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(HDiffusion, self).__init__(trial_func=run_hddm,
                                         par_names=par_names,
                                         pars=pars,
                                         par_descriptions={
                                             'm_t0' : 'Mean of Onset of evidence accumulation (seconds)',
                                             'm_v'  : 'Mean of Drift rate after t0.',
                                             'm_z'  : 'Mean of Starting point.',
                                             'm_a'  : 'Mean of Threshold.',
                                             's_t0' : 'SD of t0 (Gamma distribution)',
                                             's_v'  : 'SD of v  (Normal distribution)',
                                             's_z'  : 'SD of z  (Normal distribution)',
                                             's_a'  : 'SD of a  (Gamma distribution)',
                                             'c'    : 'Noise standard devitation.',
                                         },
                                         max_time = max_time, dt=dt,
                                         bounds=None,
                                         n_traces=1)
        self.name = 'Hierarchical Drift Diffusion Model'


class HWald(BaseModel):
    '''The Hierarchical Wald Diffusion model.

    All details are the same as `evidently.models.HDiffusion`, except
    there is no lower response boundary.

    See also `evidently.models.Diffusion` and `evidently.models.Wald`.
    '''
    def __init__(self,
                 pars = [],
                 par_names = ['m_t0', 'm_v', 'm_z', 'm_a',
                              's_t0', 's_v', 's_z', 's_a',
                              'c'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(HDiffusion, self).__init__(trial_func=run_hwald,
                                         par_names=par_names,
                                         pars=pars,
                                         par_descriptions={
                                             'm_t0' : 'Mean of Onset of evidence accumulation (seconds)',
                                             'm_v'  : 'Mean of Drift rate after t0.',
                                             'm_z'  : 'Mean of Starting point.',
                                             'm_a'  : 'Mean of Threshold.',
                                             's_t0' : 'SD of t0 (Gamma distribution)',
                                             's_v'  : 'SD of v  (Normal distribution)',
                                             's_z'  : 'SD of z  (Normal distribution)',
                                             's_a'  : 'SD of a  (Gamma distribution)',
                                             'c'    : 'Noise standard devitation.',
                                         },
                                         max_time = max_time, dt=dt,
                                         bounds=None,
                                         n_traces=1)
        self.name = 'Hierarchical Wald Diffusion Model'
