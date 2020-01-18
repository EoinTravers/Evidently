import numpy as np
import functools

from ..utils import clone_column, generate_randoms
from ..base import BaseModel

def run_ddm(pars, lower_bound=True, n=50, dt=.001, nt=5000, c=.1):
    '''Run Drift Diffusion trials, with/without a lower response bound.

    Args:
        pars: Model parameters

          - t0: Onset of evidence accumulation (seconds)
          - v: Drift rate after t0
          - z: Starting point
          - a: Threshold.

        lower_bound: Emit response if lower bound crossed?

            | If True (Default), this is a Diffusion model. If False, Wald.
            | See also accumulate.models.Wald().

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
    t0, v, z, a = pars
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=c, size=(nt, n))
    colV = np.repeat(v, nt)     # Input as column vector
    colV[times < (t0 / dt)] = 0.  # Input is 0 before t0
    V = clone_column(colV, n)   # Input as (nt x n) matrix
    X = z + np.cumsum(V * dt + NOISE * np.sqrt(dt), 0)  # Accumulation
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

    | The details below are for the .run_trial() method.
    | The rest of the methods are inherited from accumulate.BaseModel.

    ---

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0', 'v', 'z', 'a'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Diffusion, self).__init__(trial_func=run_ddm,
                                        par_names=par_names,
                                        pars=pars,
                                        par_descriptions={
                                            't0': 'Non-decision time',
                                            'v': 'Drift rate',
                                            'z': 'Starting point',
                                            'a': 'Threshold (±)'
                                        },
                                        max_time = max_time, dt=dt,
                                        bounds=None,
                                        n_traces=1)
        self.name = 'Classic Drift Diffusion Model'
Diffusion.__doc__ += run_ddm.__doc__


run_wald = functools.partial(run_ddm, lower_bound=False)
run_wald.__doc__ = run_wald.func.__doc__.replace('with/', '')


class Wald(BaseModel):
    '''The Wald Diffusion model - or a DDM with no lower boundary.

    | The details below are for the .run_trial() method.
    | The rest of the methods are inherited from accumulate.BaseModel.

    -----

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0', 'v', 'z', 'a'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Wald, self).__init__(trial_func=run_wald,
                                   par_names=par_names,
                                   pars=pars,
                                   par_descriptions={
                                       't0': 'Non-decision time',
                                       'v': 'Drift rate',
                                       'z': 'Starting point',
                                       'a': 'Threshold (positive)'
                                   },
                                   max_time = max_time, dt=dt,
                                   bounds=None,
                                   n_traces=1)
        self.name = 'Wald Drift Diffusion Model'
Wald.__doc__ = run_wald.__doc__


# Heirarchical version
def run_hddm(pars, lower_bound=True, n=50, dt=.001, nt=5000, c=.1):
    '''Run Heirarchical Drift Diffusion trials, with(out) a lower response bound.

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
    t0_mu, v_mu, z_mu, a_mu, t0_sd, v_sd, z_sd, a_sd = pars
    times = np.arange(nt)
    NOISE = np.random.normal(loc=0., scale=c, size=(nt, n))
    # Variance between trials
    T0, V, Z, A = generate_randoms(means=[t0_mu, v_mu, z_mu, a_mu],
                                   sds  =[t0_sd, v_sd, z_sd, a_sd],
                                   dist=['gamma', 'normal', 'normal', 'normal'],
                                   n=n)
    V = clone_column(V, nt).T  # Input as (nt x n) matrix
    if t0_sd > 0:  # Non-decision times
        for _t0 in T0:
            V[times < (_t0 / dt)] = 0.  # No evidence before t0 on this trial
    else:
        V[times < (t0_mu / dt)] = 0.
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


def run_hierarchical_wald(pars, n=50, dt=.001, nt=5000, c=.1):
    '''Run Heirarchical Wald Diffusion trials, without a lower response bound.

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
    return run_hddm(pars=pars, lower_bound=False, n=n, dt=dt, nt=nt, c=c)


class HDiffusion(BaseModel):
    '''The Heirarchical Drift Diffusion model

    | The details below are for the .run_trial() method.
    | The rest of the methods are inherited from accumulate.BaseModel.

    -----

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0_mu', 'v_mu', 'z_mu', 'a_mu',
                              't0_sd', 'v_sd', 'z_sd', 'a_sd'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(HDiffusion, self).__init__(trial_func=run_hddm,
                                         par_names=par_names,
                                         pars=pars,
                                         par_descriptions={
                                            't0_mu': 'Mean of Non-decision time',
                                             'v_mu': 'Mean of Drift rate',
                                             'z_mu': 'Mean of Starting point',
                                             'a_mu': 'Mean of Threshold (±)',
                                             't0_sd': 'SD of Non-decision time (Gamma distribution)',
                                             'v_sd': 'SD of Drift rate (Normal distribution)',
                                             'z_sd': 'SD of Starting point (Normal)',
                                             'a_sd': 'SD of Threshold (±) (Normal)'
                                         },
                                         max_time = max_time, dt=dt,
                                         bounds=None,
                                         n_traces=1)
        self.name = 'Heirarchical Wiener Diffusion Model'
HDiffusion.__doc__ += run_hddm.__doc__


class HWald(BaseModel):
    '''The Heirarchical Wald Diffusion model

    | The details below are for the .run_trial() method.
    | The rest of the methods are inherited from accumulate.BaseModel.

    -----

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t0_mu', 'v_mu', 'z_mu', 'a_mu',
                              't0_sd', 'v_sd', 'z_sd', 'a_sd'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(HWald, self).__init__(trial_func=run_hierarchical_wald,
                                    par_names=par_names,
                                    pars=pars,
                                    par_descriptions={
                                        't0_mu': 'Mean of Non-decision time',
                                        'v_mu': 'Mean of Drift rate',
                                        'z_mu': 'Mean of Starting point',
                                        'a_mu': 'Mean of Threshold (±)',
                                        't0_sd': 'SD of Non-decision time (Gamma distribution)',
                                        'v_sd': 'SD of Drift rate (Normal distribution)',
                                        'z_sd': 'SD of Starting point (Normal)',
                                        'a_sd': 'SD of Threshold (±) (Normal)'
                                    },
                                    max_time = max_time, dt=dt,
                                    bounds=None,
                                    n_traces=1)
        self.name = 'Heirarchical Wald Diffusion Model'
HWald.__doc__ += run_hddm.__doc__
