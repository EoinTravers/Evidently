import numpy as np
from ..utils import clone_column, generate_randoms
from ..base import BaseModel

def run_race(pars, n=50, dt=.001, nt=5000, c=.1):
    '''Run multiple vectorised Race models with two (for now) accumulators.
    In this generic version, we allow t0, v, and z to differ between accumulators.
    In practice, you'll probably want to fix some of these parameters.

    Args:
        pars: Model parameters

          - t1: Onset of evidence accumulation (seconds) for x1
          - v1: Drift rate for x1
          - z1: Starting point for x1
          - t2: Onset of evidence accumulation (seconds) for x2
          - v2: Drift rate for x2
          - z2: Starting point for x2
          - a: Threshold.

        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt
        c: Noise standard deviation. Default = 0.1

    Returns:
        (list, array, array): tuple containing:

            - [X1, X2]: (List of 2 n x nt np.ndarrays): State of accumulators over time [list of (np.array:  n x nt)]
            - responses (np.ndarray): +/-1 if upper/lower threshold crossed, 0 otherwise.
            - rts (np.ndarray): Time of threshold crossing, in seconds, or np.NaN
    '''
    t1, v1, z1, t2, v2, z2, a = pars
    times = np.arange(nt)
    _NOISE = np.random.normal(loc=0., scale=c, size=(nt, n * 2))
    # NOTE: It would be interesting to allow noise to vary between accumulators!
    NOISES = _NOISE[:, :n], _NOISE[:, n:]
    Xs, crossing_times = [], []
    for t0, v, z, NOISE in zip([t1, t2], [v1, v2], [z1, z2], NOISES):
        colV = np.repeat(v, nt)     # Input as column vector
        colV[times < (t0 / dt)] = 0.  # Input is 0 before t0
        V = clone_column(colV, n)   # Input as (nt x n) matrix
        X = z + np.cumsum(V * dt + NOISE * np.sqrt(dt), 0)  # Accumulation
        crossing_time = np.argmax(X > a, 0)
        Xs.append(X.T)
        crossing_times.append(crossing_time)
    # Figure out crossing times, responses, and rts.
    crossing_times = np.array(crossing_times)
    which_first = np.argmin(crossing_times, 0)
    responses = np.where(which_first == 1, 1, -1)
    rts = np.min(crossing_times, 0) * dt
    responses[rts == 0] = 0
    rts[rts == 0] = np.nan
    return Xs, responses, rts


class Race(BaseModel):
    '''Classic Race Model.

    | The details below are for the .run_trial() method.
    | The rest of the methods are inherited from accumulate.BaseModel.

    -----

    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t1', 'v1', 'z1', 't2', 'v2', 'z2', 'a'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Race, self).__init__(trial_func=run_race,
                                   par_names=par_names,
                                   pars=pars,
                                   par_descriptions={
                                       't1': 'Non-decision time for accumulator 1',
                                       'v1': 'Drift rate for accumulator 1',
                                       'z1': 'Starting point for accumulator 1',
                                       't2': 'Non-decision time for accumulator 2',
                                       'v2': 'Drift rate for accumulator 2',
                                       'z2': 'Starting point for accumulator 2',
                                       'a' : 'Threshold'
                                   },
                                   max_time = max_time, dt=dt,
                                   bounds=None,
                                   n_traces=2)
        self.name = 'Drift Diffusion Model'
Race.__doc__ += run_race.__doc__
