import numpy as np
from ..utils import clone_column, generate_randoms
from ..base import BaseModel

def run_race(pars, n=50, dt=.001, nt=5000):
    '''Run multiple vectorised Race models with two (for now) accumulators.
    In this generic version, we allow t0, v, and z to differ between accumulators.
    In practice, you'll probably want to fix some of these parameters.

    Args:
        pars: Model parameters

          - t1: Onset of evidence accumulation (seconds) for x1
          - v1: Drift rate for x1
          - z1: Starting point for x1
          - c1: Noise for x1
          - t2: Onset of evidence accumulation (seconds) for x2
          - v2: Drift rate for x2
          - z2: Starting point for x2
          - c2: Noise for x2
          - a: Threshold.

        n: Number of trials to simulate
        dt: Delta time
        nt: Number of time steps. Trial duration = nt/dt

    Returns:
        (list, array, array): tuple containing:

            - [X1, X2]: (List of 2 n x nt np.ndarrays): State of accumulators over time [list of (np.array:  n x nt)]
            - responses (np.ndarray): +/-1 if upper/lower threshold crossed, 0 otherwise.
            - rts (np.ndarray): Time of threshold crossing, in seconds, or np.NaN
    '''
    t1, v1, z1, c1, t2, v2, z2, c2, a = pars
    times = np.arange(nt)
    _NOISE = np.random.normal(loc=0., scale=1., size=(nt, n * 2))
    # NOTE: It would be interesting to allow noise to vary between accumulators!
    NOISES = _NOISE[:, :n] * c1, _NOISE[:, n:] * c2
    Xs, crossing_times = [], []
    for t0, v, z, NOISE in zip([t1, t2], [v1, v2], [z1, z2], NOISES):
        colV = np.repeat(v, nt)     # Input as column vector
        colV[times < (t0 / dt)] = 0.  # Input is 0 before t0
        V = clone_column(colV, n)   # Input as (nt x n) matrix
        X = (z * a) + np.cumsum(V * dt + NOISE * np.sqrt(dt), 0)  # Accumulation
        crossing_time = np.argmax(X > a, 0)
        Xs.append(X.T)
        crossing_times.append(crossing_time)
    # Figure out crossing times, responses, and rts.
    max_rt = nt + 1
    crossing_times = np.array(crossing_times)
    crossing_times[crossing_times == 0] = max_rt
    which_first = np.argmin(crossing_times, 0)
    responses = np.where(which_first == 1, 1, -1)
    rts = np.min(crossing_times, 0).astype(float)
    timeouts = rts == max_rt
    rts *= dt
    responses[timeouts] = 0
    rts[timeouts] = np.nan
    return Xs, responses, rts


class Race(BaseModel):
    '''The classic Race model.

    This model has nine parameters (!), where x1 is accumulator #1, x2
    is accumulator #2:

    - t1: Onset of evidence accumulation (seconds) for x1
    - v1: Drift rate for x1
    - z1: Starting point for x1
    - c1: Noise for x1
    - t2: Onset of evidence accumulation (seconds) for x2
    - v2: Drift rate for x2
    - z2: Starting point for x2
    - c2: Noise for x2
    - a: Threshold.

    The model equation for each accumulator is

        | x_{t+1} = x_{t} + dt * input_{t} + sqrt(dt) * ε;
        | x_{0} = z * a
        | ε ~ Normal(0, c);
        | input_{t} = where(time > t, v, 0)

    The first accumulator to cross threshold, a, produces a response.
    Both accumulators use the same threshold, but can differ in their
    starting points. It is suggested that you keep `a=1` and vary the
    other parameters.

    Notes
    =====

    This model is extremely overparameterised, and cannot be fit to
    real data without fixing the value of several parameters!
    '''
    def __init__(self,
                 pars = [],
                 par_names = ['t1', 'v1', 'z1', 'c1', 't2', 'v2', 'z2', 'c2', 'a'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(Race, self).__init__(trial_func=run_race,
                                   par_names=par_names,
                                   pars=pars,
                                   par_descriptions={
                                       't1': 'Non-decision time for accumulator 1',
                                       'v1': 'Drift rate for accumulator 1',
                                       'z1': 'Starting point for accumulator 1',
                                       'c1': 'Noise for accumulator x1',
                                       't2': 'Non-decision time for accumulator 2',
                                       'v2': 'Drift rate for accumulator 2',
                                       'z2': 'Starting point for accumulator 2',
                                       'c2': 'Noise for accumulator x2',
                                       'a' : 'Threshold'
                                   },
                                   max_time = max_time, dt=dt,
                                   bounds=None,
                                   n_traces=2)
        self.name = 'Drift Diffusion Model'
