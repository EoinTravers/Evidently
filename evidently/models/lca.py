import numpy as np
from ..utils import clone_column, generate_randoms, fill_matrix, leaky_competition
from ..base import BaseModel


def run_lca(pars, n=50, dt=.001, nt=5000):
    '''Run multiple vectorised LCA models with two (for now) accumulators.
    In this generic version, we allow t0, v, k and z to differ between accumulators.
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
          - k: Autofeedback (negative for decay)
          - b: Lateral feedback (negative for inhibition)
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
    # Note - models.run_race() is a special case of run_lca.
    # These should be refactored to reduce duplication.
    t1, v1, z1, c1, t2, v2, z2, c2, k, b, a = pars
    times = np.arange(nt)
    x0 = np.repeat([z1, z2], n).reshape(2, n).T
    INPUTS = np.ones((n, 2, nt))
    NOISE = np.random.normal(0., 1., (n, 2, nt))
    for i, t, v, c in zip([0, 1], [t1, t2], [v1, v2], [c1, c2]):
        INPUTS[:, i, times < t] = 0
        INPUTS[:, i] *= v
        NOISE[:, i] *= c
    INPUTS = dt * INPUTS + np.sqrt(dt) * NOISE
    X_matrix = leaky_competition(x0, k, b, INPUTS,
                                 n_trials=n, n_accums=2, n_times=nt,
                                 rectify=True, dt=dt)
    Xs, crossing_times = [], []
    for i in range(2):
        iX = X_matrix[:, i]
        crossing_time = np.argmax(iX > a, 1)
        Xs.append(iX)
        crossing_times.append(crossing_time)
    crossing_times = np.array(crossing_times)
    max_rt = nt + 1
    crossing_times[crossing_times == 0] = max_rt
    which_first = np.argmin(crossing_times, 0)
    responses = np.where(which_first == 1, 1, -1)
    rts = np.min(crossing_times, 0).astype(float)
    timeouts = rts == max_rt
    rts *= dt
    responses[timeouts] = 0
    rts[timeouts] = np.nan
    return Xs, responses, rts


class LCA(BaseModel):
    '''Leaky Competing Accumulators.

    This model has 11 parameters (!), where x1 is accumulator #1, x2
    is accumulator #2:

    - t1: Onset of evidence accumulation (seconds) for x1
    - v1: Drift rate for x1
    - z1: Starting point for x1
    - c1: Noise for x1
    - t2: Onset of evidence accumulation (seconds) for x2
    - v2: Drift rate for x2
    - z2: Starting point for x2
    - c2: Noise for x2
    - k: Autofeedback (negative for decay)
    - b: Lateral feedback (negative for inhibition)
    - a: Threshold.

    The model equation for each accumulator is

        | x_{t+1} = x_{t} + dt * (input_{t} + k * x_t + b * y_t) + sqrt(dt) * ε;
        | x_{0} = z * a
        | ε ~ Normal(0, c);
        | input_{t} = where(time > t, v, 0)

    where y_t is the state of the OTHER accumulator at time t.

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
                 par_names = ['t1', 'v1', 'z1', 'c1', 't2', 'v2', 'z2', 'c2', 'k', 'b', 'a'],
                 max_time = 5., dt=.001,
                 bounds=None):
        super(LCA, self).__init__(trial_func=run_lca,
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
                                      'k': 'Autofeedback (negative for decay)',
                                      'b': 'Lateral feedback (negative for inhibition)',
                                      'a' : 'Threshold'
                                  },
                                  max_time = max_time, dt=dt,
                                  bounds=None,
                                  n_traces=2)
        self.name = 'Leaky Competing Accumulator'
