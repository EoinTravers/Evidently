import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from scipy import stats
# from scipy.optimize import curve_fit
# from numba import jit
# from sklearn.neighbors.kde import KernelDensity


from . import viz

class BaseModel(object):
    '''Base class for accumulator models.
    This class doesn't do anything by itself, but all other model
    classes should inherit from it.
    '''
    def __init__(self,
                 trial_func=None,
                 par_names=[], pars = [],
                 par_descriptions = None,
                 n_traces=1,
                 max_time = 5., dt=.001,
                 stimuli=None,
                 bounds=None,
                 name='Base Accumulator model'):
        self.par_dict = {}  # Order is important here.
        self.par_names = par_names
        self.pars = np.array(pars)
        self.par_descriptions = par_descriptions
        self.max_time = max_time
        self.dt = dt
        self.stimuli = stimuli
        self.trial_func = trial_func
        self.bounds = bounds
        self.n_traces = n_traces
        self.name = name
        if len(pars) > 0 and len(par_names) > 0:
            assert(len(par_names) == len(pars)), 'Wrong number of parameters'

    def __repr__(self):
        txt = self.name
        par_repr = []
        for p in self.pars:
            try:
                par_repr.append('%.2f' % p)
            except TypeError:
                par_repr.append(repr(p))
        if self.pars is not None:
            txt += '\nParameters: ['
            if len(self.par_names) > 0:
                par_txt = ['%s = %s' % (k, v) for k, v in zip(self.par_names, par_repr)]
                txt += ', '.join(par_txt)
                txt += ']'
            else:
                txt += ', '.join(['%s' % v for v in par_repr]) + ']'
        if self.stimuli is not None:
            if len(self.stimuli) == 0:
                txt += '\nStimuli not set.'
            else:
                txt += '\nStimuli:\n '
                stim_rep = repr(self.stimuli)
                if len(stim_rep) > 70:
                    txt += stim_rep[:70] + ' [...]'
                else:
                    txt += stim_rep
        return txt

    def __setattr__(self, name: str, value):
        '''Overwrite the built-in method for setting attributes.'''
        if name == 'pars':
            # Whenever 'pars' set, make sure it's an array, and update the dict.
            value = np.array(value)
            self.__dict__['par_dict'] = dict(zip(self.par_names, value))
        if name == 'stimuli':
            # Should I be doing something here?
            pass
        self.__dict__[name] = value

    def describe_parameters(self):
        '''Pretty-prints the values of model.par_descriptions.'''
        print('Parameters for %s:' % self.name)
        if self.par_descriptions is None:
            print('No parameter descriptions found.')
        else:
            is_set = len(self.pars) == len(self.par_names)
            # else:
            #     pars = ['<not set>'] * len(self.par_names)
            #     par_dict = dict(zip(self.par_names, pars))
            for p in self.par_names:
                if is_set:
                    v = '%.2f' % self.par_dict[p]
                else:
                    v = '<not set>'
                print('- {:<5}: {:<5} ~ {}'.format(p, v, self.par_descriptions[p]))


    def _do_trial(self, pars=None, *args, **kwargs):
        '''Simulate a single trial. Not currently used, but useful for debugging.'''
        res = self._do_trials(n=1, *args, **kwargs)
        res = [v[0] for v in res]
        return res

    def _do_trials(self, n=50, pars=None, *args, **kwargs):
        '''Simulates model and returns raw trace(s), responses and rts as numpy arrays.
        For more polished output, use model.do_dataset() instead.
        See documentation for model.trial_func for details.

            n: Number of trials to simulate
            pars: None (default) or list.
                  If None, model.pars used. Otherwise, model.pars is updated.
            Other arguments are passed to model.trial_func.
        Returns:
            X or [X1, X2]: Numpy of accumulator traces for each simulation.
                           If n_traces > 1, returns a list of arrays, one per accumulator.
            responses: Array of responses (-1, 0, 1)
            rts: Array of response times (seconds)
        '''
        self.pars = np.array(self.pars)
        if pars is None:
            pars = self.pars
        return self.trial_func(pars, n=n, *args, **kwargs)

    def do_dataset(self, n=50, pars=None, max_time=None):
        """Simulate a dataset from the model.
        This is the most important function in the package.

    Args:
        n: Number of trials to simulate.
        pars (list or array):

            | A list of model paramters.
            | If missing, defaults to `model.pars`
            | See `model.trial_func?` or model `model.describe_parameters()` for more information.
        max_time (int or float):
            How many seconds to simulate? Defaults to `model.max_time`

    Returns:
        (pandas.DataFrame, array, array): tuple containing:

            - X (pandas.DataFrame):
              A data frame of the `n` simulated accumulator trajectories over time.

              - Columns indicate the time, in seconds.
              - If model has only a single accumulator, the index indicates the simulation number.
              - If model has multiple accumulators, indices indicate sim. number, and accumulator number.
              - If trial-by-trial inputs are provided, indices indicate input number, and sim. number. This feature is not implemented yet.
            - responses (np.ndarray): +/-1 if upper/lower threshold crossed, 0 otherwise.
            - rts (np.ndarray): Time of threshold crossing, in seconds, or np.NaN

        """
        if max_time is None:
            max_time = self.max_time
        pars = np.array(pars)
        dt = self.dt
        nt = int(max_time / dt)
        if self.stimuli is None:
            X, responses, rts = self._do_trials(n=n, nt=nt)
            if self.n_traces == 1:
                X = pd.DataFrame(X)
                X.index = X.index.rename('sim')
                X.columns *= dt
            else:
                Xout = []
                for accum, Xi in enumerate(X):
                    dfX = pd.DataFrame(Xi)
                    dfX.index = dfX.index.rename('sim')
                    dfX.columns *= dt
                    dfX['accumulator'] = accum
                    Xout.append(dfX)
                X = pd.concat(Xout).reset_index().set_index(['sim', 'accumulator']).sort_index(level='sim')

            return X, responses, rts
        else:
            raise NotImplementedError()
            # return utils.do_dataset_no_stimuli(self, n=n,
            #                                    pars=pars, dt=dt, nt=nt)

    # def plot_single_trial(self, pars=None):
    #     if self.n_traces == 1:
    #         viz.plot_single_trial_onetrace(self, pars)
    #     else:
    #         raise NotImplementedError('TODO: Plotting functions for multiple accumulators.')
