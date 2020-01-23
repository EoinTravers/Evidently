import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class MetropolisSampler(object):
    def __init__(self, func, par_names, starting_pars, proposal_sd):
        self.func = func
        self.par_names = par_names
        self.starting_pars = starting_pars
        self.pars = starting_pars
        self.proposal_sd = proposal_sd
        self.iterations = 0
        self.jumps = 0
        self.n_pars = len(starting_pars)
        assert(len(starting_pars) == len(proposal_sd))
        n_iter = 1e+6
        self.parameter_chain = np.zeros((n_iter, self.n_pars))  # Start with room for 1 million samples
        selt.results = None
        # For debugging
        self.ll_chain = np.zeros(n_iter)
        self.p_accept_chain = np.zeros(n_iter)

    def expand_chains(self, expand_by=1e+6):
        new_par_chain = np.zeros(self.interations + expand_by, self.n_pars)
        new_par_chain[:self.iterations] = self.parameter_chain
        self.parameter_chain = new_par_chain
        # Loop the rest

        def expand_1d_chain(chain, by=expand_by):
            i = len(chain)
            new_chain = np.zeros(i + by)
            new_chain[:i] = chain
            return new_chain
        self.ll_chain = expand_1d_chain(self.ll_chain, by=expand_by)
        self.p_accept_chain = expand_1d_chain(self.p_accept_chain, by=expand_by)

    def sample_once(self):
        ix = self.iterations
        ll_old = self.func(self.pars)
        proposal = stats.t.rvs(df=1, loc=self.pars, scale=self.proposal_sd)
        ll_new = self.func(proposal)
        p_accept = np.exp(ll_new - ll_old)
        self.proposal_chain[ix] = ll_new
        p_accept_chain[ix] = p_accept
        p_accept = min(p_accept, 1)
        if np.random.binomial(1, p_accept):
            self.pars = proposal
            self.jumps += 1
        self.parameter_chain[ix] = self.pars
        self.iterations += 1
        if self.iterations > self.parameter_chain.shape[0]:
            self.expand_chains()

    def sample(self, n):
        for i in range(n):
            self.sample_once()

    def update_results(self, thin=4):
        samples = pd.DataFrame(self.parameter_chain[:ix:thin], columns=self.par_names)
        samples['ll'] = self.ll_chain
        self.results = samples

    def get_results(self, thin=4):
        self.update_results(thin=thin)
        return self.results

    def plot_chains(par_names=None, thin=4):
        if par_names is None:
            par_names = self.par_names
        self.update_results(thin=thin)
        n = len(par_names)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(self.par_names[i])
            x = parameter_chain[:ix:thin, i]
            plt.plot(x)
