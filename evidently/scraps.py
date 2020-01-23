
# ## Interactive stuff


# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 3
# s = a0*np.sin(2*np.pi*f0*t)
# l, = plt.plot(t, s, lw=2, color='red')
# plt.axis([0, 1, -10, 10])


# axcolor = 'lightgoldenrodyellow'
# axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

# sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
# samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


# def update(val):
#     amp = samp.val
#     freq = sfreq.val
#     l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#     fig.canvas.draw_idle()
# sfreq.on_changed(update)
# samp.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# def reset(event):
#     sfreq.reset()
#     samp.reset()
# button.on_clicked(reset)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()
# radio.on_clicked(colorfunc)

# plt.show()


# def do_accum_trial(pars, dt=.001, nt=5000):
#     '''
#     Parameters:
#     t0: Onset of evidence accumulation
#     v: Drift rate after t0
#     k: Decay
#     z: Mean starting point (in % of a)
#     c: Noise standard deviation = 0.1
#     dv: Drift rate SD
#     dz: Starting point SD
#     Fixed parameter:
#     a: Threshold
#     Returns:
#     X (trace), response, rt
#     '''
#     t0, v, k, z, c, dv, dz = pars
#     a = 1.0
#     kill = 0
#     sqrt_dt = np.sqrt(dt)
#     NOISE = np.random.normal(loc=0., scale=c, size=nt)
#     X = np.zeros(nt)
#     X[:] = np.nan
#     x = np.random.normal(z, dz)
#     v = np.random.normal(v, dv)
#     for t in range(nt):
#         if t*dt > t0:
#             drift = v
#         else:
#             drift = 0.
#         dx = dt*(drift - x*k) + NOISE[t]*sqrt_dt
#         X[t] = x
#         x += dx
#         if(np.abs(x) > a):
#             v = 0.
#             k = 10
#     rt = np.argmax(X > a) - 1 # In msec
#     response = np.sign(X[int(rt)])
#     if rt == 0:
#         rt = np.nan
#         response = 0
#     return X, response, rt


# pars = [.2, 1., .1, 0., .1, .1, .1]
# par_names = [ 't0', 'v', 'k', 'z', 'c', 'dv', 'dz']

# model = BaseAccumulatorModel(trial_func = do_accum_trial, pars=pars, max_time=8)
# model.plot_single_trial()
# X, R, T = model.do_dataset(max_time=8)

# # def plot_pars(pars):
# #     X, R, T = do_dataset(pars, 5., 20)
# #     plot_both(X, R, T)

# # plateau, steepness = 10., 2.
# # pars = t0, v, k, z, c, dv, dz = [1., plateau*steepness, steepness, 0., 0., 0., 0.]
# # plot_pars(pars)


# def _run_trial(t0, v, k, z, c, dt=.001, nt=5000, a=1.):
#     '''
#     Parameters:
#     t0: Onset of evidence accumulation
#     v: Drift rate after t0
#     k: Decay
#     z: Mean starting point (in % of a)
#     c: Noise standard deviation = 0.1
#     Fixed parameter:
#     a = 1.: Threshold
#     Returns:
#     X (trace), response, rt
#     '''
#     sqrt_dt = np.sqrt(dt)
#     NOISE = np.random.normal(loc=0., scale=c, size=nt)
#     x = z
#     X = np.zeros(nt)
#     X[:] = np.nan
#     for t in range(nt):
#         if t*dt > t0:
#             drift = v
#         else:
#             drift = 0.
#         dx = dt*(drift - x*k) + NOISE[t]*sqrt_dt
#         X[t] = x
#         x += dx
#         if(np.abs(x) > a):
#             v = 0.
#             k = 10
#     rt = np.argmax(np.abs(X) > a) - 1 # In msec
#     response = np.sign(X[int(rt)])
#     if rt < 1:
#         rt = np.nan
#         response = 0
#     return X, response, rt
# run_trial = jit(_run_trial, nopython=True)

# if type(trial_func) != nb.targets.registry.CPUDispatcher:
#     try:
#         compiled_trial_func = nb.jit(trial_func)
#         _ = compiled_trial_func()
#     except TypingError:
#         try:
# type(trial_func)
# # self.trial_func = trial_func


# Manual KDE
# @nb.jit()
# def K(x):
#     if x >= -1 and x <= 1:
#         return .75 * (1-x**2)
#     else:
#         return 0

# def ll_single_obs(x, sims, h=None):
#     if h is None:
#         h = bw_silverman(sims)
#     return np.mean( [ (x - sims[j])/h for j in range(len(sims)) ] ) * 1./h

# def ll_observed(X, sims):
#     h = bw_silverman(sims)
#     return np.sum([ll_single_obs(X[i], sims, h=h) for i in range(len(X))])
