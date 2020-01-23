
# Evidently: Simulate Evidence Accumulation Models in Python

`Evidently` is a python package for working with evidence accumulation models.

It provides

- Efficient functions for simulating data from a range of models.
- Classes that make it easier to tweak model parameters and manage simulated data.
- A consistent way to implement new models.
- Visualisation, including interactive widgets for Jupyter.
- Kernel density-based methods for estimating 
  the likelihood of real data under a given model/set of parameters,
  allowing parameter estimation and model comparision.
  


## Basic Use


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evidently
```

    /home/eoin/miniconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.kde module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)


## Set up a model and provide parameters


```python
model = evidently.models.Diffusion(pars=[1., .5, -.25, .8, .4], max_time=5., dt=.001)
model
```




    Classic Drift Diffusion Model
    Parameters: [t0 = 1.00, v = 0.50, z = -0.25, a = 0.80, c = 0.40]




```python
model.describe_parameters()
```

    Parameters for Classic Drift Diffusion Model:
    - t0   : 1.00  ~ Non-decision time
    - v    : 0.50  ~ Drift rate
    - z    : -0.25 ~ Starting point
    - a    : 0.80  ~ Threshold (±)
    - c    : 0.40  ~ Noise SD


## Simulate data


```python
X, responses, rts = model.do_dataset(n=1000)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.000</th>
      <th>0.001</th>
      <th>0.002</th>
      <th>0.003</th>
      <th>0.004</th>
      <th>0.005</th>
      <th>0.006</th>
      <th>0.007</th>
      <th>0.008</th>
      <th>0.009</th>
      <th>...</th>
      <th>4.990</th>
      <th>4.991</th>
      <th>4.992</th>
      <th>4.993</th>
      <th>4.994</th>
      <th>4.995</th>
      <th>4.996</th>
      <th>4.997</th>
      <th>4.998</th>
      <th>4.999</th>
    </tr>
    <tr>
      <th>sim</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.205269</td>
      <td>-0.206654</td>
      <td>-0.199312</td>
      <td>-0.208307</td>
      <td>-0.200256</td>
      <td>-0.188223</td>
      <td>-0.203513</td>
      <td>-0.193312</td>
      <td>-0.211066</td>
      <td>-0.195750</td>
      <td>...</td>
      <td>2.466137</td>
      <td>2.459109</td>
      <td>2.467809</td>
      <td>2.471437</td>
      <td>2.474827</td>
      <td>2.482113</td>
      <td>2.474570</td>
      <td>2.462066</td>
      <td>2.460595</td>
      <td>2.469467</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.212663</td>
      <td>-0.229858</td>
      <td>-0.248795</td>
      <td>-0.234927</td>
      <td>-0.246239</td>
      <td>-0.241695</td>
      <td>-0.242164</td>
      <td>-0.231270</td>
      <td>-0.238369</td>
      <td>-0.242416</td>
      <td>...</td>
      <td>-0.299059</td>
      <td>-0.310426</td>
      <td>-0.316268</td>
      <td>-0.308877</td>
      <td>-0.313708</td>
      <td>-0.300249</td>
      <td>-0.290620</td>
      <td>-0.289434</td>
      <td>-0.298776</td>
      <td>-0.306962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208901</td>
      <td>-0.185590</td>
      <td>-0.206732</td>
      <td>-0.209250</td>
      <td>-0.191910</td>
      <td>-0.208846</td>
      <td>-0.218917</td>
      <td>-0.226665</td>
      <td>-0.222832</td>
      <td>-0.254173</td>
      <td>...</td>
      <td>1.703630</td>
      <td>1.712503</td>
      <td>1.709552</td>
      <td>1.726211</td>
      <td>1.726535</td>
      <td>1.725116</td>
      <td>1.725894</td>
      <td>1.720248</td>
      <td>1.710661</td>
      <td>1.702784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.199132</td>
      <td>-0.218318</td>
      <td>-0.229405</td>
      <td>-0.225806</td>
      <td>-0.218270</td>
      <td>-0.239541</td>
      <td>-0.254558</td>
      <td>-0.261797</td>
      <td>-0.244953</td>
      <td>-0.258269</td>
      <td>...</td>
      <td>3.560717</td>
      <td>3.568986</td>
      <td>3.556551</td>
      <td>3.573069</td>
      <td>3.589833</td>
      <td>3.584273</td>
      <td>3.571641</td>
      <td>3.583492</td>
      <td>3.578438</td>
      <td>3.579548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.233813</td>
      <td>-0.214260</td>
      <td>-0.221887</td>
      <td>-0.228352</td>
      <td>-0.242280</td>
      <td>-0.245695</td>
      <td>-0.234245</td>
      <td>-0.253752</td>
      <td>-0.256572</td>
      <td>-0.259053</td>
      <td>...</td>
      <td>1.460541</td>
      <td>1.478860</td>
      <td>1.476589</td>
      <td>1.485294</td>
      <td>1.499031</td>
      <td>1.496546</td>
      <td>1.477432</td>
      <td>1.476409</td>
      <td>1.459197</td>
      <td>1.455652</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5000 columns</p>
</div>




```python
print(responses[:5]) 
print(rts[:5])
```

    [ 1.  0. -1.  1.  1.]
    [2.49    nan 0.538 1.716 4.392]


## Visualise

The `evidently.viz` submodule contains a collection of `matplotlib`-based functions for visualising model simulations. Here are a few examples.


```python
ax = evidently.viz.setup_ddm_plot(model) # Uses model info to draw bounds.
evidently.viz.plot_trace_mean(model, X, ax=ax); # Plots simulations
```


![png](README_files/README_12_0.png)



```python
ax = evidently.viz.setup_ddm_plot(model)
evidently.viz.plot_traces(model, X, responses, rts, ax=ax, 
                          terminate=True, show_mean=True); # Show raw data
```

    /home/eoin/miniconda3/lib/python3.7/site-packages/evidently/viz.py:161: RuntimeWarning: invalid value encountered in greater
      X.iloc[i, t > rt] = np.nan



![png](README_files/README_13_1.png)



```python
ax = evidently.viz.setup_ddm_plot(model)
for resp in [1, -1]:
    mask = (responses == resp) # Split by response
    evidently.viz.plot_trace_mean(model, X[mask], ax=ax, label='Response: %i' % resp)
plt.legend();
```


![png](README_files/README_14_0.png)



```python
mX = evidently.utils.lock_to_movement(X, rts, duration=2) # Time-lock to threshold crossing
ax = evidently.viz.setup_ddm_plot(model, time_range=(-2, 0))
evidently.viz.plot_traces(model, mX, responses, rts, ax=ax, show_mean=True);
```


![png](README_files/README_15_0.png)



```python
ax = evidently.viz.setup_ddm_plot(model, time_range=(-2, 0))
for resp in [1, -1]:
    mask = responses == resp
    resp_mX = evidently.utils.lock_to_movement(X[mask], rts[mask])
    evidently.viz.plot_trace_mean(model, resp_mX, ax=ax, label='Response: %i' % resp)
plt.legend();
```


![png](README_files/README_16_0.png)


There high-level functions can create multi-axis figures.


```python
evidently.viz.visualise_model(model, model_type='ddm', measure='means');
```


![png](README_files/README_18_0.png)


## Interactive Visualisation

Using the `ipywidgets` package, we can wrap high level visualisation functions like `accum.viz.visualise_ddm` in a call to `ipywidgets` to make them interactive.

To try the interactive plots, download this repository to your own computer,
or run the code in the cloud by visiting [this Binder notebook]().


![](./imgs/interactive.gif)



```python
from ipywidgets import interact, FloatSlider
def fs(v, low, high, step, desc=''):
    return FloatSlider(value=v, min=low, max=high, step=step, description=desc, continuous_update=False)

def ddm_simulation_plot(t0=1., v=.5, z=0., a=.5, c=.1):
    model = evidently.Diffusion(pars=[t0, v, z, a, c])
    evidently.viz.visualise_model(model)
    title = 't0 = %.1f, Drift = %.1f, Bias = %.1f, Threshold = %.1f; Noise SD = %.1f' % (t0, v, z, a, c)
    plt.suptitle(title, y=1.01)

interact(ddm_simulation_plot,
         t0  = fs(1., 0, 2., .1,   't0'),
         v   = fs(.5, 0, 2., .1,   'Drift'),
         z   = fs(0., -1., 1., .1,  'Bias'),
         a     = fs(.5, 0., 2., .1,   'Threshold'),
         c   = fs(.1, 0., 1., .1,   'Noise SD'));
```


![png](README_files/README_21_0.png)


# Other Models

The following model classes are currently available:

- Diffusion
- Wald
- HDiffision (Hierarchical Diffusion)
- HWald (Hierarchical Wald)
- Race

See the [API](http://eointravers.com/code/evidently/api.html) for more details.

# Road Map


## More Models!

I have already implemented several of these models, but have to integrate them with the rest of the package.

- Leaky Competing Accumulator model.
- LCA/Race models with > 2 options.
- Leaky/unstable Diffusion.
- Time-varying parameters, including
    - Collapsing decision bounds
    - Time-varying evidence
- Heirarchical models with regressors that differ across trials.

## Reparameterisation

Ideally, parameterisation with other packages used for fitting accumulator models 
such as [HDDM]() and [PyDDM](), (for Python) 
and [rtdists]() and [DMC]() (for R). 
This would make it possible to efficiently fit models using those packages, 
then explore their dynamics here.

Model probably should also specify default parameters.

##  Visualisation

There's no shortage of ways to visualise accumulator models. 
Future versions will include both more low-level plotting functions
and high-level wrappers.

I'll also be implementing vector field plots, e.g. Figure 2 of 
[Bogacz et al. (2007)](https://people.socsci.tau.ac.il/mu/usherlab/files/2014/03/m2.pdf).

## Likelihood


The `evidently.likelihood` model contains functions for estimating 
the likelihood of data $x$ under parameters $\theta$ and model $M$,
based on the "likelihood-free" technique introduced by 
[Turner and Sederberg (2007)](https://link.springer.com/article/10.3758/s13423-013-0530-0).
These functions aren't properly tested yet,
and haven't been documented.


