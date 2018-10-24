import pymc3 as pm
from theano import scan, shared

import numpy as np
import pandas as pd

def build_model(xx):
    x = shared(xx)
    with pm.Model() as arma_model:
        sigma = pm.HalfNormal('sigma', 5.)
        theta = pm.Normal('theta', 0., sd=1.)
        phi = pm.Normal('phi', 0., sd=2.)
        mu = pm.Normal('mu', 0., sd=10.)
        y_0 = mu + phi * mu

        err0 = x[0] - y_0

        def calc_next(this_x, last_y, err, mu, phi, theta):
            nu_t = mu + phi * last_y + theta * err
            return (this_x - nu_t,nu_t)

        (err,ys), _ = scan(fn=calc_next,
                      sequences=dict(input=x),
                      outputs_info=[err0,y_0],
                      non_sequences=[mu, phi, theta])
        y = pm.Deterministic("y",ys)
        pm.Potential('like', pm.Normal.dist(0, sd=sigma).logp(err))
    return arma_model

from pymc3.step_methods.metropolis import Metropolis
def run(xx,n_samples=1000):
    model = build_model(xx)
    with model:
        db = pm.backends.NDArray('arima')
        trace = pm.sample(draws=n_samples,
                          tune=1000,
                          step= Metropolis(),
                           # nuts_kwargs=dict(target_accept=.99)
                          trace=db)

    return trace


if __name__ == '__main__':
    x = pd.Series.from_csv("data/canela.csv").values
    x_train = x[:17520]
    x_test = x[17520:26280]
    # x = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)
    tr = run(x,1000)
