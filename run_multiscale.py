import pymc3 as pm
from pymc3.step_methods.metropolis import Metropolis
from theano import scan, shared

import numpy as np
import pandas as pd
import pickle

def build_model(xx,yy):
    x = shared(xx)
    y = shared(yy)
    with pm.Model() as arma_model:
        sigma = pm.HalfNormal('sigma', 5.)
        theta_x = pm.Normal('theta_x', 0., sd=1.)
        phi_x = pm.Normal('phi_x', 0., sd=2.)
        mu_x = pm.Normal('mu_x', 3., sd=3.)
        theta_y = pm.Normal('theta_y', 0., sd=1.)
        phi_y = pm.Normal('phi_y', 0., sd=2.)
        mu_y = pm.Normal('mu_y', 3., sd=3.)
        y_0 = mu_y + phi_y * mu_y
        x_0 = mu_x + phi_x * mu_x

        err0 = x[0] - x_0
        def calc_next_x(this_x, last_x, err, mu_x, phi_x, theta_x):
            nu_t = mu_x + phi_x * last_x + theta_x * err
            return (this_x - nu_t,nu_t)

        def calc_next_y(this_y, last_y, err, mu_y, phi_y, theta_y):
            nu_t = mu_y + phi_y * last_y + theta_y * err
            return (this_y - nu_t,nu_t)


        (err_x,xs), _ = scan(fn=calc_next_x,
                      sequences=dict(input=x),
                      outputs_info=[err0,x_0],
                      non_sequences=[mu_x, phi_x, theta_x])
        (err_y,ys), _ = scan(fn=calc_next_y,
                      sequences=dict(input=y),
                      outputs_info=[err0,x_0],
                      non_sequences=[mu_y, phi_y, theta_y])
        ys = pm.Deterministic("ys",ys)
        xs = pm.Deterministic("x",xs)
        p_x = pm.Potential('like_x', pm.Normal.dist(0, sd=sigma).logp(err_x))
        p_y = pm.Potential('like_y', pm.Normal.dist(0, sd=sigma).logp(err_y))
        p_total = pm.Deterministic("p_t", p_x*p_y.repeat(6))
    return arma_model

def run(xx,yy,n_samples=1000,tune=1000):
    model = build_model(xx,yy)
    with model:
        db = pm.backends.Text('multiscale')
        trace = pm.sample(draws=n_samples,
                          tune=tune,
                          step= Metropolis(),
                          trace=db,njobs=8)
    return trace


if __name__ == '__main__':
    x = pd.Series.from_csv("data/canela.csv").values.astype(np.float32)
    x_train = x[:(24*365)]
    y_train = np.mean(x_train.reshape((-1,6)),axis=1)
    x_test = x[(24*365):(24*365)+(24*60)]
    y_test = np.mean(x_test.reshape((-1,6)),axis=1)
    tr = run(x_train,y_train,10,10)
    with open("multiscale/results.pkl","wb") as f:
        pickle.dump(tr,f)
