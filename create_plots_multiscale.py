import pymc3 as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns


x_real = pd.Series.from_csv("~/simulations/data/canela.csv").values
x_train = x_real[:(365*24)]
x_test = x_real[(24*365):(24*365)+(24*60)]

trace_ms =pm.backends.text.load("multiscale")

g = sns.jointplot(x="theta_x", y="phi_x", data=df_ms, kind="kde")
g.ax_joint.legend_.remove()
plt.savefig("img/multiscale/posterior_joint_x.pdf")

g = sns.jointplot(x="theta_y", y="phi_y", data=df_ms, kind="kde")
g.ax_joint.legend_.remove()
plt.savefig("img/multiscale/posterior_joint_y.pdf")

pm.plot_posterior(trace_ms,varnames=["theta_x","theta_y","phi_x","phi_y"])
plt.savefig("img/multiscale/posterior_hist.pdf")

y_train = np.mean(x_train.reshape((-1,6)),axis=1)
y_test = np.mean(x_test.reshape((-1,6)),axis=1)

mean_train = np.mean(trace_ms["x"],axis=0)
plt.plot(mean_train[:100],label="ARMA(1,1)")
plt.plot(x_train[:100],label="Real")
plt.title("$x_{t}$ variable")
plt.legend()
plt.savefig("img/multiscale/x_mode_fit.pdf")

mean_train = np.mean(trace_ms["ys"],axis=0)
plt.plot(mean_train[:100],label="ARMA(1,1)")
plt.plot(y_train[:100],label="Real")
plt.title("$y_{t}$ variable")
plt.legend()
plt.savefig("img/multiscale/y_mode_fit.pdf")
