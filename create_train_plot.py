import pymc3 as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

x_real = pd.Series.from_csv("data/canela.csv").values
x_train = x_real[:(365*24)]

with open("arma/results.pkl","rb") as f:
    trace = pickle.load(f)

mean_train = np.mean(trace["y"],axis=0)
lower_train = np.percentile(trace["y"],2.5,axis=0)
upper_train =  np.percentile(trace["y"],97.5,axis=0)
plt.figure(figsize=(15,4))
plt.plot(mean_train[:100],label="ARMA(1,1) - Mean")
plt.fill_between(range(100),lower_train[:100],upper_train[:100],alpha=0.3, label="2.5% and 97.5% Quantile")
plt.plot(x_train[:100],label="Real")
plt.legend()
plt.savefig("model_fit.pdf")
