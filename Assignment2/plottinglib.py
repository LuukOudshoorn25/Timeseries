import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

def plot_raw_data(df):
    times = df.index
    logreturns = df.logreturns
    transformed = df.transformed_returns

    fig,[[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(5, 3.5))
    ax1.plot(times, logreturns, lw=1, color='black')
    ax2.plot(times, transformed, lw=1, color='black')
    ax3.hist(logreturns)
    sm.graphics.tsa.plot_acf(logreturns, lags=40, ax=ax4)
    plt.tight_layout()
    # plt.xlabel('Year')
    plt.show()


def QMLplot(df):
    fig, ax = plt.subplots()
    ax.hist(df.transformed_returns)
    plt.tight_layout()
    plt.show()