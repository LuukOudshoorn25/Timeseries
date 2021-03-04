import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

def make_titles(axes):
    for i in range(len(axes)):
        axes[i].set_title('('+str(i+1)+')   ',loc='right')
    return axes


def plot_raw_data(df):
    times = df.index
    returns = df.returns
    transformed = df.transformed_returns

    fig,[[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6.5, 5))
    ax1.plot(times, returns, lw=0.5, color='black')
    ax2.plot(times, transformed, lw=0.5, color='black')
    ax3.hist(returns,bins=40)
    sm.graphics.tsa.plot_acf(returns, lags=15, ax=ax4)
    ax4.set_xlim(0.8,16)
    ax4.set_title('')
    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])
    plt.tight_layout()
    plt.savefig('data_fig.pdf', bbox_inches='tight')
    # plt.xlabel('Year')
    
    plt.show()


def QMLplot(df):
    fig, ax = plt.subplots()
    ax.hist(df.transformed_returns)
    plt.tight_layout()
    plt.show()

def plot_smoothed(x, y, alphas, fname):
    fig, ax = plt.subplots(1, figsize=(6,2.5))
    ax.scatter(x,y,s=1,color='black')
    ax.plot(x,alphas,lw=1,color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\log (y_t-\overline{y})^2$')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    