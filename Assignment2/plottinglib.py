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
    returns = df.logreturns
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
    plt.savefig('data_SP500.pdf', bbox_inches='tight')
    # plt.xlabel('Year')
    
    plt.show()


def QMLplot(df):
    fig, ax = plt.subplots()
    ax.hist(df.transformed_returns)
    plt.tight_layout()
    plt.show()

def plot_smoothed(df, filtered_alphas, smoothed_alphas, xi, fname='KF_pounddollarexchange.pdf'):
    fig, axs = plt.subplots(2, sharex=True, figsize=(6,2.5))
    axs[0].scatter(df.index,df['transformed_returns'],s=1,color='black')
    axs[0].plot(df.index[1:], filtered_alphas,lw=1,color='red', label='Filtered Signal')
    axs[0].plot(df.index[1:], smoothed_alphas, lw=1, color='blue', label='Smoothed Signal')
    axs[0].set_ylabel(r'$\log (y_t-\overline{y})^2$')
    # axs[0].legend()
    axs[1].scatter(df.index, df['logreturns']*100, s=1, color='black')
    axs[1].plot(df.index[1:], filtered_alphas-xi, lw=1,color='red', label='Filtered Signal')
    axs[1].plot(df.index[1:], smoothed_alphas-xi, lw=1, color='blue', label='Smoothed Signal')
    # axs[1].legend()
    axs[1].set_ylabel('$y_t$')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    