import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf
def plot_raw_data(df):
    plt.figure()
    plt.scatter(df.index, df.volume,s=2,color='black')
    plt.tight_layout()
    plt.xlabel('Year')
    plt.ylabel('Volume of Nile')
    plt.show()

def plot_fig2_1(times, y, a, std, P, v, F,fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], a[1:],color='black',lw=1)
    ax1.scatter(times[1:], y[1:], s=1,color='black')
    ax1.plot(times[1:], a[1:]+1.645*std[1:],color='grey',lw=0.7)
    ax1.plot(times[1:], a[1:]-1.645*std[1:],color='grey',lw=0.7)
    ax1.set_ylabel('Volume of Nile')

    ax2.plot(times[1:], P[1:],color='black',lw=1)
    ax2.set_ylabel('Filtered state variance')

    ax3.plot(times[1:], v[1:],color='black',lw=1)
    ax3.set_ylabel('Prediction errors')
    
    ax4.plot(times[1:], F[1:],color='black',lw=1)
    ax4.set_ylabel('Prediction variance')

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()

def plot_fig2_2(times, y, alphas, std, V, r, N, fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], alphas[1:],color='black',lw=1)
    ax1.scatter(times, y, s=1,color='black')
    ax1.plot(times[1:], alphas[1:]+1.645*std,color='grey',lw=0.7)
    ax1.plot(times[1:], alphas[1:]-1.645*std,color='grey',lw=0.7)
    ax1.set_ylabel('Volume of Nile')

    ax2.plot(times[1:-1], V[1:-1],color='black',lw=1)
    ax2.set_ylabel('Smoothed state variance')

    ax3.plot(times[1:], r[1:],color='black',lw=1)
    ax3.set_ylabel('Smoothing cumulant')
    ax3.axhline(0,ls='--',lw=0.5,color='black')
    
    ax4.plot(times[:-1], N[:-1],color='black',lw=1)
    ax4.set_ylabel('Smoothing variance cumulant')

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()

def plot_fig2_3(times, eps_hat, var_eps, eta_hat,var_eta,fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times, eps_hat,color='black',lw=1)
    ax1.set_ylabel('Observation error')

    ax2.plot(times[1:], np.sqrt(var_eps[1:]),color='black',lw=1)
    ax2.set_ylabel('Observation error variance')

    ax3.plot(times, eta_hat,color='black',lw=1)
    ax3.set_ylabel('State error')
    ax3.axhline(0,ls='--',lw=0.5,color='black')
    
    ax4.plot(times, np.sqrt(var_eta),color='black',lw=1)
    ax4.set_ylabel('State error variance')

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()


def plot_fig2_5(times, y,a,P,alphas,V, fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], a[1:],color='tomato',lw=0.7)
    ax1.plot(times, y,color='black',lw=0.7)
    ax1.set_ylabel('Volume of Nile (filtered state)')

    ax2.plot(times[1:], P[1:],color='black',lw=1)
    ax2.set_ylabel('Filtered state variance')

    ax3.plot(times[1:], alphas[1:],color='tomato',lw=0.7)
    ax3.plot(times, y,color='black',lw=0.7)
    ax3.set_ylabel('Smoothed State')
    
    ax4.plot(times[1:-2], V[1:-2],color='black',lw=1)
    ax4.set_ylabel('Smoothed state Variance')

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()


def plot_fig2_7(times, eps, fname):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(5, 3.5))
    ax1.plot(times[1:], eps[1:], color='black', lw=0.7)
    ax1.plot(times[1:], np.zeros(len(eps)-1), color='black', lw=0.7)
    sns.distplot(eps, hist=True, bins=13, kde=True, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 0.7}, ax=ax2)
    
    sm.qqplot(eps, line ='45', ax=ax3,ms=1,lw=0.1)
    ax3.set_ylabel('')
    acf_ = acf(eps,nlags=10)
    ax4.bar(np.arange(len(acf_))[1:],acf_[1:],color='grey')
    ax4.axhline(0,ls='--',color='black',lw=0.5)
    ax4.set_ylim(-1,1)
    
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_fig2_8(times, obs_res, stat_res, fname):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(5, 3.5))
    ax1.plot(times[1:], obs_res[1:], color='black', lw=0.7)
    ax1.plot(times[1:], np.zeros(len(obs_res) - 1), color='black', lw=0.7)
    sns.distplot(obs_res, hist=True, bins=13, kde=True, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 0.7}, ax=ax2)

    ax3.plot(times[1:], stat_res[1:], color='black', lw=0.7)
    ax3.plot(times[1:], np.zeros(len(stat_res) - 1), color='black', lw=0.7)
    sns.distplot(stat_res, hist=True, bins=13, kde=True, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 0.7}, ax=ax4)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()