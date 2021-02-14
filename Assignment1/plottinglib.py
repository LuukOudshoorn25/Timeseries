import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf

def make_titles(axes):
    for i in range(len(axes)):
        axes[i].set_title('('+str(i+1)+')   ',loc='right')
    return axes

def plot_raw_data(df, var_name='Volume of Nile'):
    plt.figure()
    plt.scatter(df.index, df.dep_var,s=2,color='black')
    plt.tight_layout()
    plt.xlabel('Year')
    plt.ylabel(var_name)
    plt.tight_layout()

    fname = var_name + 'raw_data.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_fig2_1(times, y, a, std, P, v, F,fname, var_name='Volume of Nile'):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], a[1:],color='black',lw=1)
    ax1.scatter(times[1:], y[1:], s=1,color='black')
    ax1.plot(times[1:], a[1:]+1.645*std[1:],color='grey',lw=0.7)
    ax1.plot(times[1:], a[1:]-1.645*std[1:],color='grey',lw=0.7)
    ax1.set_ylabel(var_name)
    

    ax2.plot(times[1:], P[1:],color='black',lw=1)
    ax2.set_ylabel('Filtered state variance')
    

    ax3.plot(times[1:], v[1:],color='black',lw=1)
    ax3.set_ylabel('Prediction errors')
    

    ax4.plot(times[1:], F[1:],color='black',lw=1)
    ax4.set_ylabel('Prediction variance')
    
    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])
    
    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()

def plot_fig2_2(times, y, alphas, std, V, r, N, fname, var_name='Volume of Nile'):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times, alphas,color='black',lw=1)
    ax1.scatter(times, y, s=1,color='black')
    ax1.plot(times[1:], alphas[1:]+1.645*std,color='grey',lw=0.7)
    ax1.plot(times[1:], alphas[1:]-1.645*std,color='grey',lw=0.7)
    ax1.set_ylabel(var_name)

    ax2.plot(times[1:], V[1:],color='black',lw=1)
    ax2.set_ylabel('Smoothed state variance')

    ax3.plot(times, r,color='black',lw=1)
    ax3.set_ylabel('Smoothing cumulant')
    ax3.axhline(0,ls='--',lw=0.5,color='black')
    
    ax4.plot(times, N,color='black',lw=1)
    ax4.set_ylabel('Smoothing variance cumulant')
    
    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()

def plot_fig2_3(times, eps_hat, var_eps, eta_hat,var_eta,fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times, eps_hat,color='black',lw=1)
    ax1.set_ylabel('Observation error')
    ax1.axhline(0,ls='--',lw=0.5,color='black')


    ax2.plot(times, np.sqrt(var_eps),color='black',lw=1)
    ax2.set_ylabel('Observation error variance')

    ax3.plot(times, eta_hat,color='black',lw=1)
    ax3.set_ylabel('State error')
    ax3.axhline(0,ls='--',lw=0.5,color='black')
    
    ax4.plot(times, np.sqrt(var_eta),color='black',lw=1)
    ax4.set_ylabel('State error variance')

    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()


def plot_fig2_5(times, y,a,P,alphas,V, fname, var_name='Volume of Nile'):
    ylabel_name = var_name + ' (filtered state)'
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], a[1:],color='tomato',lw=0.7)
    ax1.plot(times, y, color='black',lw=0.7)
    ax1.set_ylabel(ylabel_name)

    ax2.plot(times[1:], P[1:],color='black',lw=1)
    ax2.set_ylabel('Filtered state variance')

    ax3.plot(times[1:], alphas[1:],color='tomato',lw=0.7)
    ax3.plot(times, y,color='black',lw=0.7)
    ax3.set_ylabel('Smoothed state')
    
    ax4.plot(times[1:-2], V[1:-2],color='black',lw=1)
    ax4.set_ylabel('Smoothed state variance')

    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])

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
    
    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_fig2_8(times, obs_res, stat_res, fname):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(5, 3.5))
    ax1.plot(times, obs_res, color='black', lw=0.7)
    ax1.plot(times, np.zeros(len(obs_res)), color='black', lw=0.7)
    sns.distplot(obs_res, hist=True, bins=13, kde=True, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 0.7}, ax=ax2)

    ax3.plot(times, stat_res, color='black', lw=0.7)
    ax3.plot(times, np.zeros(len(stat_res)), color='black', lw=0.7)
    sns.distplot(stat_res, hist=True, bins=13, kde=True, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 0.7}, ax=ax4)

    ax1, ax2, ax3, ax4 = make_titles([ax1,ax2,ax3,ax4])                 

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()