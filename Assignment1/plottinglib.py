import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    ax1.scatter(times, y, s=1,color='black')
    ax1.plot(times[1:], a[1:]+1.645*std,color='grey',lw=0.7)
    ax1.plot(times[1:], a[1:]-1.645*std,color='grey',lw=0.7)
    ax1.set_ylabel('Volume of Nile')

    ax2.plot(times[1:], P[1:],color='black',lw=1)
    ax2.set_ylabel('Filtered state variance')

    ax3.plot(times[1:], v[1:],color='black',lw=1)
    ax3.set_ylabel('Prediction errors')
    
    ax4.plot(times[1:-1], F[1:-1],color='black',lw=1)
    ax4.set_ylabel('Prediction variance')

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()

def plot_fig2_2(times, y, alphas, std, V, r, N, fname):
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, sharex=True,figsize=(5,3.5))
    ax1.plot(times[1:], alphas[1:],color='black',lw=1)
    ax1.scatter(times, y, s=1,color='black')
    ax1.plot(times[2:], alphas[2:]+1.645*std[1:],color='grey',lw=0.7)
    ax1.plot(times[2:], alphas[2:]-1.645*std[1:],color='grey',lw=0.7)
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

