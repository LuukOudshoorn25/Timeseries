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
    parzen = np.log(df.rk_parzen)

    fig,[[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6.5, 5))
    ax1.plot(times, returns, lw=0.5, color='black')
    ax2.plot(times, transformed, lw=0.5, color='black', label='Transformed returns')
    ax2.plot(times, parzen, lw=0.5, color='red', label='Parzen volatility')
    ax2.legend()
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
    axs[0].plot(df.index[:], filtered_alphas,lw=1,color='red', label='Filtered Signal')
    axs[0].plot(df.index[:], smoothed_alphas, lw=1, color='blue', label='Smoothed Signal')
    axs[0].set_ylabel(r'$\log (y_t-\overline{y})^2$')
    # axs[0].legend()
    axs[1].scatter(df.index, df['logreturns']*100, s=1, color='black')
    axs[1].plot(df.index[:], filtered_alphas-xi, lw=1,color='red', label='Filtered Signal')
    axs[1].plot(df.index[:], smoothed_alphas-xi, lw=1, color='blue', label='Smoothed Signal')
    # axs[1].legend()
    axs[1].set_ylabel('$y_t$')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_pf(outputs):
    fig, [ax1,ax2] = plt.subplots(2, figsize=(6,3), sharex=True)
    ax1.plot(outputs[2,:],lw=0.8,color='black')
    ax1.set_ylabel('Effective Sample Size')
    ax2.plot(outputs[0,:],lw=0.8,color='black')
    ax2.plot(outputs[0,:]+1.86*np.sqrt(outputs[1,:]),lw=0.8,color='grey')
    ax2.plot(outputs[0,:]-1.86*np.sqrt(outputs[1,:]),lw=0.8,color='grey')
    plt.axhline(0,ls='--',color='grey',lw=0.5)
    ax2.set_ylabel('SP500 return rate')
    ax2.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('SP500_fig_q_f.png',dpi=500)
    plt.show()
    

def plot_Hts(signal1, signal2,signal3, estimates):
    phi, omega, sigma_eta,_ = estimates
    xi = omega / (1-phi)
    fig, ax1 = plt.subplots(1, figsize=(6,2), sharex=True)
    ax1.plot(signal1,lw=1,color='black',label='Particle Filter')
    ax1.plot(signal2-xi,lw=0.8,color='red',label='Kalman Smoother', ls='--')
    ax1.plot(signal3-xi,lw=0.6,color='dodgerblue',label='Kalman Filter', ls='--', alpha=0.9)
    plt.axhline(0,ls='--',color='grey',lw=0.5)
    ax1.set_ylabel('SP500 return rate')
    ax1.set_xlabel('Time')
    plt.legend(frameon=1)
    plt.tight_layout()
    plt.savefig('SP500_KFS_signal_vs_particle.png',dpi=500)
    plt.show()
