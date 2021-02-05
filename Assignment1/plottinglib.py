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

def plot_fig2_1(times, y, a, std, P, v, F):
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
    plt.savefig('Fig21.pdf',bbox_inches='tight')
    plt.show()
