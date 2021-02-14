import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plottinglib import *
class KFpredictor():
    def __init__(self,df, init_pars, var='dep_var', var_name='Volume of Nile'):
        """Initialisation, where df is a pandas DataFrame and var is the name of the column to study and
           init_pars is a dictionary with initial values"""
        self.df = df
        self.var = var
        self.var_name = var_name
        self.y = np.array(df[var].values.flatten())
        self.times = df.index
        self.pardict = init_pars

    def iterate(self,plot=True):
        """Iterate over the observations and update the filtered values after each iteration"""
        # Create empty arrays to store values
        F = np.zeros(len(self.y))
        a = np.zeros(len(self.y))
        v = np.zeros(len(self.y))
        P = np.zeros(len(self.y))
        # Initialize at the initial values parsed to the class
        P[0] = self.pardict['P1']
        sigma_eps2 = self.pardict['sigma_eps2']
        sigma_eta2 = self.pardict['sigma_eta2']
        # initialise a for estimated model
        if self.var_name != 'Volume of Nile':
            a[0] = self.y[0]
        # Iterate 
        for t in range(0,len(self.y)-1):
            F[t] = P[t]+sigma_eps2
            # K is defined as ratio of P and F
            Kt = P[t]/F[t] if np.isfinite(self.y[t]) else 0
            v[t] = self.y[t]-a[t]
            a[t+1] = a[t] + np.nan_to_num(Kt*v[t])
            F[t] = P[t]+sigma_eps2
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        F[-1] = P[-1]+sigma_eps2
        v[-1] = self.y[-1]-a[-1]
        # Obtain std error of prediction form variance
        std = np.sqrt((P*sigma_eps2)/(P+sigma_eps2))
        
        if plot:
            fig_name = self.var_name + 'Fig26.pdf'
            plot_fig2_1(self.times, self.y,a, std, P, a, F,fig_name)
        return a, std, P, v, F