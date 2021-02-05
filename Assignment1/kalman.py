import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plottinglib import plot_fig2_1
class KFclass():
    def __init__(self,df, init_pars, var='volume'):
        """Initialisation, where df is a pandas DataFrame and var is the name of the column to study and
           init_pars is a dictionary with initial values"""
        self.y = df[var].values.flatten()
        self.times = df.index
        self.pardict = init_pars
       
    def iterate(self):
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
        # Iterate 
        for t in range(0,len(self.y)-1):
            F[t] = P[t]+sigma_eps2
            # K is defined as ratio of P and F
            Kt = P[t]/F[t]
            v[t] = self.y[t]-a[t]
            a_cond = a[t] + Kt*v[t]
            a[t+1] = a[t] + Kt*v[t]
            F[t] = P[t]+sigma_eps2
            P_cond = P[t]*(1-Kt)
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        # Obtain std error of prediction form variance
        std = np.sqrt((P[t]*sigma_eps2)/(P[t]+sigma_eps2))
        return a, std, P, v, F

    def run(self):
        """Caller function to run Kalman filter and to plot stuff afterwards"""
        # Run KF filter and obtain arrays
        a, std, P, v, F = self.iterate()
        # Now plot the results
        plot_fig2_1(self.times, self.y,a, std, P, v, F)