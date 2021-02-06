import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plottinglib import *
class KFclass():
    def __init__(self,df, init_pars, var='volume'):
        """Initialisation, where df is a pandas DataFrame and var is the name of the column to study and
           init_pars is a dictionary with initial values"""
        self.df = df
        self.var = var
        self.y = np.array(df[var].values.flatten())
        self.times = df.index
        self.pardict = init_pars

    def reset_data(self):
        self.y = self.df[self.var].values.flatten()
    
    def remove_data(self):
        self.y = np.array(self.df[self.var].values.flatten(),dtype=float)
        self.y[20:40] = np.nan
        self.y[60:80] = np.nan
        
       
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
        # Iterate 
        for t in range(0,len(self.y)-1):
            F[t] = P[t]+sigma_eps2
            # K is defined as ratio of P and F
            Kt = P[t]/F[t]
            v[t] = self.y[t]-a[t]
            a[t+1] = a[t] + Kt*v[t]
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        F[-1] = P[-1]+sigma_eps2
        v[-1] = self.y[-1]-a[-1]        
        # Obtain std error of prediction form variance
        std = np.sqrt((P*sigma_eps2)/(P+sigma_eps2))
        
        if plot:
            plot_fig2_1(self.times, self.y,a, std, P, v, F,'Fig21.pdf')
        return a, std, P, v, F

    def state_smooth(self,plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain all time values for L
        L = self.pardict['sigma_eps2']/F
        # Do the recursion for r
        r = np.zeros(len(self.y))
        N = np.zeros(len(self.y))
        V = np.zeros(len(self.y))
        
        for t in np.arange(len(self.y)-1,0,-1):
            r[t-1] = v[t]/F[t]+L[t]*r[t]
        for t in np.arange(len(self.y)-1,0,-1):
            N[t-1] = 1/F[t] + L[t]**2*N[t]
        for t in np.arange(len(self.y)-1,0,-1):
            V[t] = P[t] - P[t]**2*N[t-1]
        
        # Do the recursion for alpha
        alphas = np.zeros(len(self.y))
        alphas[0] = a[t]
        for t in range(1,len(self.y)):
            alphas[t] = a[t] + P[t]*r[t-1]
        alphas[-1]=np.nan
        std = np.sqrt(V)[1:]
        if plot:
            plot_fig2_2(self.times, self.y,alphas, std, V, r, N,'Fig22.pdf')
        return alphas, N

    def disturbance_smoothing(self):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain alpha hats
        alphas, N = self.state_smooth(plot=False)
        # Obtain Observation error
        eps_hat = self.y-alphas
        # Obtain State error
        eta_hat = np.roll(alphas,-1)-alphas
        # Obtain D
        D = 1/F + N*(P/F)**2
        # Obtain State error variance
        var_eta = self.pardict['sigma_eta2'] - self.pardict['sigma_eta2']**2*N
        # Obtain Observation error variance
        var_eps = self.pardict['sigma_eps2'] - (self.pardict['sigma_eps2']**2)*D
        plot_fig2_3(self.times, eps_hat,var_eps,eta_hat, var_eta,'Fig23.pdf')

    def missing_data(self,plot=True):
        """Iterate over the observations and update the filtered values after each iteration"""
        # Set some of the observations to missing 
        self.remove_data()
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
            Kt = P[t]/F[t] if np.isfinite(self.y[t]) else 0
            v[t] = self.y[t]-a[t]
            a[t+1] = a[t] + np.nan_to_num(Kt*v[t])
            F[t] = P[t]+sigma_eps2
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        v[-1] = self.y[-1]-a[-1]
        F[-1] = P[-1]+sigma_eps2
        # Obtain smoothed state
        # Obtain all time values for L
        L = self.pardict['sigma_eps2']/F
        # Do the recursion for r
        r = np.zeros(len(self.y))
        N = np.zeros(len(self.y))
        V = np.zeros(len(self.y))
        
        for t in np.arange(len(self.y)-1,0,-1):
            r[t-1] = v[t]/F[t]+L[t]*r[t]
        for t in np.arange(len(self.y)-1,0,-1):
            N[t-1] = 1/F[t] + L[t]**2*N[t]
        for t in np.arange(len(self.y)-1,0,-1):
            V[t] = P[t] - P[t]**2*N[t-1]
        
        # Do the recursion for alpha
        alphas = np.zeros(len(self.y))
        alphas[0] = a[t]
        for t in range(1,len(self.y)-1):
            alphas[t] = a[t] + np.nan_to_num(P[t]*r[t-1])
        alphas[-1]=np.nan
 
        if plot:
            plot_fig2_5(self.times, self.y,a,P,alphas,V,'Fig25.pdf')
        # Restore data
        self.reset_data()

    def diag_predict(self, plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # obtain standardised forecast errors
        eps = v/np.sqrt(F)
        print(eps, self.times)
        if plot:
            plot_fig2_7(self.times, eps, 'Fig27.pdf')

    def diag_residuals(self, plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain alpha hats
        alphas, N = self.state_smooth(plot=False)
        # Obtain Observation error
        eps_hat = self.y-alphas
        # Obtain State error
        eta_hat = np.roll(alphas,-1)-alphas
        # Obtain D
        D = 1/F + N*(P/F)**2
        # Obtain State error variance
        var_eta = self.pardict['sigma_eta2'] - self.pardict['sigma_eta2']**2*N
        # Obtain Observation error variance
        var_eps = self.pardict['sigma_eps2'] - (self.pardict['sigma_eps2']**2)*D

        obs_res = eps_hat/np.sqrt(var_eps)
        stat_res = eta_hat/np.sqrt(var_eta)

        if plot:
            plot_fig2_8(self.times, obs_res, stat_res, 'Fig28.pdf')