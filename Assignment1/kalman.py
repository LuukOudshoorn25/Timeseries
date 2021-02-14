import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime
from plottinglib import *
class KFclass():
    def __init__(self,df, init_pars, var='dep_var', var_name='Volume of Nile'):
        """Initialisation, where df is a pandas DataFrame and var is the name of the column to study and
           init_pars is a dictionary with initial values"""
        self.df = df
        self.var = var
        self.var_name = var_name
        self.y = np.array(df[var].values.flatten())
        self.times = df.index
        self.pardict = init_pars
        self.options = {'eps':1e-09,
                      'maxiter':2000}

    def __llik_fun__(self, par_ini):
        # likelihood function of state space model
        n = len(self.y)
        _, __, ___, v, F = self.iterate(plot=False, estimate=True, init_params=par_ini)
        L = -(n/2)*np.log(2*np.pi) - 0.5*(np.log(F) + (v**2/F))
        llik = np.mean(L)
        return -1*llik

    def fit_model(self):
        # Initialize at the initial values parsed to the class
        sigma_eps2 = self.pardict['sigma_eps2']
        sigma_eta2 = self.pardict['sigma_eta2']

        par_ini = [sigma_eps2, sigma_eta2]
        # minimize the likelihood function
        Lprime = lambda x: approx_fprime(x, self.__llik_fun__, 0.01)
        est = minimize(self.__llik_fun__, x0=par_ini,
                       options = self.options,
                       method='SLSQP', bounds=((0,100),(0,100)), jac=Lprime)
        self.pardict['P1'] = est.x[0]
        self.pardict['sigma_eps2'] = est.x[0]
        self.pardict['sigma_eta2'] = est.x[1]
        print(est.x)

    def reset_data(self):
        self.y = self.df[self.var].values.flatten()
    
    def remove_data(self):
        self.y = np.array(self.df[self.var].values.flatten(),dtype=float)
        self.y[20:40] = np.nan
        self.y[60:80] = np.nan

    def iterate(self,plot=True, estimate=False, init_params=None):
        """Iterate over the observations and update the filtered values after each iteration"""
        # Create empty arrays to store values
        F = np.zeros(len(self.y))
        a = np.zeros(len(self.y))
        v = np.zeros(len(self.y))
        P = np.zeros(len(self.y))
        # Initialize at the initial values parsed to the class
        if estimate == True:
            sigma_eps2 = init_params[0]
            sigma_eta2 = init_params[1]
            P[0] = sigma_eps2
        else:
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
            Kt = P[t]/F[t]
            v[t] = self.y[t]-a[t]
            a[t+1] = a[t] + Kt*v[t]
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        F[-1] = P[-1]+sigma_eps2
        v[-1] = self.y[-1]-a[-1]

        # Obtain std error of prediction form variance
        std = np.sqrt((P*sigma_eps2)/(P+sigma_eps2))

        if plot:
            fig_name = self.var_name + 'Fig21.pdf'
            plot_fig2_1(self.times, self.y,a, std, P, v, F, fname=fig_name, var_name=self.var_name)
        return a, std, P, v, F

    def state_smooth(self,plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain all time values for L
        L = self.pardict['sigma_eps2']/F
        # Do the recursion for r
        r = np.zeros(len(self.y))
        N = np.zeros(len(self.y))
        V = np.zeros(len(self.y))

        for t in np.arange(len(self.y)-1,-1,-1):
            r[t-1] = v[t]/F[t]+L[t]*r[t]
        for t in np.arange(len(self.y)-1,0,-1):
            N[t-1] = 1/F[t] + L[t]**2*N[t]
        for t in np.arange(len(self.y)-1,0,-1):
            V[t] = P[t] - P[t]**2*N[t-1]
        print(V[0])
        # Do the recursion for alpha
        alphas = np.zeros(len(self.y))
        alphas[0] = a[t]
        for t in range(0,len(self.y)):
            alphas[t] = a[t] + P[t]*r[t-1]
        alphas[-1]=np.nan
        std = np.sqrt(V)[1:]
        if plot:
            fig_name = self.var_name + 'Fig22.pdf'
            plot_fig2_2(self.times, self.y,alphas, std, V, r, N, fname=fig_name, var_name=self.var_name)
        return alphas, N

    def disturbance_smoothing(self):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain alpha hats
        alphas, N = self.state_smooth(plot=False)
        # Obtain Observation error
        eps_hat = self.y-alphas
        # Obtain State error
        # Do the recursion for r
        r = np.zeros(len(self.y))
        # Obtain all time values for L
        L = self.pardict['sigma_eps2']/F
        for t in np.arange(len(self.y) - 1, 0, -1):
            r[t - 1] = v[t] / F[t] + L[t] * r[t]

        eta_hat = self.pardict['sigma_eta2']*r
        # Obtain D
        D = 1/F + N*(P/F)**2
        # Obtain State error variance
        var_eta = self.pardict['sigma_eta2'] - self.pardict['sigma_eta2']**2*N
        # Obtain Observation error variance
        var_eps = self.pardict['sigma_eps2'] - (self.pardict['sigma_eps2']**2)*D

        fig_name = self.var_name + 'Fig23.pdf'
        plot_fig2_3(self.times, eps_hat,var_eps,eta_hat, var_eta, fname=fig_name)

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
        # initialise a for estimated model
        if self.var_name != 'Volume of Nile':
            a[0] = self.y[0]
        # Iterate 
        for t in range(0,len(self.y)-1):
            v[t] = np.nan_to_num(self.y[t]-a[t])
            F[t] = P[t]+sigma_eps2 if np.isfinite(self.y[t]) else np.inf
            # K is defined as ratio of P and F
            Kt = P[t]/F[t] if np.isfinite(self.y[t]) else 0
            a[t+1] = a[t] + Kt*v[t]
            P[t+1] = P[t]*(1-Kt)+sigma_eta2
        v[-1] = self.y[-1]-a[-1]
        F[-1] = P[-1]+sigma_eps2
        # Obtain smoothed state
        # Obtain all time values for L
        L = self.pardict['sigma_eps2'] / F
        index_missing = np.argwhere(np.isnan(self.y))
        L[index_missing] = 1

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
        print(F[index_missing], F)
        # Do the recursion for alpha
        alphas = np.zeros(len(self.y))
        alphas[0] = a[t]
        for t in range(1,len(self.y)-1):
            alphas[t] = a[t] + P[t]*r[t-1]
        alphas[-1]=np.nan
 
        if plot:
            fig_name = self.var_name + 'Fig25.pdf'
            plot_fig2_5(self.times, self.y,a,P,alphas,V,fname=fig_name, var_name=self.var_name)
        # Restore data
        self.reset_data()

    def diag_predict(self, plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # obtain standardised forecast errors
        eps = v/np.sqrt(F)
        if plot:
            fig_name = self.var_name + 'Fig27.pdf'
            plot_fig2_7(self.times, eps, fname=fig_name)

    def diag_residuals(self, plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain alpha hats
        alphas, N = self.state_smooth(plot=False)
        # Obtain Observation error

        # Obtain D
        D = 1/F + N*(P/F)**2
        # Obtain State error variance

        # Obtain all time values for L
        L = self.pardict['sigma_eps2'] / F
        # Do the recursion for r
        r = np.zeros(len(self.y))
        for t in np.arange(len(self.y)-1,0,-1):
            r[t-1] = v[t]/F[t]+L[t]*r[t]

        K = P[t] / F[t]
        u = v/F - K*r
        # obtain standardised smoothed residuals
        obs_res = u/np.sqrt(D)
        stat_res = r/np.sqrt(N)

        if plot:
            fig_name = self.var_name + 'Fig28.pdf'
            plot_fig2_8(self.times, obs_res, stat_res, fname=fig_name)