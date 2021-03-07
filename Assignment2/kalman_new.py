from scipy.optimize import minimize, approx_fprime
from plottinglib import *
from scipy.optimize import minimize, approx_fprime

from plottinglib import *


class KFnew():
    def __init__(self, df, Z, Q, H, T, R, d, c, var='dep_var', secondvar=None, method='iterate'):
        """Initialisation, where df is a pandas DataFrame and var is the name of the column to study and
           init_pars is a dictionary with initial values"""
        self.method = method
        self.Z = Z
        self.Q = Q
        self.H = H
        self.T = T
        self.R = R
        self.d = d
        self.c = c

        self.df = df
        self.var = var
        self.y = np.array(df[var].values.flatten())
        self.times = df.index
        self.options = {'eps': 1e-09,
                        'maxiter': 2000}

    def init_filter(self, a, P, b=None):
        self.a_start = a
        self.b_start = b
        self.P_start = P

    def __llik_fun__(self, par_ini):
        # likelihood function of state space model
        n = len(self.y)
        if self.method == 'iterate':
            _, __, P, v, F = self.iterate(plot=False, estimate=True, init_params=par_ini)
        elif self.method == 'iterateRegression':
                _, __, P, v, F = self.iterateRegression(plot=False, estimate=True, init_params=par_ini)
        L= (n - 1) / 2 - 0.5 * np.sum(np.log(F[1:])) - 0.5 * np.sum((v[1:] ** 2) / F[1:]) #+ np.log(self.P_start)
        return (-1)*L

    def fit_model(self):
        # Initialize at the initial values parsed to the class
        phi = self.T
        omega = self.c
        sigma_eta2 = self.R
        # sigma_eta2 = self.Q
        # sigma_eps2 = self.H
        par_ini = [phi, omega, sigma_eta2]
        # par_ini = [sigma_eps2, sigma_eta2]
        # minimize the likelihood function
        Lprime = lambda x: approx_fprime(x, self.__llik_fun__, 0.01)
        est = minimize(self.__llik_fun__, x0=par_ini,
                       options=self.options,
                       method='Newton-CG', bounds=((-1, 1), (-100, 100), (0, 10000)), jac=Lprime)
        self.T = est.x[0]
        self.c = est.x[1]
        self.R = est.x[2]
        # self.Q = est.x[1]
        # self.H = est.x[0]
        print('omega', est.x[1])
        print('phi', est.x[0])
        print('sigma', est.x[2])

    def iterate(self, plot=True, estimate=False, init_params=None):
        """Iterate over the observations and update the filtered values after each iteration"""
        # Create empty arrays to store values
        F = np.zeros(len(self.y))
        a = np.zeros(len(self.y))
        v = np.zeros(len(self.y))
        P = np.zeros(len(self.y))
        # Initialize at the initial values parsed to the class
        if estimate == True:
            self.T = np.array(init_params[0])
            self.c = np.array(init_params[1])
            self.R = np.array(init_params[2])
            # self.H = np.array(init_params[0])
            # self.Q = np.array(init_params[1])
        P[0] = self.P_start
        a[0] = self.a_start
        # Iterate
        for t in range(0, len(self.y) - 1):
            v[t] = self.y[t] - self.Z * a[t] - self.d
            F[t] = self.Z * P[t] * self.Z.transpose() + self.H
            a_t = a[t] + ((P[t] * self.Z.transpose()) / F[t]) * v[t]
            a[t + 1] = self.T * a_t + self.c
            P_t = P[t] - ((P[t] * self.Z.transpose()) / F[t]) * self.Z * P[t]
            P[t + 1] = self.T * P_t * self.T.transpose() + self.R * self.Q * self.R.transpose()
        F[-1] = P[-1] + self.H
        v[-1] = self.y[-1] - a[-1]
        # Obtain std error of prediction form variance
        std = np.sqrt((P * self.H) / (P + self.H))
        if plot:
            plot_fig2_1(self.times, self.y, a, std, P, v, F)
        return a, std, P, v, F

    def iterateRegression(self, plot=True, estimate=False, init_params=None):
        """Iterate over the observations and update the filtered values after each iteration"""
        # Create empty arrays to store values
        F = np.zeros(len(self.y))
        a_b = np.zeros((2,len(self.y))) # alphas and betas
        v = np.zeros(len(self.y))
        P = np.zeros((len(self.y),2,2))
        # Initialize at the initial values parsed to the class
        if estimate == True:
            self.T = np.array([[init_params[0],0],[0,1]])
            self.c = np.array([init_params[1],0])
            self.R = np.array([[init_params[2]],[0]])
            # self.H = np.array(init_params[0])
            # self.Q = np.array(init_params[1])
        P[0,:,:] = self.P_start
        a_b[:,0] = [self.a_start, self.b_start] 
        # Iterate
        for t in range(0, len(self.y) - 1):
            v[t] = self.y[t] - np.dot(self.Z[:,t],a_b[:,t]) - self.d
            F[t] = np.dot(np.dot(self.Z[:,t], P[t]),self.Z[:,t].T) + self.H
            a_t = a_b[:,t] + np.dot(P[t],self.Z[:,t].T) / F[t] * v[t]
            a_t = a_t.reshape(1,-1).T
            a_b[:,t + 1] = np.dot(self.T, a_t).flatten() + self.c
            P_t = P[t] - np.dot((np.dot(P[t],self.Z[:,t].transpose()) / F[t]),np.dot(self.Z[:,t], P[t]))
            P[t + 1,:,:] = np.dot(self.T * P_t,self.T.transpose()) + np.dot(self.R * self.Q,self.R.transpose())
        F[-1] = np.dot(np.dot(self.Z[:,-1], P[-1]),self.Z[:,t].T) + self.H
        v[-1] = self.y[-1] - a_b[0,-1]
        # Obtain std error of prediction form variance
        std = np.sqrt((P * self.H) / (P + self.H))
        if plot:
            plot_fig2_1(self.times, self.y, a, std, P, v, F)
        return a_b, std, P, v, F

    def state_smooth(self, plot=True):
        a, std, P, v, F = self.iterate(plot=False)
        # Obtain all time values for L
        L = self.T - P / F

        # Do the recursion for r
        r = np.zeros(len(self.y))
        N = np.zeros(len(self.y))
        V = np.zeros(len(self.y))

        for t in np.arange(len(self.y) - 1, -1, -1):
            r[t-1] = (self.Z.transpose()/F[t])*v[t] + L[t] * r[t]
        for t in np.arange(len(self.y) - 1, 0, -1):
            N[t-1] = (self.Z.transpose()/F[t])*self.Z + L[t]**2 * N[t]
        for t in np.arange(len(self.y) - 1, 0, -1):
            V[t] = P[t] - P[t] ** 2 * N[t - 1]
        V[0] = V[-1]
        N[0] = N[-2]

        # Do the recursion for alpha
        alphas = np.zeros(len(self.y))
        alphas[0] = a[t]
        for t in range(0, len(self.y)):
            alphas[t] = a[t] + P[t] * r[t - 1]
        alphas[-1] = np.nan
        std = np.sqrt(V)[1:]
        # print(self.y[0])
        if plot:
            plot_fig2_2(self.times, self.y, alphas, std, V, r, N)
        return alphas, N