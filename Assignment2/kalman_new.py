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
                        'maxiter': 200}

    def init_filter(self, a, P, b=None, alpha_mean=None):
        self.a_start = a
        self.b_start = b
        self.P_start = P
        self.alpha_mean = alpha_mean

    def __llik_fun__(self, par_ini):
        # likelihood function of state space model
        n = len(self.y)
        if self.method == 'iterate':
            _, __, P, v, F = self.iterate(plot=False, estimate=True, init_params=par_ini)
        elif self.method == 'iterateRegression':
                _, __, P, v, F = self.iterateRegression(plot=False, estimate=True, init_params=par_ini)
        L= -(n / 2)*np.log(2*np.pi) - 0.5 * np.sum(np.log(np.abs(F[1:]))) - 0.5 * np.sum((v[1:] ** 2) / F[1:]) #+ np.log(self.P_start)
        return (-1)*L

    def fit_model(self, phi, omega, sigma_eta, beta=0):
        # Initialize at the initial values parsed to the class
        par_ini = [phi, omega, sigma_eta, beta]
        # minimize the likelihood function
        Lprime = lambda x: approx_fprime(x, self.__llik_fun__, 0.01)
        if self.method == 'iterateRegression':
            est = minimize(self.__llik_fun__, x0=par_ini,
                           options=self.options,
                           method='Newton-CG', jac=Lprime)
        else:
            est = minimize(self.__llik_fun__, x0=par_ini,
                           options=self.options,
                           method='BFGS')
        return est.x

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
        P[0] = self.P_start #0.8035 / (1 - 0.995 ** 2)
        a[0] = self.a_start
        # Iterate
        for t in range(0, len(self.y) - 1):
            v[t] = self.y[t] - self.Z * a[t] - self.d
            F[t] = self.Z * P[t] * self.Z.transpose() + self.H
            a_t = a[t] + ((P[t] * self.Z.transpose()) / F[t]) * v[t]
            a[t + 1] = self.T * a_t + self.c
            # print(v[t], F[t], P[t], a[t], self.y[t])
            P_t = P[t] - ((P[t] * self.Z.transpose()) / F[t]) * self.Z * P[t]
            P[t + 1] = self.T * P_t * self.T.transpose() + self.R * self.Q * self.R.transpose()
            # print(self.T * P_t * self.T.transpose() + self.R * self.Q * self.R.transpose(), self.R * self.Q * self.R.transpose() )
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
            self.c = np.vstack(([init_params[1],0]))
            self.R = np.vstack(([init_params[2]],[0]))
            self.a_start = np.vstack(([self.alpha_mean], [init_params[3]]))
        P[0,:,:] = self.P_start
        a_b[:,0:1] = self.a_start
        # print(np.shape(a_b[:,0:1]), P[0,:,:], a_b[:,0:1])
        # Iterate
        for t in range(0, len(self.y) - 1):
            v[t] = self.y[t] - np.dot(self.Z[:,t:t+1].T,a_b[:,t]) - self.d
            F[t] = np.dot(np.dot(self.Z[:,t:t+1].T, P[t]),self.Z[:,t:t+1]) + self.H
            a_t = a_b[:,t:t+1] + np.dot(P[t],self.Z[:,t:t+1] / F[t]) * v[t]
            a_b[:,t + 1:t+2] = np.dot(self.T, a_t) + self.c
            P_t = P[t] - np.dot((np.dot(P[t],self.Z[:,t:t+1]) / F[t]),np.dot(self.Z[:,t:t+1].T, P[t]))
            P[t + 1,:,:] = np.dot(np.dot(self.T, P_t),self.T.transpose()) + np.dot(self.R * self.Q,self.R.transpose())
        F[-1] = np.dot(np.dot(self.Z[:,-1:].T, P[-1]),self.Z[:,-1:]) + self.H
        v[-1] = self.y[-1] - a_b[0,-1:]
        # Obtain std error of prediction form variance
        std = np.sqrt((P[:,0,0] * self.H) / (P[:,0,0] + self.H))
        if plot:
            plot_fig2_1(self.times, self.y, a, std, P, v, F)
        return a_b, std, P, v, F

    def state_smoothRegression(self, plot=True):
        a, std, P, v, F = self.iterateRegression(plot=False)
        # Do the recursion for r
        r = np.zeros((2, len(self.y)))
        N = np.zeros((len(self.y), 2, 2))
        V = np.zeros((len(self.y), 2, 2))

        for t in np.arange(len(self.y) - 1, -1, -1):
            K = np.dot(self.T, np.dot(P[t], self.Z[:,t:t+1] / F[t]))
            L = self.T - np.dot(K, self.Z[:,t:t+1].T)
            r[:,t-1:t] = (self.Z[:,t:t+1]/F[t])*v[t] + np.dot(L.T, r[:,t:t+1])
        for t in np.arange(len(self.y) - 1, 0, -1):
            K = np.dot(self.T, np.dot(P[t], self.Z[:,t:t+1] / F[t]))
            L = self.T - np.dot(K, self.Z[:,t:t+1].T)
            N[t-1] = np.dot(self.Z[:,t:t+1]/F[t],self.Z[:,t:t+1].T) + np.dot(L**2, N[t])
        for t in np.arange(len(self.y) - 1, 0, -1):
            V[t] = P[t] - np.dot(P[t] ** 2, N[t - 1])
        V[0] = V[-1]
        N[0] = N[-2]

        # Do the recursion for alpha
        alphas = np.zeros((2, len(self.y)))
        alphas[:,0] = a[:,t]
        for t in range(0, len(self.y)):
            alphas[:,t] = a[:,t] + np.dot(P[t], r[:,t - 1])
        alphas[:,-1] = np.nan
        return alphas, N

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
        return alphas, N

    def particle_filter(self, estimates,y):
        alphas, _ = self.state_smooth(plot=False)
        phi, omega, sigma_eta,_ = estimates
        print(y)
        weights = lambda y,theta: np.exp(-0.5*np.log(2*np.pi) - 0.5*np.exp(theta+xi)-0.5*y**2 / np.exp(theta+xi))
        outputs = np.zeros((3,len(alphas)-2))        
        for i in range(0,len(alphas[:-1])):
            # step 1: draw N values alpha_tilde N(at-1,sigma_eta)
            xi = omega / (1-phi)
            if i ==0:
                # Initial values
                samples = np.random.randn(10000)*(sigma_eta/(1-phi**2))
            else:
                theta = alphas[i-1] - xi
                samples = np.random.randn(10000)*sigma_eta + phi*theta
            # step 2: compute corresponding weights
            w = weights(y[i], samples)
            w = w / np.sum(w)
            ESS = (np.sum(w**2)**(-1))
            signal_hat = np.sum(w*samples)
            P_hat = np.sum(w*samples**2)-signal_hat**2
            resampling = np.mean(np.random.choice(samples, p=w, size=10000))
            outputs[0,i-1] = resampling
            outputs[1,i-1] = P_hat
            outputs[2,i-1] = ESS
        return outputs





