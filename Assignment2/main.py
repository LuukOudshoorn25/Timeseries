####################################
###                              ###
###     Main python file         ###
###                              ###
####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Import our own libs with functions
from plottinglib import *
from kalman import KFclass
from kalman_new import KFnew
from kalman_prediction import *

def DK_book():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('sv.dat')
    df.columns = ['returns'] # dollar exchange rate
    df['returns'] = df['returns'] / 100
    
    x_t = np.log((df['returns']-np.mean(df['returns']))**2)
    df['transformed_returns'] = x_t
    
    init_parameters = {'phi': 0.99,
                       'omega': -0.08,
                       'sigma_eta2':0.083**2}
    KFobj = KFclass(df, init_parameters,'transformed_returns', var_name='Transformed returns')
    print(-KFobj.__llik_fun__([0.9912, -0.0878, 0.0837**2]))

    KFobj.pardict = {'phi': 0.99,
                     'omega': -0.08,
                     'sigma_eta2':0.083**2}

    KFobj.fit_model()
    # print(KFobj.iterate(plot=False)[0])
    plt.plot(KFobj.state_smooth(plot=False)[0],color='black',lw=1)
    plt.scatter(df.index, df['transformed_returns'], s=1)
    plt.show()
    #print(alphas)
    
"""
KFobj.fit_model()
# Plot basic Kalman filtering (fig1)
KFobj.iterate()
# Plot state smoothing  (fig2)
KFobj.state_smooth()
# Plot disturbance smoothing (fig3)
KFobj.disturbance_smoothing()
# Now with missing values (fig4)
KFobj.missing_data()
"""

def DK_book_new():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('sv.dat')
    df.columns = ['returns']  # dollar exchange rate
    df['returns'] = df['returns'] / 100

    x_t = np.log((df['returns'] - np.mean(df['returns'])) ** 2)
    df['transformed_returns'] = x_t

    # Define the system matrices
    sigma_eta = 0.0991
    phi = 0.99
    omega = -0.08
    Z = np.array(1)
    R = np.array(sigma_eta)  # to be estimated
    T = np.array(phi)  # to be estimated
    d = np.array(-1.27)
    c = np.array(omega) # to be estimated
    H = np.array(np.pi**2/2)
    Q = np.array(1)

    # Run Kalman filter and reproduce fig 2.1
    # Create Kalman filter object
    KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns')
    KFobj.init_filter(a=(omega/(1-phi)), P= sigma_eta**2 / (1 - phi ** 2))
    estimates = KFobj.fit_model(phi=np.array(phi), omega=np.array(omega), sigma_eta=np.array(sigma_eta) )
    
    KFobj.T = np.array(estimates[0])
    KFobj.c = np.array(estimates[1])
    KFobj.R = np.array(estimates[2])
    #
    xi = KFobj.c/(1-KFobj.T)
    filtered_signal = KFobj.iterate(plot=False)[0][1:]
    smoothed_signal = KFobj.state_smooth(plot=False)[0][1:]
    #plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi)
    # Particle filter
    print(estimates)
    PF_outputs = KFobj.particle_filter(estimates,df['returns'] - np.mean(df['returns']))
    #plot_pf(PF_outputs)
    plot_Hts(PF_outputs[0,:],smoothed_signal,filtered_signal,estimates)
    
    

def SP500_regression():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')
    # Load the data
    df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv', index_col=0)
    df = df[df['Symbol'] == '.SPX']
    df['returns'] = (df['close_price'] - df['close_price'].shift(1))/100
    df['logreturns'] = np.log(df['close_price']) - np.log(df['close_price'].shift(1))
    df = df.iloc[1:]
    df['transformed_returns'] = np.log((df['returns'] - np.mean(df['returns'])) ** 2)
    df.index = np.linspace(2000, 2021, len(df))

    # set parameters
    sigma_eta = 0.20175882
    phi =  0.98399089
    omega = -0.06589895
    beta = -0.1
    alpha_mean = omega / (1 - phi)
    alpha_var = sigma_eta ** 2 / (1 - phi ** 2)

    Z = np.vstack((np.ones(len(df)), np.log(df['rk_parzen'].values)))
    R = np.vstack(([sigma_eta], [0]))
    T = np.array([[phi, 0], [0, 1]])
    d = np.array([-1.27])
    c = np.vstack(([omega], [0]))
    H = np.array([np.pi ** 2 / 2])
    Q = np.array([1])
    KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns', method='iterateRegression')

    a_b_start = np.vstack(([alpha_mean], [beta]))
    P_start = np.array([[alpha_var, 0], [0, 0]])
    KFobj.init_filter(a=a_b_start, P=P_start, b=beta, alpha_mean=alpha_mean)

    # # fit the model
    # estimates = KFobj.fit_model(phi=phi, omega=omega, sigma_eta=sigma_eta, beta=beta)
    # print(estimates)
    # phi = estimates[0]
    # omega = estimates[1]
    # sigma_eta = estimates[2]
    # beta = estimates[3]
    KFobj.T = np.array([[phi, 0], [0, 1]])
    KFobj.c = np.vstack(([omega], [0]))
    KFobj.R = np.vstack(([sigma_eta], [0]))
    KFobj.a_start = np.vstack(([omega / (1 - phi)], [beta]))
    KFobj.P_start = np.array([[sigma_eta ** 2 / (1 - phi ** 2), 0], [0, 0]])

    # plot for regression type model
    xi = omega / (1 - phi)
    # filtered_signal = KFobj.iterateRegression(plot=False)[0][0,:]
    smoothed_signal = KFobj.state_smoothRegression(plot=False)[0][0, :]
    plt.scatter(df.index, df['transformed_returns'], s=1)
    plt.plot(df.index, smoothed_signal, color='red')
    plt.show()
    # smoothed_signal = KFobj.state_smooth(plot=False)[0][1:]
    # plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi, fname='KF_SP500.pdf')

def SP500():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv', index_col=0)
    df = df[df['Symbol'] == '.SPX']
    df['returns'] = (df['close_price'] - df['close_price'].shift(1))/100
    df['logreturns'] = np.log(df['close_price']) - np.log(df['close_price'].shift(1))
    df = df.iloc[1:]
    df['transformed_returns'] = np.log((df['returns'] - np.mean(df['returns'])) ** 2)
    # df.index = np.arange(0, len(df), 1)
    df.index = np.linspace(2000, 2021, len(df))

    # plot_raw_data(df)
    sigma_eta = 0.20175882
    phi =  0.98399089
    omega = -0.06589895
    alpha_mean = omega / (1 - phi)
    alpha_var = sigma_eta ** 2 / (1 - phi ** 2)
    # Define the system matrices
    Z = np.array(1)
    R = np.array(sigma_eta)  # to be estimated
    T = np.array(phi)  # to be estimated
    d = np.array(-1.27)
    c = np.array(omega) # to be estimated
    H = np.array(np.pi**2/2)
    Q = np.array(1)
    #
    # # Create Kalman filter object
    KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns', method='iterate')
    KFobj.init_filter(a=alpha_mean, P=alpha_var)
    estimates = KFobj.fit_model(phi=np.array(0.99), omega=np.array(-0.08), sigma_eta=np.array(np.sqrt(0.083**2)) )
    print(estimates)
    phi = estimates[0]
    omega = estimates[1]
    sigma_eta = estimates[2]
    alpha_mean = omega / (1 - phi)
    alpha_var = sigma_eta ** 2 / (1 - phi ** 2)
    KFobj.init_filter(a=alpha_mean, P=alpha_var)
    KFobj.T = np.array(estimates[0])
    KFobj.c = np.array(estimates[1])
    KFobj.R = np.array(estimates[2])
    # plot filtered and smoothed estimates
    xi = KFobj.c/(1-KFobj.T)
    filtered_signal = KFobj.iterate(plot=False)[0][1:]
    smoothed_signal = KFobj.state_smooth(plot=False)[0][1:]
    #plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi, fname='KF_SP500.pdf')

def main():
    DK_book_new()
    #SP500()
    # SP500_regression()

if __name__ == "__main__":
    main()