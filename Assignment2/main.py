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
    print('The sample moments of X_t are: ', 'mean = ', np.round(np.mean(df['transformed_returns']),2))
    print('variance = ', np.round(np.var(df['transformed_returns']),2))
    print('skewness = ', np.round(skew(df['transformed_returns']), 2))
    print('kurtosis = ', np.round(kurtosis(df['transformed_returns']), 2))
    # df.x = df.x-df.x.mean()
    # df['y'] = np.log(1+df['x']/100)
    

    # plt.figure(figsize=(8,3))
    # plt.plot(df.index, df.y, lw=0.3)
    # plt.ylim(-0.035, 0.05)
    # plt.show()

    #    df['transformed_returns'] = np.log(df['y']**2)
    
    #-np.mean(df.returns)
    #print(df.transformed_returns)
    # plot raw data
    #plot_raw_data(df)

    # Perform QML-method
    #QMLplot(df)
    
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
    print(df['transformed_returns'])
    print('The sample moments of X_t are: ', 'mean = ', np.round(np.mean(df['transformed_returns']), 2))
    print('variance = ', np.round(np.var(df['transformed_returns']), 2))
    print('skewness = ', np.round(skew(df['transformed_returns']), 2))
    print('kurtosis = ', np.round(kurtosis(df['transformed_returns']), 2))
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
    print(KFobj.__llik_fun__(par_ini=[estimates[0], estimates[1], estimates[2]]))
    print(estimates)
    KFobj.T = np.array(estimates[0])
    KFobj.c = np.array(estimates[1])
    KFobj.R = np.array(estimates[2])
    # #
    # # # KFobj.state_smooth(plot=True)
    plt.plot(KFobj.state_smooth(plot=False)[0][1:],color='black',lw=1)
    plt.scatter(df.index, df['transformed_returns'], s=1)
    plt.show()



def SP500():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv', index_col=0)
    df = df[df['Symbol'] == '.SPX']
    df['returns'] = (df['close_price'] - df['close_price'].shift(1))/100
    df['logreturns'] = np.log(df['close_price']) - np.log(df['close_price'].shift(1))
    df = df.iloc[1:]
    df['transformed_returns'] = np.log((df['logreturns'] - np.mean(df['logreturns'])) ** 2)
    df.index = np.arange(0, len(df), 1)

    sigma_eta = 0.0991
    phi = 0.99
    omega = -0.08
    alpha_mean = omega / (1 - phi)
    alpha_var = sigma_eta ** 2 / (1 - phi ** 2)
    # Define the system matrices
    # Z = np.array(1)
    # R = np.array(np.sqrt(0.83**2))  # to be estimated
    # T = np.array(0.99)  # to be estimated
    # d = np.array(-1.27)
    # c = np.array(-0.08) # to be estimated
    # H = np.array(np.pi**2/2)
    # Q = np.array(1)
    #
    # # Create Kalman filter object
    # KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns', method='iterate')
    # KFobj.init_filter(a=-10.89, P=R/(1-T**2))
    # estimates = KFobj.fit_model(phi=np.array(0.99), omega=np.array(-0.08), sigma_eta=np.array(np.sqrt(0.083**2)) )
    # print(estimates)
    # KFobj.T = np.array(estimates[0])
    # KFobj.c = np.array(estimates[1])
    # KFobj.R = np.array(estimates[2])
    # # KFobj.iterate()
    # plt.plot(KFobj.state_smooth(plot=False)[0],color='black',lw=1)
    # plt.scatter(df.index, df['transformed_returns'], s=1)
    # plt.show()

    Z = np.vstack((np.ones(len(df)), np.log(df['rk_parzen'].values)))
    R = np.vstack(([sigma_eta], [0]))
    T = np.array([[phi, 0], [0, 1]])
    d = np.array([-1.27])
    c = np.vstack(([omega], [0]))
    H = np.array([np.pi**2/2])
    Q = np.array([1])
    KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns', method='iterateRegression')

    # initialize beta
    beta = 0.05
    a_b_start = np.vstack(([alpha_mean], [beta]))
    P_start = np.array([[alpha_var, 0], [0, 0]])
    KFobj.init_filter(a=a_b_start, P=P_start, b=beta, alpha_mean=alpha_mean)

    # fit the model
    # estimates = KFobj.fit_model(phi=phi, omega=omega, sigma_eta=sigma_eta, beta=beta)
    # print(estimates)
    # phi = estimates[0]
    # omega = estimates[1]
    # sigma_eta = estimates[2]
    # beta = estimates[3]
    # KFobj.T = np.array([[phi, 0], [0, 1]])
    # KFobj.c = np.vstack(([omega], [0]))
    # KFobj.R = np.vstack(([sigma_eta], [0]))
    # KFobj.a_start = np.vstack(([omega / (1 - phi)], [beta]))
    # KFobj.P_start = np.array([[sigma_eta ** 2 / (1 - phi ** 2), 0], [0, 0]])

    # run iteration
    a, std, P, v, F= KFobj.iterateRegression(plot=False)
    plt.plot(df.index[1:], a[0,1:], color='black', lw=1)
    plt.scatter(df.index, df['transformed_returns'], s=1)
    plt.scatter(df.index, np.log(df['rk_parzen'].values), s=1)
    plt.show()

def main():
    # DK_book_new()
    SP500()

if __name__ == "__main__":
    main()