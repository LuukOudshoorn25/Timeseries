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

def DK_book_new():
    """Function to run all Kalman Related stuff for the dollar-pound exchange rate"""
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('sv.dat')
    df.columns = ['returns']  # dollar exchange rate
    # Divide the returns to get true returns instead of percentges
    df['returns'] = df['returns'] / 100
    # Convert the returns and demean
    x_t = np.log((df['returns'] - np.mean(df['returns'])) ** 2)
    # Save in the dataframe
    df['transformed_returns'] = x_t

    # Define the system matrices
    # Initial guesses
    sigma_eta = 0.0991
    phi = 0.99
    omega = -0.08
    # Plug these in the system matrices
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
    # Run the optimizer itself
    estimates = KFobj.fit_model(phi=np.array(phi), omega=np.array(omega), sigma_eta=np.array(sigma_eta) )
    # Extract the estimated parameters and update the system matrices
    KFobj.T = np.array(estimates[0])
    KFobj.c = np.array(estimates[1])
    KFobj.R = np.array(estimates[2])

    xi = KFobj.c/(1-KFobj.T)
    # Get from the Kalman Filter object the KF and KFS signals
    filtered_signal = KFobj.iterate(plot=False)[0][1:]
    smoothed_signal = KFobj.state_smooth(plot=False)[0][1:]
    # Plot the smoothed signal
    #plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi)
    # Particle filter
    PF_outputs = KFobj.particle_filter(estimates,df['returns'] - np.mean(df['returns']))
    #plot_pf(PF_outputs)
    #plot_Hts(PF_outputs[0,:],smoothed_signal,filtered_signal,estimates)

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

    # plot_raw_data(df)
    # set parameters
    sigma_eta = 0.202
    phi =  0.984
    omega = -0.066
    beta = -0.1
    alpha_mean = omega / (1 - phi)
    alpha_var = sigma_eta ** 2 / (1 - phi ** 2)
    # Again initialize the system matrices but now add the realized kernel estiamtes to it
    Z = np.vstack((np.ones(len(df)), np.log(df['rk_parzen'].values)))
    R = np.vstack(([sigma_eta], [0]))
    T = np.array([[phi, 0], [0, 1]])
    d = np.array([-1.27])
    c = np.vstack(([omega], [0]))
    H = np.array([np.pi ** 2 / 2])
    Q = np.array([1])
    KFobj = KFnew(df, Z=Z, R=R, d=d, c=c, H=H, Q=Q, T=T, var='transformed_returns', method='iterateRegression')
    # We have instead of only alpha, alpha and beta in a vector
    a_b_start = np.vstack(([alpha_mean], [beta]))
    # Same holds for P
    P_start = np.array([[alpha_var, 0], [0, 0]])
    KFobj.init_filter(a=a_b_start, P=P_start, b=beta, alpha_mean=alpha_mean)

    # Fit the model and estimate all four coefficients
    estimates = KFobj.fit_model(phi=phi, omega=omega, sigma_eta=sigma_eta, beta=beta)
    print(estimates)
    phi = estimates[0]
    omega = estimates[1]
    sigma_eta = estimates[2]
    beta = estimates[3]
    # Update the system matrices
    KFobj.T = np.array([[phi, 0], [0, 1]])
    KFobj.c = np.vstack(([omega], [0]))
    KFobj.R = np.vstack(([sigma_eta], [0]))
    KFobj.a_start = np.vstack(([omega / (1 - phi)], [beta]))
    KFobj.P_start = np.array([[sigma_eta ** 2 / (1 - phi ** 2), 0], [0, 0]])

    # plot for regression type model
    xi = omega / (1 - phi)
    filtered_signal = KFobj.iterateRegression(plot=False)[0][0,:]
    smoothed_signal = KFobj.state_smoothRegression(plot=False)[0][0,:]
    plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi, fname='KFregression_SP500.pdf')

def SP500():
    """Identical to DK_book_new but for SP500 data
       see code comments in that function for explanation etc
    """
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

    # plot_raw_data(df)
    sigma_eta = 0.202
    phi =  0.984
    omega = -0.0659
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
    
    # Create Kalman filter object
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
    plot_smoothed(df=df, filtered_alphas=filtered_signal, smoothed_alphas=smoothed_signal, xi=xi, fname='KF_SP500.pdf')
    df.index = np.arange(0, len(df), 1)
    PF_outputs = KFobj.particle_filter(estimates,df['returns'] - np.mean(df['returns']))
    plot_pf(PF_outputs)
    plot_Hts(PF_outputs[0,:],smoothed_signal,filtered_signal,estimates)


def main():
    """Main function, from here we call the functions for SP500. 
       SP500 with regression (extension 1) and the dollar-pound echange rate data
    """
    # DK_book_new()
    SP500()
    # SP500_regression()

if __name__ == "__main__":
    main()