#Assignment 2 code of group 29

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import statsmodels.api as sm
import scipy

def specification(y_t):
    
    z_t = (y_t - np.mean(y_t))/np.std(y_t)
    A_t = 2 * np.exp()
    return [z_t, A_t]

def simulation(N, alpha_t, V_t, theta):

    alpha_tilde = np.zeros(N)
    eta_t = np.zeros(N-1)
    eps_t = np.zeros(N)
    y_tilde = np.zeros(N)

    alpha_tilde[1] = np.random.normal(alpha_t[2], V_t[2]) #initial value
    
    for t in range (1, N-1):
        eta_t[t] = np.random.normal(0, theta[2])
        alpha_tilde[t+1] = theta[0] + theta[1] * alpha_tilde[t] + eta_t [t]
    for t in range (1, N):
        eps_t[t] = np.random.normal(0, 4.93) # Log chi squared(1) variance
        y_tilde[t] = - 1.27 + alpha_tilde[t] + eps_t[t] # Log chi squared(1) mean

    return[y_tilde, alpha_tilde]

def kalmanFilter_smooth(a_t, P_t, v_t, F_t, x_t, theta):
    n = len(x_t)

    phi = theta[1]

    alpha_t = np.zeros(n+1)
    V_t = np.zeros(n+1)
    r_t = np.zeros(n+1)
    N_t = np.zeros(n+1)
    L_t = np.zeros(n+1)

    #Initial values of r_t and N are 0, so no need for specification because we start with a vector filled
    #with zeros
    for t in range(n,1,-1):
        L_t[t] = phi - P_t[t] / F_t[t]
        N_t[t - 1] = (1 / F_t[t]) + L_t[t] ** 2 * N_t[t]
        V_t[t] = P_t[t] - P_t[t] ** 2 * N_t[t - 1]
        r_t[t - 1] = (1 / F_t[t]) * v_t[t] + L_t[t] * r_t[t]
        alpha_t[t] = a_t[t] + P_t[t] * r_t[t - 1]

    return [alpha_t, V_t, r_t, N_t]

def kalmanFilter(x_t, theta):
    N = len(x_t) + 1

    omega = theta[0]
    phi = theta[1]
    sig_eta = theta[2]

    a_1 = 0  # Initial value a
    P_1 = sig_eta / (1 - phi ** 2)
    c = -1.27  # Log chi squared(1) mean
    sig_eps = 4.93  # Log chi squared(1) variance

    # Creating vectors filled with zeros
    v_t = np.zeros(N)
    a_t = np.zeros(N + 1)
    P_t = np.zeros(N + 1)
    k_t = np.zeros(N)
    F_t = np.zeros(N)

    # Initialization at t=1
    a_t[1] = a_1
    P_t[1] = P_1

    for t in range(1, N):
        v_t[t] = x_t.iloc[t - 1] - a_t[t] - c  # fixed mean adjustment
        F_t[t] = P_t[t] + sig_eps


        k_t[t] = (phi * P_t[t]) / F_t[t]
        a_t[t + 1] = omega + phi * a_t[t] + k_t[t] * v_t[t]  # omega is the mean adjustment to be estimated
        print(v_t[t], F_t[t], P_t[t], a_t[t])
        # print(a_t[t])
        P_t[t + 1] = phi ** 2 * P_t[t] + sig_eta - (k_t[t] ** 2) * F_t[t]
        # print(phi ** 2 * P_t[1] + sig_eta - (k_t[1] ** 2) * F_t[1], sig_eta)
    return [a_t, P_t, v_t, F_t]

def estimation(x_t):
    omega_ini = -0.01  # initial value for intercept
    phi_ini = 0.985   # initial value for ar coefficient
    sig_ini = 0.098    # initial value for innovation variance

    theta_ini = np.array([omega_ini,
                       phi_ini,
                       sig_ini
                      ])
    options ={'eps':1e-09,  # argument convergence criteria
              'disp': False,  # display iterations
              'maxiter':200} # maximum number of iterations
    
    results = scipy.optimize.minimize(likelihood, theta_ini, args=(x_t),
                                   options = options,
                                     bounds=( (-10,  10), # Parameter Space Bounds
                                                               (0.00001, 0.99999),
                                                               (0.00001, 10)
                                                             )) #restrictions in parameter space

    return [results.x[0], results.x[1], results.x[2]]

def likelihood(theta, x_t):
    N = len(x_t) + 1

    omega = theta[0]
    phi = theta[1]
    sig_eta = theta[2]

    a_1 = 0                              #Initial value a
    P_1 = sig_eta / (1-phi**2)      #Initial value P
    c = -1.27                             #Log chi squared(1) mean
    sig_eps = 4.93                       # Log chi squared(1) variance

    # Creating vectors filled with zeros
    v_t = np.zeros(N)
    a_t = np.zeros(N + 1)
    P_t = np.zeros(N + 1)
    k_t = np.zeros(N)
    F_t = np.zeros(N)

    #Initialization at t=1
    a_t[1] = a_1
    P_t[1] = P_1
  
    for t in range(1, N):
        v_t[t] = x_t.iloc[t - 1] - a_t[t] - c
        F_t[t] = P_t[t] + sig_eps
        k_t[t] = (phi * P_t[t]) / F_t[t]
        a_t[t + 1] = omega + phi * a_t[t] + k_t[t] * v_t[t]
        P_t[t + 1] = phi ** 2 * P_t[t] + sig_eta - (k_t[t] ** 2) * F_t[t]

    l = (N - 1)/2 - 0.5 * np.sum(np.log(np.abs(F_t[2:]))) - 0.5 * np.sum((v_t[2:]**2) / F_t[2:])

    llik = -np.mean(l)
    return llik

def basicDescriptives(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    skewness = st.skew(data)
    kurtosis = st.kurtosis(data, fisher=False)
    return [mean,median,std,skewness,kurtosis]


############## Plot Functions #################
def plot_d(x_t, y1):
    plt.plot(x_t, 'o', color= 'k', markersize = 0.5)
    plt.plot(y1, color= 'r', linewidth =0.5)
    plt.show()

def plot_b(y1):
    plt.plot(y1, 'o', color= 'k', markersize = 0.5)
    plt.show()

def plot_qq(y1):
    y = y1.values.flatten() 
    y.sort()
    norm = np.random.normal(-0.025,0.025,len(y))
    norm.sort()
    plt.plot(norm,y,"r")
    z = np.polyfit(norm,y, 1)
    p = np.poly1d(z)
    plt.plot(norm,p(norm),"k", linewidth=2)
    plt.show()

def plot_a(y1):
    plt.xlim(-10,960)
    plt.ylim(-0.0375,0.050)
    plt.plot(y1, color= 'k', linewidth =0.5)
    plt.axhline(y=0, color= 'k')
    plt.show()

#def mode(n,theta):
#    sigma = np.exp(0.5 * (theta[0]/1-theta[1]))
#    H_t = np.zeros(n+1)
#    for t in range (n):
#        H_t[t+1] = theta[0] + theta[1] * H_t[t] + sigma *
#    return H_t
##################################################
def main():
    data = pd.read_csv('sv.dat')
    N = len(data)


    ###Question a###
    r_t = data      #y_t: Daily logreturns of pound-dollar exchange rates, mean corrected
    y_t = r_t/100                       #tranformation by dividing by 100
    #plot_a(y_t)
    #
    # print("Mean:",basicDescriptives(y_t)[0])
    # print("Median:",basicDescriptives(y_t)[1])
    # print("Std:",basicDescriptives(y_t)[2])
    # print("Skewness:",basicDescriptives(y_t)[3])
    # print("Kurtosis:",basicDescriptives(y_t)[4])
    
    #for i in range(5):
    #    print(basicDescriptives(y_t)[i]) #mean,median,std,skewness,kurtosis
    #plot_qq(y_t)


    ###Question b###
    x_t = np.log((y_t - np.mean(y_t))**2)
    # plot_b(x_t)
    
    ###Question c###
    # omega = estimation(x_t)[0]
    # print("omega:",omega)
    # phi = estimation(x_t)[1]
    # print("phi:",phi)
    # sig_eta = estimation(x_t)[2]
    # print("sig_eta:",sig_eta)

    omega = -0.015  # estimated value for intercept from book
    phi = 0.995  # estimated value for ar coefficient from book
    sig_eta = 0.8035  # estimated value for innovation variance from book

    ###Question d###    
    theta = np.array([omega,
                          phi,
                          sig_eta
                          ])
    
    #define vectors for AR(1)+noise model
    a_t = kalmanFilter(x_t,theta)[0]  # Filtered state a_t
    # P_t = kalmanFilter(x_t,theta)[1]  # Filtered state variance p_t,
    # v_t = kalmanFilter(x_t,theta)[2]  # Prediction errors v_t
    # F_t = kalmanFilter(x_t,theta)[3]  # Prediction variance F_t
    # alpha_t = kalmanFilter_smooth(a_t, P_t, v_t, F_t, x_t, theta)[0]
    # V_t = kalmanFilter_smooth(a_t, P_t, v_t, F_t, x_t, theta)[1]
    # r_t = kalmanFilter_smooth(a_t, P_t, v_t, F_t, x_t, theta)[2]
    # N_t = kalmanFilter_smooth(a_t, P_t, v_t, F_t, x_t, theta)[3]
    #
    # #simulated y's and alpha's
    # y_plus = simulation(N, alpha_t, V_t, theta)[0]
    # y_plus = pd.DataFrame(y_plus)
    #
    # alpha_plus = simulation(N, alpha_t, V_t, theta)[1]
    #
    # a_t_sim = kalmanFilter(y_plus, theta)[0]  # Filtered state a_t
    # P_t_sim = kalmanFilter(y_plus, theta)[1]  # Filtered state variance p_t,
    # v_t_sim = kalmanFilter(y_plus, theta)[2]  # Prediction errors v_t
    # F_t_sim = kalmanFilter(y_plus, theta)[3]  # Prediction variance F_t
    # alpha_plus_hat = kalmanFilter_smooth(a_t_sim, P_t_sim, v_t_sim, F_t_sim, y_plus, theta)[0]
    #
    # h_t = alpha_t[2:] + alpha_plus[1:] - alpha_plus_hat[2:]
    
    # plot_d(x_t, h_t[2:])
    return
    ###Question e###
    print(mode(N,theta))
    plt.plot(mode(N,theta))
    plt.show()

if __name__ == '__main__':
    main()