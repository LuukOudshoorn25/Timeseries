####################################
###                              ###
###     Main python file         ###
###                              ###
####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our own libs with functions
from plottinglib import *
from kalman import KFclass
from kalman_prediction import *

def main():
    # Load Data
    df = pd.read_excel('temperatures_Bilt.xlsx', header=0)
    df = df.rename(columns={' TG_hom': 'dep_var', 'YYYYMMDD': 'times'})
    df.index = pd.to_datetime(df.times)
    # Aggregate to 1 year and drop 2021
    df = df.resample('1Y').mean()
    df = df[:-1]
    # Define years as index
    df['year'] = np.arange(1901,1901+len(df.index),1)
    df = df.set_index('year')

    # Let us look at the original data
    # plot_raw_data(df, var_name='Temperature Bilt')

    # Run Kalman filter and reproduce fig 2.1
    init_parameters = {'P1': 31.07320341,
                  'sigma_eps2': 85.2698937,
                  'sigma_eta2': 195.03523569}
    # Create Kalman filter object
    KFobj = KFclass(df, init_pars=init_parameters, var='dep_var', var_name='Temperature Bilt')
    KFobj.fit_model()
    # Plot basic Kalman filtering (fig1)
    KFobj.iterate()
    # Plot state smoothing  (fig2)
    # KFobj.state_smooth()


def nile_data():
    # Import required libs
    
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('Nile.dat')
    df.columns = ['volume']
    df['year'] = np.arange(1871,1971,1)
    df = df.set_index('year')
    df = pd.DataFrame(df)
    df = df.rename(columns={'volume':'dep_var'})

    # Let us look at the original data
    #plot_raw_data(df)

    # Run Kalman filter and reproduce fig 2.1
    parameters = {'P1':1e7,
                'sigma_eps2':15099,
                'sigma_eta2':1469.1}
    # Create Kalman filter object
    KFobj = KFclass(df, init_pars=parameters, var='dep_var')
    # Plot basic Kalman filtering (fig1)
    KFobj.iterate()
    # Plot state smoothing  (fig2)
    KFobj.state_smooth()
    # Plot disturbance smoothing (fig3)
    KFobj.disturbance_smoothing()
    # Now with missing values (fig4)
    KFobj.missing_data()
    # Now predictions using Kalman filter
    # Extend df with missing observations
    df_ext = pd.DataFrame({'year':np.arange(1971,2001), 'dep_var':np.ones(30)*np.nan}).set_index('year')
    df_extended = pd.concat((df, df_ext))
    # fig 5
    KFpred = KFpredictor(df_extended, init_pars=parameters, var='dep_var')
    KFpred.iterate()
    # Fig 6
    KFobj = KFclass(df, init_pars=parameters, var='dep_var')
    KFobj.diag_predict()
    # Fig 7
    KFobj.diag_residuals()


if __name__ == "__main__":
    # nile_data()
    main()