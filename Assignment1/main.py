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

def main():
    # Import required libs
    
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('Nile.dat')
    df.columns = ['volume']
    df['year'] = np.arange(1871,1971,1)
    df = df.set_index('year')
    df = pd.DataFrame(df)

    # Let us look at the original data
    #plot_raw_data(df)

    # Run Kalman filter and reproduce fig 2.1
    parameters = {'P1':1e7,
                'sigma_eps2':15099,
                'sigma_eta2':1469.1}
    # Create Kalman filter object
    KFobj = KFclass(df, init_pars=parameters, var='volume')
    # Plot basic Kalman filtering
    #KFobj.iterate()
    # Plot state smoothing 
    #KFobj.state_smooth()
    # Plot disturbance smoothing
    #KFobj.disturbance_smoothing()
    # Now with missing values
    KFobj.missing_data()



if __name__ == "__main__":
    main()