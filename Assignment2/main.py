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

def DK_book():
    # Set matplotlib style for fancy plotting
    plt.style.use('MNRAS_stylesheet')

    # Load the data
    df = pd.read_table('sv.dat')
    df.columns = ['returns'] # dollar exchange rate
    df['logreturns'] = df.returns/100
    df['transformed_returns'] = np.log((df.returns-np.mean(df.returns))**2)

    #print(df.transformed_returns)
    # plot raw data
    #plot_raw_data(df)

    # Perform QML-method
    #QMLplot(df)
    
    init_parameters = {'phi': 0.99,
                       'omega': -0.08,
                       'sigma_eta2':0.083**2}
    KFobj = KFclass(df, init_parameters,'transformed_returns', var_name='Transformed returns')
    print(-KFobj.__llik_fun__([0.9912, -0.0878, 0.0837]))

    KFobj.fit_model()
    #print(KFobj.iterate(plot=False)[0])
    #plt.plot(KFobj.iterate(plot=False)[4])
    #plt.show()
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


def main():
    DK_book()

if __name__ == "__main__":
    main()