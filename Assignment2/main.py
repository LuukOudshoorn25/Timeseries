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


    print(df.transformed_returns)
    # plot raw data
    # plot_raw_data(df)


def main():
    DK_book()

if __name__ == "__main__":
    main()