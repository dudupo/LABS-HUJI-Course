
import numpy as np 
from matplotlib import pyplot as plt 

import pandas


from scipy.optimize import curve_fit

def extract_coef( time, distance ):
    def f(x, a, b,c, d ):
        return a*np.cos(b*x+c)**2 + d  
    popt, pcov = curve_fit(f, time, distance)
    _range = np.linspace( np.min(time) , np.max(time), 100 )
    return popt, (_range, f(_range, *popt)) 

""" cm """
LASER_POSITION = 89 
RESONATOR_POSITION = 7
MIDDELE_POSITOPN = 52
CONSTAT_POL_POSITION = 43.5 
CONSTAT_POL_ANGLE = 223


_lightexp = "./csv/STRIKE/test-1.xlsx" 
_darkexp = "./csv/STRIKE-DARK/test-1.xlsx"
_lightoffset = "./csv/OFFSET/test-3.xlsx"


ANGELS = [ 218, 200, 180, 160, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10 , 0, 350]
""" Light [\mu A]  """
AMP = [ 1.79, 1.605,  1.651, 1.618, 2.21, 3.210, 4.51, 5.66, 6.51, 6.67, 6.33, 5.08, 3.77, 2.65, 1.97,  1.626, 1.55, 1.603, 1.602, 1.57 ]



""" second stage"""
ANGELS2 = [ 0 , 10 ,20 ,30, 40 ,50 ,60, 70, 80, 90, 100, 110, 120, 130 ,140, 150, 160, 170, 180, 190, 200, 210 ]
AMP2 = [ 1.73, 2.03, 2.31, 2.51, 2.50, 2.32, 2.07, 1.74, 1.54, 1.53, 1.65, 1.78, 1.92, 1.944, 1.81, 1.63, 1.52, 1.559, 1.737, 2.03, 2.32, 2.52]


def covert_data_frame( df ):
    return df[0][5:].values.astype(float) 



def plot_phase_diff(angels,  amp):
    x, y = np.deg2rad((np.array(angels) - CONSTAT_POL_ANGLE) % 360), np.array(amp)
    coeff, cosfunc = extract_coef(x, y)
    plt.plot(x,y)
    plt.plot(cosfunc[0], cosfunc[1])
    plt.show()

def main():
    print( len(ANGELS2) == len(AMP2))
    # return 
    light_strike = covert_data_frame(pandas.read_excel(_lightexp)) 
    dark_strike = covert_data_frame(pandas.read_excel(_darkexp))
    lightoffset = covert_data_frame(pandas.read_excel(_lightoffset))
    
    print( light_strike)
    # plt.plot( light_strike- np.average( lightoffset ) )
    # plt.show()

    print( np.linalg.norm( dark_strike - (light_strike- np.average( lightoffset ) ) ))
    # plt.savefig( "" )

    print(np.average( lightoffset ))
    plot_phase_diff(ANGELS, np.array(AMP) - np.mean( lightoffset ))
    plot_phase_diff(ANGELS2, np.array(AMP2) - np.mean( lightoffset ))
    

if __name__ == "__main__" :
    main()