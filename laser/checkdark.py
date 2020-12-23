
import numpy as np 
from matplotlib import pyplot as plt 

import pandas


from scipy.optimize import curve_fit

def extract_coef( time, distance ):
    def f(x, a, b, c, d):
        return a*np.cos(b*x +c) + d
    popt, pcov = curve_fit(f, time, distance)
    _range = np.linspace( np.min(time) , np.max(time), 100 )
    return popt, (_range, f(_range, *popt)) 

""" cm """
LASER_POSITION = 89 
RESONATOR_POSITION = 7
CONSTAT_POL_POSITION = 43.5 
CONSTAT_POL_ANGLE = 223

_lightexp = "./csv/STRIKE/test-1.xlsx" 
_darkexp = "./csv/STRIKE-DARK/test-1.xlsx"
_lightoffset = "./csv/OFFSET/test-4.xlsx"


ANGELS = [ 218, 200, 180, 160, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10 , 0, 350]
""" Light [\mu A]  """
AMP = [ 1.79, 1.605,  1.651, 1.618, 2.21, 3.210, 4.51, 5.66, 6.51, 6.67, 6.33, 5.08, 3.77, 2.65, 1.97,  1.626, 1.55, 1.603, 1.602, 1.57 ]


def covert_data_frame( df ):
    return df[0][5:].values.astype(float) 


def main():
    
    light_strike = covert_data_frame(pandas.read_excel(_lightexp)) 
    dark_strike = covert_data_frame(pandas.read_excel(_darkexp))
    lightoffset = covert_data_frame(pandas.read_excel(_lightoffset))
    
    print( light_strike)
    # plt.plot( light_strike- np.average( lightoffset ) )
    # plt.show()

    print( np.linalg.norm( dark_strike - (light_strike- np.average( lightoffset ) ) ))
    # plt.savefig( "" )

    x, y = np.deg2rad((np.array(ANGELS) - CONSTAT_POL_ANGLE) % 360), np.array(AMP)
    coeff, cosfunc = extract_coef(x, y)
    plt.plot(x,y)
    plt.plot(cosfunc[0], cosfunc[1])
    plt.show()

if __name__ == "__main__" :
    main()