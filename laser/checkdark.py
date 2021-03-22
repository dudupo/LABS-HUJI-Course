
import numpy as np 
from matplotlib import pyplot as plt 

import pandas


from scipy.optimize import curve_fit

def extract_coef( time, distance ):
    def f(x, a, a2, b, b2, c, c2, d ):
        return a*np.cos(b*x+c)**2 + a2*np.cos(b2*x+c2)**2 + d 
    popt, pcov = curve_fit(f, time, distance)
    _range = np.linspace( np.min(time) , np.max(time), 100 )
    return popt, (_range, f(_range, *popt)) 


def extract_coef_onecos( time, distance ):
    def f(x, a, b, c, d ):
        return a*np.cos(b*x+c)**2 
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


"in the dark"
ANGELS3 = [ 0 , 10 ,20 ,30, 40 ,50 ,60, 70, 80, 90, 100, 110, 120, 130 ,140, 150, 160, 170, 180, 190, 200, 210 ]
AMP3 = [ 129, 332, 885, 1720, 2700, 3900, 4800, 5100, 4930, 4080, 3000, 1930, 1030, 413, 108, 15, 6, 7.5, 48, 260, 700, 1470]

"reva gal"
ANGELS4 = [ 0 , 10 ,20 ,30, 40 ,50 ,60, 70, 80, 90, 100, 110, 120, 130 ,140, 150, 160, 170, 180, 190, 200, 210 ]
AMP4x = [ 7.2, 8.5, 12.9, 8.1, 9.3, 46, 180, 410, 710, 1050, 1200, 1280, 1210, 945, 645, 390, 160, 48, 8, 13, 8.7] #8.4
AMP4y = [ 8.9, 20, 93, 290, 520, 710, 900, 870, 706, 527, 260, 115, 110, 18, 12, 50, 83, 48, 16, 104, 273]


def covert_data_frame( df ):
    return df[0][5:].values.astype(float) 


def plot_phase_diff(angels,  amp):
    x, y = np.deg2rad((np.array(angels) - CONSTAT_POL_ANGLE) % 360), np.array(amp)
    plt.plot(x,y)
    
    for _func in [extract_coef_onecos ]:
        coeff, cosfunc = _func(x, y)
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
    plt.show()

    print( np.linalg.norm( dark_strike - (light_strike- np.average( lightoffset ) ) ))
    # plt.savefig( "" )

    print(np.average( lightoffset ))
    plot_phase_diff(ANGELS, np.array(AMP) - np.mean( lightoffset ))
    # plot_phase_diff(ANGELS2, np.array(AMP2)  - np.mean( lightoffset )) # 
    plot_phase_diff(ANGELS3, np.array(AMP3) / np.max(np.array(AMP3) ) ) # 
    plt.show()
    print( len(AMP4x ), len(AMP4y))
    AMP4Y, AMP4X = np.array(AMP4y), np.array(AMP4x)  
    
    comp =  AMP4X + 1j * AMP4Y
    
    teta = np.angle( comp )
    r = np.abs( comp )
    plt.polar( teta , r )
    plt.show()

if __name__ == "__main__" :
    main()