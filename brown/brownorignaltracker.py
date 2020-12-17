import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit





def extract_coef( time, distance ):
    def f(x, a, b):
        return a*x + b
    popt, pcov = curve_fit(f, time, distance)
    return popt, f(time, *popt) 


def plot_linear_line(data, fig, case):
    plt.scatter(*data)
    plt.title( r' $ E [ r^2 ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ frames ]')
    plt.ylabel(r'$r$ [px]')
    coef, poly = extract_coef( *data )
    plt.plot(data[0], poly)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0])]) #] )
    fig.savefig("./fig/E-{0}.png".format(case))
    return fig, coef
    

def plotoneparticale( dataset ):
    windowsize = 20

    dataset = np.array( [dataset[i][1: 1-(len(dataset[0]) % windowsize) ] for i in range(3)] ).astype(float)
    dataset = dataset.reshape( 3, len(dataset[0]) //windowsize, windowsize ) 
    t, x, y = dataset
    
    def set_first_point_to_zero(_arr):
        _arr = _arr.transpose()
        return ( _arr -  _arr[0]).transpose() 

    ret = []
    for axis, _arr in zip( 'xyr' , [ x, y, (x**2 + y**2)**0.5 ]):
        _arr = set_first_point_to_zero(_arr)
        fig = plt.gcf()
        _, coef = plot_linear_line( [t[0], np.var(_arr, axis= 0)], fig, "{0}".format(axis) )
        # plt.show()
        ret.append( coef )
    return ret

from thermo.chemical import Mixture
from random import random 

roomtempKelvin  = 294.15
earthpressure =  101325
bolzmanfactor = [ ]

# just for compiling. should take real values.
viscosities = [  Mixture(['glycerol','water'],ws=[ p, 1 - p ],T=roomtempKelvin,P=earthpressure).mu\
        for p in [ 0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98 ] ]

def original_tracker():
    
    print(viscosities) 

    for _filename, viscosity in zip(open( "./csv/files" , "r").readlines()[1:], viscosities):
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in [ 'A', 'B', 'C', 'D'] ]

        # just for compiling. should take real values.
        radiuses = np.array([ random( ) * 10 ** -5   for _ in range(4) ])
        coefs = np.array([ plotoneparticale(particale) for particale in bunch ])        
        
        constant = 3 *np.pi 
        print(coefs[:,:,0][:,0])
        coefs ,poly = extract_coef( 1/(constant * radiuses) ,  coefs[:,:,0][:,0])        
        bolzmanfactor.append( coefs[0] )

    print( bolzmanfactor )
    print(np.mean( np.array( bolzmanfactor ) ) / roomtempKelvin)

def generateTable():
    return "\\begin{{center}}\n\\begin{{tabular}}{0}\\end{{tabular}}\\end{{center}}".format(\
          "& ".join(["{0}".format(cell) for cell in viscosities]))

def generatelyx( ):

    _generators = { "generateTable\n" : generateTable }
    _str = ""
    for line in open( './doc/paper.lyx', 'r', encoding="utf-8" ).readlines():
        if line in _generators: 
            _str += _generators[line]()
        else:
            _str += line

    open( './doc/paper.lyx', 'w', encoding="utf-8" ).write( _str )

if __name__ == "__main__" :
    original_tracker()
    generatelyx()