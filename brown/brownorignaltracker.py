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
    plt.plot(*data)
    plt.title( r' $ E [ r^2 ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ frames ]')
    plt.ylabel(r'$r$ [px]')
    coef, poly = extract_coef( *data )
    plt.plot(data[0], poly)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0])]) #] )
    # fig.savefig("./fig/E-{0}.png".format(case))
    return fig
    


def plotoneparticale( dataset ):
    windowsize = 20

    dataset = np.array( [dataset[i][1: 1-(len(dataset[0]) % windowsize) ] for i in range(3)] ).astype(float)
    dataset = dataset.reshape( 3, len(dataset[0]) //windowsize, windowsize ) 
    t, x, y = dataset
    
    def set_first_point_to_zero(_arr):
        _arr = _arr.transpose()
        return ( _arr -  _arr[0]).transpose() 
    for axis, _arr in zip( 'xyr' , [ x, y, (x**2 + y**2)**0.5 ]):
        _arr = set_first_point_to_zero(_arr)
        fig = plt.gcf()
        plot_linear_line( [t[0], np.var(_arr, axis= 0)], fig, "{0}".format(axis) )
        plt.show()

def original_tracker():
    for _filename in open( "./csv/files" , "r").readlines()[1:]:
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in [ 'A', 'B', 'C', 'D'] ]
        plotoneparticale(bunch[0])



if __name__ == "__main__" :
    original_tracker()