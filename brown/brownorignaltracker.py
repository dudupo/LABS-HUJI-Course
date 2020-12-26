import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit



K = 1.38*10**(-23)
T = 295
A = 2*T*K/(3*np.pi)

def extract_coef( time, distance ):
    def f(t, a, b):
        return a*t + b
    popt, pcov = curve_fit(f, time, distance)
    return popt, f(time, *popt) 


def plot_linear_line_E(data, fig, case):
    plt.scatter(*data)
    plt.title( r' $ E [ r^2 ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ frames ]')
    plt.ylabel(r'$E [ r^2 ] $ [px]')
    coef, poly = extract_coef( *data )
    plt.plot(data[0], poly)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0])]) #] )
    fig.savefig("./fig/first/E-{0}.png".format(case))
    return fig, coef

def plot_linear_line_V(data, fig, case):
    plt.scatter(*data)
    plt.title( r' $ V [ r ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ frames ]')
    plt.ylabel(r'$V [ r ] $ [px]')
    coef, poly = extract_coef( *data )
    plt.plot(data[0], poly)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0])]) #] )
    fig.savefig("./fig/sec/E-{0}.png".format(case))
    return fig, coef


def createplotoneparticale( func, _plotfunc ):

    def genericplotoneparticale( dataset, viscosity, letter ):
        windowsize = 60

        dataset = np.array( [dataset[i][1: 1-(len(dataset[0]) % windowsize) ] for i in range(3)] ).astype(float)
        dataset = dataset.reshape( 3, len(dataset[0]) //windowsize, windowsize )
        t, x, y = dataset
        
        def set_first_point_to_zero(_arr):
            _arr = _arr.transpose()
            return ( _arr -  _arr[0]).transpose() 

        ret = []
        for axis, _arr in zip( 'xyr' , [ x, y, (x**2 + y**2)**0.5 ]):
            _arr = set_first_point_to_zero(_arr)
            plt.clf()
            fig = plt.gcf()
            _, coef = _plotfunc( [t[0], func(_arr)], fig, "{0}-g{1:.3f}-{2}".format(axis, viscosity, letter) )
            # plt.show()
            ret.append( coef )
        return ret
    return genericplotoneparticale

plotoneparticale = createplotoneparticale( lambda _arr : np.var(_arr, axis= 0), plot_linear_line_V )
firsttry_plotoneparticale = createplotoneparticale( lambda _arr : np.mean(_arr**2, axis= 0), plot_linear_line_E )


from thermo.chemical import Mixture
from random import random
import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

roomtempKelvin  = 298.15
earthpressure =  101325
bolzmanfactor = [ ]


precents = [ 0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98 ]

# just for compiling. should take real values.
viscosities = [  Mixture(['glycerol','water'],ws=[ p, 1 - p ]).mu\
        for p in precents ]


def get_radius():
    Kbs = []
    coeff_x = []
    coeff_y = []
    for _f in open( "./csv/files" , "r").readlines()[1:]:
        i = 0  # TODO update
        cur_v = viscosities[i]
        lets = [ 'A', 'B', 'C', 'D']
        data = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_f.format(letter)[:-1] ))  for letter in lets ]
        for j, mass in enumerate(data):
            try:
                x = mass['x']
                y = mass['y']
            except:
                print("particle radius of ", _f, lets[j], ": bad format" )
                continue
            l = len(x) % 20
            if l != 0:
                x = x[:-l]
                y = y[:-l]
            x = x.to_numpy().reshape(-1, 20)
            y = y.to_numpy().reshape(-1, 20)
            x = x*((10**(-4))/2) #TODO check scale
            y = y*((10**(-4))/2)

            f = lambda z: z - z[0]
            x = np.apply_along_axis(f, 1, x)
            y = np.apply_along_axis(f, 1, y)
            x = np.var(x, axis=0)
            y = np.var(y, axis=0)
            r_mass = np.power(x,2) + np.power(y,2)
            # r_mass = np.apply_along_axis(f, 1, r_mass)
            # r_mass = np.var(r_mass, axis=0)
            # plt.plot(r_mass)
            cur = A*1.9/(cur_v*r_mass[-1])
            if cur > 1:
                print("particle radius of ", _f, lets[j], ": bad format")
                continue
            coeff_x.append(cur)
            coeff_y.append(A/(cur_v*cur))
            print("particle radius of ", _f, lets[j], ": ", cur)
            cur = truncate(cur, 5)
            print(cur)
            Kbs.append(3*cur_v*np.pi*cur*r_mass[-1]/(2*1.9*294.15))

        i+=1

    from scipy.stats import describe
    print(describe(Kbs))
    plt.plot(coeff_x, coeff_y, 'o', color='black')
    plt.show()
    print(coeff_x)
    print(coeff_y)







def original_tracker():
    
    print(viscosities)
    _id = 0
    for _filename, viscosity in zip(open( "./csv/files" , "r").readlines()[1:], viscosities):
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in [ 'A', 'B', 'C', 'D'] ]

        # just for compiling. should take real values.
        radiuses = np.array([ random( ) * 10 ** -5   for _ in range(4) ])
        coefs = np.array([ firsttry_plotoneparticale(particale, viscosity, letter + "{0}".format(_id)) for particale,letter in zip(bunch,[ 'A', 'B', 'C', 'D']) ])
        coefs = np.array([ plotoneparticale(particale, viscosity, letter + "{0}".format(_id)) for particale,letter in zip(bunch,[ 'A', 'B', 'C', 'D']) ])
        print(coefs.shape)
        
        constant = 3 *np.pi 
        # print(coefs[:,:,0][:,0])
        coefs ,poly = extract_coef( 1/(constant * radiuses) ,  coefs[:,:,0][:,0])        
        bolzmanfactor.append( coefs[0] )
        _id += 1

    print( bolzmanfactor )
    print(np.mean( np.array( bolzmanfactor ) ) / roomtempKelvin)

def generateGeneralTable(  rows ):

    def _generateGeneralTable ( rows ):
        if len(rows) == 0 :
            return ""
        else :
            return """
\\begin_layout Plain Layout\n
{0} 
\\backslash 
\\backslash \n
\\end_layout  """.format(" &".join( rows[0][::-1] ) ) +  _generateGeneralTable(rows[1:])

    return """
\\begin_layout Standard\n
\\begin_inset ERT\n
status open\n
\\begin_layout Plain Layout\n
\\backslash
begin{{center}}\n
\\end_layout\n
\\begin_layout Plain Layout\n
\\backslash
begin{{tabular}}{{  |{0}| }}\n
\\end_layout\n
{1}\n
\\begin_layout Plain Layout\n
\\backslash
end{{tabular}}\n
\\end_layout\n
\\begin_layout Plain Layout\n
\\backslash
end{{center}}\n
\\end_layout\n
\\end_inset\n
\\end_layout\n
        """.format( "|".join( ['c'] * len(rows[0]) ) , _generateGeneralTable(rows) )

def stringilize(_arr):
    return ["{0:.3f}".format(_) for _ in _arr]

def generateTable():
    return generateGeneralTable([ ["precents"] + stringilize(precents), ["viscositie[Pa*s]"] + stringilize(viscosities)]) 

def generateDataTables():
    ret = [ ]
    for _filename, viscosity in zip(open( "./csv/files" , "r").readlines()[1:], viscosities):
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in [ 'A', 'B', 'C', 'D'] ]

        for particale in bunch :
            print(particale)
            rows = [ particale[_] for _ in range(3) ]
            #  +  list(map(stringilize,  np.array( [ _[1:] for  _ in particale ] ).transpose()))
            ret.append( generateGeneralTable(rows) )
    return "\n".join(ret) 


from os.path import abspath


def insertFig ( _path  ):
    return """
\\begin_inset Graphics\n
	filename {0}\n
	scale 65\n
\\end_inset\n""".format( abspath(_path) )


from os import walk

def generateFigsOneParticale( ):
    _str, i  = "\\begin_layout\n", 0
    for (root,dirs,files) in walk('Fig'): 
        for _file in files :
            if len(_file) < 18 and "E-r-" in _file : 
                _str += insertFig( "Fig/" + _file )
                if i % 2 == 0 and i > 0:
                    _str += "\n\n\\end_layout\n\n\\begin_layout\n"
                i += 1

    return _str  +"\\end_layout\n"

# "generateDataTables\n" : generateDataTables,
def generatelyx( ):
    _generators = { "generateTable\n" : generateTable,  "generateFigsOneParticale\n" : generateFigsOneParticale }
    _str = ""
    for line in open( './doc/papertemplate.lyx', 'r', encoding="utf-8" ).readlines():
        if line in _generators: 
            _str += _generators[line]()
        else:
            _str += line

    open( './doc/paper.lyx', 'w+', encoding="utf-8" ).write( _str )



if __name__ == "__main__" :
    original_tracker()
    # get_radius()
    generatelyx()
# ()