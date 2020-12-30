import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit



K = 1.38*10**(-23)
T = 295
A = 2*T*K/(3*np.pi)

def extract_coef( time, distance , f = lambda t,a,b : a*t    ):
    popt, pcov = curve_fit(f, time, distance)
    return popt, f(time, *popt), pcov



def plot_linear_line_E(data, fig, case):

    t, (data, mean)= data
    data = (t, data)
    scale, _ = convert_meters_to_pix(1, 0)
    scale = scale  ** 2
    plt.title( r' $ E [ r^2 ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ s ]')
    plt.ylabel(r'$E [ r^2 ] $ [px]')
    coef, poly, pcov = extract_coef( *data )

    #  \sum( (r_i + r0)^2 ) / n = r0^2 +2 \cdot E[r] * r0

    _error = 1
    yerror = _error **2 + 2 * _error * mean  
    print(yerror)
    plt.plot(data[0], poly * scale, c = "teal", alpha=0.6)
    plt.errorbar(data[0], data[1] * scale, yerr = yerror, fmt='o', c= "black" ,alpha=0.2)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0] * scale)]) #] )
    fig.savefig("./fig/first/E-{0}.svg".format(case))
    return fig, coef

def plot_linear_line_V(data, fig, case , f = lambda t,a,b : a*t):
    plt.title( r' $ V [ r ] $ as function of time {0}'.format( case)  )
    plt.xlabel(r'time [ s ]')
    plt.ylabel(r'$V [ r ] $ [px]')
    coef, poly, pcov = extract_coef( *data , f= f )
    scale, _ = convert_meters_to_pix(1, 0)
    scale = scale ** 2 
    _error = 1
    plt.plot(data[0], poly * scale , c = "teal", alpha=0.6)
    plt.errorbar(data[0],data[1] * scale, yerr = _error ,fmt='o', c= "black" ,alpha=0.2)
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0] * scale )]  ) #] )
    fig.savefig("./fig/sec/E-{0}.svg".format(case))
    return fig, coef

def plot_linear_line_bolzman(data, fig, case ,relevent_radiuses, f = lambda t,a,b : a*t):
    plt.title( r' $ m = \frac{k_{B}T}{ 3 \pi \mu a } $ as function of $ \mu a $' )
    plt.xlabel(r'$ (\mu a )^{-1} $')
    plt.ylabel(r'$  \frac{k_{B}T}{ 3 \pi \mu a } $')
    scale, _ = convert_meters_to_pix(1, 0)

    yerrors = ((1 /(relevent_radiuses * scale) ) + 0.1536) * data[1] 
    xerrors = ((1 /(relevent_radiuses * scale)) + 0.1) * data[0]
    
    coef, poly, pcov = extract_coef( *data , f= f )
    plt.errorbar(data[0],data[1], yerr=yerrors, xerr=xerrors , fmt='o'  , c = "black" ,alpha=0.2)
    plt.plot(data[0], poly  , c = "teal", alpha=0.6)
    # _error = -0.1 * 10**-4 * scale
    plt.legend( [r'measured', r'liner fitting $D$=' + "{0:.3f}".format(coef[0]  )]  ) #] )
    fig.savefig("./fig/sec/E-{0}.svg".format(case))
    return fig, coef

def crop_windows( _arr, windowsize):
    return  _arr.reshape(-1, windowsize)

def convert_to_array( dataset ):
    return np.array( [dataset[i][1:] for i in range(3)] ).astype(float)

def convert_pix_to_meters(x,y):
    x = x*((10**(-5))/20) #TODO check scale
    y = y*((10**(-5))/20)
    return x,y 

def convert_meters_to_pix(x,y):
    x = x * 20 * 10**5 
    y =y* 20 * 10**5
    return x,y 



def createplotoneparticale( func, _plotfunc ):

    def genericplotoneparticale( dataset, viscosity, letter ):
        windowsize = 40

        dataset = convert_to_array(  dataset)[:,: 1 - (len(dataset[0]) % windowsize) ]
        t, x, y = map(lambda _arr : crop_windows(  _arr, windowsize), dataset ) 
        x, y = convert_pix_to_meters(x,y)
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
firsttry_plotoneparticale = createplotoneparticale( lambda _arr : ( np.mean(_arr**2, axis= 0), np.mean(_arr, axis= 0)) , plot_linear_line_E )


from thermo.chemical import Mixture
from random import random
import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

roomtempKelvin  = 298.15
earthpressure =  101325
bolzmanfactor = [ ]


precents = [ 0, 0.1, 0.2, 0.35, 0.5, 0.6] # 0.7, 0.8, 0.9, 0.95, 0.98 ]

# just for compiling. should take real values.
viscosities = [  Mixture(['glycerol','water'],ws=[ p, 1 - p ]).nu for p in precents ]

def get_radius():
    Kbs = []
    coeff_x = []
    coeff_y = []

    # dict, experiment -> radius 
    ret = {  }

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
            x, y = convert_pix_to_meters(x,y)

            f = lambda z: z - z[0]
            x = np.apply_along_axis(f, 1, x)
            y = np.apply_along_axis(f, 1, y)
            x = np.var(x, axis=0)
            y = np.var(y, axis=0)
            r_mass = x + y
            cur = A*1.9/(cur_v*r_mass[-1])
            if cur > 1:
                print("particle radius of ", _f, lets[j], ": bad format")
                continue
            coeff_x.append(cur)
            coeff_y.append(A/(cur_v*cur))
            print("particle radius of ", _f, lets[j], ": ", cur)
            ret[ _f + lets[j] ] = cur
            cur = truncate(cur, 5)
            # print(cur)
            Kbs.append(3*cur_v*np.pi*cur*r_mass[-1]/(2*1.9*294.15))

        i+=1

    # from scipy.stats import describe
    # print(describe(Kbs))
    # plt.plot(coeff_x, coeff_y, 'o', color='black')
    # plt.show()
    # print(coeff_x)
    # print(coeff_y)
    return ret


def original_tracker(radiuses):
    
    print(viscosities)
    _id = 0
    letters = [ 'A', 'B', 'C', 'D']
    calibrated = [ [] , [ ]]
    relevent_radiuses = [] 
    for _filename, (p, viscosity) in zip(open( "./csv/files" , "r").readlines()[1:], zip(precents, viscosities)):
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in letters ]

        np.array([ firsttry_plotoneparticale(particale, p, letter + "{0}".format(_id))\
             for particale,letter in zip(bunch,letters) ])
        
        coefs = np.array([ plotoneparticale(particale, p, letter + "{0}".format(_id))\
             for particale,letter in zip(bunch,letters) ])
        
        # coefs = coefs.astype(dtype=flaot64)
        constant = 3 *np.pi 
        for j, letter in enumerate(letters):
            if  _filename + letter in radiuses and coefs[j][-1][0] != 0:
                calibrated[0].append( (radiuses[ _filename + letter ] * viscosity * constant) ** -1 )
                relevent_radiuses.append(radiuses[ _filename + letter ])
                calibrated[1].append(  coefs[j][-1][0] )  

        # bolzmanfactor_estimate ,poly = extract_coef( (1/constant) * ) ,  coefs[:,:,0][j,0])        
        # bolzmanfactor.append( bolzmanfactor_estimate[0] )
        _id += 1
    relevent_radiuses = np.array(relevent_radiuses)
    calibrated = np.array(calibrated )
    # calibrated = np.array([calibrated[ _ ].flatten() for _ in range(2)])
    # calibrated = np.var(calibrated, axis= 1)
    print(calibrated)
    print(calibrated.shape)
    plt.clf()
    aa = plt.gcf()
    #plt.scatter( calibrated[0] ,calibrated[1]  ) 
    aa, coef = plot_linear_line_bolzman( ( calibrated[0] ,calibrated[1] ), aa, "calibrated" , relevent_radiuses, f = lambda t,a,b : a*t)
    plt.show()
    print(  coef[0] /   roomtempKelvin ,coef[1] )
    import scipy.constants as constants
    print ( constants.k )
    # print( bolzmanfactor )
    # print(np.mean( np.array( bolzmanfactor ) ) / roomtempKelvin)

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
    return ["{0:.3f}".format(float(_)) for _ in _arr]

def generateTable():
    return generateGeneralTable([ ["precents"] + stringilize(precents), ["viscositie[Pa*s]"] + stringilize(viscosities)]) 

def generateDataTables():
    ret = [ ]
    for _filename, viscosity in zip(open( "./csv/files" , "r").readlines()[1:], viscosities):
        bunch = [ pd.read_csv('./csv/BrownCSV/{0}'.format(_filename.format(letter)[:-1] ),\
            sep=',',header=None)  for letter in [ 'A', 'B', 'C', 'D'] ]

        # result = pd.concat(bunch, axis=1, join="inner")
        # for particale in bunch :
        # 
        ret.append ( """ \\begin_layout Plain Layout\n {0} \\end_layout\n """.format( _filename ) )
        for particale in bunch:
            gen = particale.iterrows()
            __ , title = next( gen )
            rows =  [title.astype(str)] + [ stringilize(row) for _, row in gen ]
            for chunk in np.array_split(rows, len(rows)// 30):
                ret.append( generateGeneralTable( chunk.tolist())  )
        # ret.append( generateGeneralTable(rows) )
    return "\n".join(ret)


from os.path import abspath


def insertFig ( _path  ):
    return """
\\begin_inset Graphics\n
	filename {0}\n
	scale 50\n
\\end_inset\n""".format( abspath(_path) )


from os import walk

def generateFigsOneParticale( ):
    _str, i  = "\\begin_layout\n", 0
    for (root,dirs,files) in walk('Fig/sec'): 
        for _file in files :
            if len(_file) < 18 and "E-r-" in _file : 
                _str += insertFig( "Fig/sec/" + _file )
                # if i % 2 == 0 and i > 0:
                _str += "\n\n\\end_layout\n\n\\begin_layout\n"
                # i += 1

    return _str  +"\\end_layout\n"

# "generateDataTables\n" : generateDataTables,
def generatelyx( ):
    _generators = { "generateTable\n" : generateTable,  "generateFigsOneParticale\n" : generateFigsOneParticale , "generateDataTables\n" : generateDataTables}
    _str = ""
    for line in open( './doc/papertemplate.lyx', 'r', encoding="utf-8" ).readlines():
        if line in _generators: 
            _str += _generators[line]()
        else:
            _str += line

    open( './doc/paper.lyx', 'w+', encoding="utf-8" ).write( _str )



if __name__ == "__main__" :
    radiuses = get_radius()
    original_tracker(radiuses)
    generatelyx()
# ()