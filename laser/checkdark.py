
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

"reva gal - 15/1 last meeting"
ANGELS6 =  [ 0 , 10 ,20 ,30, 40,
                50 ,60, 70, 80, 90,
                    100, 110, 120, 130,140,
                        150, 160, 170, 180, 190,
                         200, 210, 220, 230, 240, 250,
                            260, 270, 280, 290, 300, 310,
                             320, 330, 340, 350, 360  ]

angleee = [ 0, 45, 90, 135, 180, 225, 270, 315  ]
AMP6x =  [ 3500, 47000, 15700, 1000, 3690, 44700, 15000, 900 ]
AMP6y = [ 43000, 13700, 1197, 3600, 43900, 14700, 1170, 3000  ]

angleee_1 = [ 22, 68 , 112, 158, 202, 248, 292, 338  ]
AMP6x_1 =  [ 23500, 37700, 700, 240, 24700, 49800, 910, 293]
AMP6y_1 = [ 41500, 535, 388, 23900, 42000, 430, 461, 23900 ]


"revagal linear"
angleee7 = [ 0, 45, 90, 135, 180, 225, 270, 315  ]
AMP7x =  [ 3100, 42000, 15300, 930, 3600, 47000, 16000, 980]
AMP7y = [ 45000, 13300, 1290, 2700, 44500, 14000, 1255, 3300]

angleee7_1 = [ 22, 68 , 112, 158, 202, 248, 292, 338  ]
AMP7x_1 =  [ 23500, 37700, 700, 240, 24700, 49800, 910, 293]
AMP7y_1 = [ 41500, 535, 388, 23900, 42000, 430, 461, 23900 ]




"reva gal"
ANGELS5 =   [ 0 , 10 ,20 ,30, 40 ,50 ,60, 70, 80 ]
pass5x =    [ 17500 , 11000 ,3950 ,2150, 950 ,350 ,162, 54, 25]
# ret5x =     [ 0,19, 15.3, 11.9, 10.4 ,3 , 2.8 , 34.6, 84.5 , -, -, -, -, - ,-, -, -, -, -, -, -, - ]
# pass5y =    [ 39800, 40900, 38000, 33800 ,12000 ,1830, 235, 36 , 4, -, -, -, - ,-, -, -, -, -, -, -, - ] 
# ret5y =     [ 0,30, 23.6, 29.5, 94.5 , 201, 695, 640, 510 , -, -, -, -, - ,-, -, -, -, -, -, -, - ]

brosterlocalxpass = [ 
        [50, 51, 52, 53, 53, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68 , 69, 70, 75 ],
        [35200, 32100, 24200, 21300, 22900, 16900, 11800, 10700, 7550, 5000, 3600, 2500, 2300, 1960, 1740, 1380, 925, 480, 302, 260, 211 , 180, 160, 58], 
    ]

brosterlocalxret = [ 
        [50, 51, 52, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 70, 75 ],
        [2, 1.95, 1.7, 1.49, 1.36, 1.32, 1.32, 1.4, 1.54, 1.63, 1.95, 2.5, 3.7, 6.24, 13.3, 35.3  ]
]

ANGELS_LAST =   [ 0 , 20 ,40 ,60, 80 ,100 ,120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340 ]
AMP_LAST =   [ 41.9, 39.9, 40.1, 43, 44.6, 44, 43.9, 41.6, 39.9, 38.9, 40.1, 42, 43, 43.9, 43.8, 42.4, 42.4, 40.6  ]






# AMP4y = [ - , - ,- ,-, - ,- ,-, -, -, -, -, -, -, - ,-, -, -, -, -, -, -, - ]


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
    # # return 
    # light_strike = covert_data_frame(pandas.read_excel(_lightexp)) 
    # dark_strike = covert_data_frame(pandas.read_excel(_darkexp))
    # lightoffset = covert_data_frame(pandas.read_excel(_lightoffset))
    
    # print( light_strike)
    # # plt.plot( light_strike- np.average( lightoffset ) )
    # # plt.show()

    # print( np.linalg.norm( dark_strike - (light_strike- np.average( lightoffset ) ) ))
    # # plt.savefig( "" )

    # print(np.average( lightoffset ))
    # plot_phase_diff(ANGELS, np.array(AMP) - np.mean( lightoffset ))
    # plt.show()
    # # plot_phase_diff(ANGELS2, np.array(AMP2)  - np.mean( lightoffset )) # 
    # plot_phase_diff(ANGELS3, np.array(AMP3) / np.max(np.array(AMP3) ) ) # 
    # plt.show()
    # # plt.plot(*brosterlocalxpass)
    # plt.plot(*brosterlocalxret)
    # plt.show()
    # plt.plot(brosterlocalxret[0][:-4], brosterlocalxret[1][:-4])
    # plt.show()
    
    def plotpolar (A, X,Y):

        print( len(X ), len(Y))

        X = np.array(X)
        Y = np.array(Y)

        plt.polar(np.deg2rad(A),  (X + Y)**0.5 )

        # plt.scatter( AMP4x , AMP4y )
        plt.show()

    def merge( X,Y ):
        Z = [ ]
        for _ in range(min(len(X), len(Y))):
            Z.append( X[_] )
            Z.append( Y[_] )
        return np.array(Z)
    # plotpolar( angleee7, AMP7y,AMP7x )
    # plotpolar( merge(angleee,angleee_1), merge(AMP6y,AMP6y_1) , merge(AMP6x, AMP6x_1))
    global AMP_LAST
    AMP_LAST = np.array(AMP_LAST)
    AMP_LAST = AMP_LAST/ np.max(AMP_LAST)
    plt.polar(np.deg2rad( ANGELS_LAST ),  AMP_LAST )
        # plt.scatter( AMP4x , AMP4y )
    plt.show()


if __name__ == "__main__" :
    main()