from imageio import imread
import numpy as np


testcases = [ "t10p20g.avi0004{0}.png".format(_) for _ in range(10) ] 


def define_limit():
    ''' 
        define the frame, marks the right shadow as outside of the image.   
    '''
    pass


def marks_objects(frame):
    '''
        detect the particals, as first stage, using naive approach. 
    '''
    pass

def find_matching(prevframe, nextframe):
    '''
        find match between pair of ascending frames 
    '''
    pass

def plot_graphs():
    '''
    '''
    pass


def test_read():
    _arr = [ ]
    for testcase in testcases:
        _arr.append( imread(testcase))
    print(_arr[-1] )

if __name__ == "__main__":
    test_read()



