from imageio import imread, imwrite
import numpy as np

import matplotlib.pyplot as plt


from pickle import dump, load
from pathlib import Path

BLACK = 0
WHITE = 255

THRESHOLD = 10

HISTS = 0, 1000

# numberofcases = 10


testcases_exp = [[  clip +".avi{0}{1}.png".format("0" * (5 - len(str(3*_)))   , 3 * _) for _ in range(1, 40) ] for clip in [ "tstraigh2t10ppure", "t10p10g.avi", "t10p20g", "t10p35g", "t10p50g", "t10p60g", "t10p75g", "t10p85g" ]]

testcasesout = [ "rect10p20g.avi000{0}.png".format(_) for _ in range(10) ]

EPS = 100

COLORS = set()

class particale():

    
    id = 0

    def __init__(self, positions):
        self.x = np.array( [  position[0] for position in positions  ] )
        self.y = np.array( [  position[1] for position in positions  ] )
        self.CM = np.array( [sum(self.x), sum(self.y)] ) / len(self.x)      

        self.next = None
        self.prev = None
        
        self.id = particale.id
        particale.id += 1

    def distance(self, other):
        return np.linalg.norm(self.CM - other.CM)

    def concate(self, other):
        self.next = other
        other.prev = self 
    
    def draw_arrow(self, matrix ):
        
        if self.id not in COLORS: 
            COLORS.add(self.id)

            # PATCH 
            number_of_points = 10
            if self.prev is not None: 
                xs= np.linspace(self.CM[0],self.prev.CM[0],number_of_points+2)
                ys= np.linspace(self.CM[1],self.prev.CM[1],number_of_points+2)
                
                for x,y in zip(xs, ys):  
                    draw_red_pont(int(x), int(y), matrix )
                
                self.prev.draw_arrow(matrix)    

    def last(self):
        ret = self
        while ret.next != None:
            ret = ret.next
        return ret

    def calculate_total_distance(self):
        _last = self.last()
        return np.sqrt(self.distance())

def reasonable_kernel(_particale):
    temp = _particale
    while temp.next != None: 
        if temp.distance( temp.next  ) > 10:
            if temp.prev is None:
                return None
            else:
                temp.prev.next = None
                return _particale
        temp = temp.next
    return _particale

def calculate_time_distance(_particale):
    temp = _particale
    ret = [ ]
    while temp.next != None:
        ret.append( temp.distance( _particale ) )
        temp = temp.next
    return np.array(ret), np.arange(len(ret))

def set_red( v ):   
    v[0], v[1], v[2] = 255, 0, 0

def draw_red_pont(x, y, matrix):
    set_red(matrix[HISTS[0] + x][HISTS[1] + y]) 

def create_particales(  positions_lists ):
    return [ particale(positions) for positions in positions_lists ]

def define_limit():
    ''' 
        define the frame, marks the right shadow as outside of the image.   
    '''
    pass


def frame_iterator(frame):
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            yield (x,y), frame[x][y]

        
def marks_objects(frame):
    '''
        detect the particals, as first stage, using naive approach. 
    '''

    flags = np.zeros(shape = (frame.shape[0],frame.shape[1]) )
    def mark_dfs(x,y, _obj = [], depth = 100 ):
        if depth > 0 and x < frame.shape[0] and y < frame.shape[1]:   
            if flags[x][y]:
                return _obj
           
            flags[x][y] = 1
            
            if frame[x][y][0] < THRESHOLD:
                _obj.append( (x, y) )

                mark_dfs(x,y+1, _obj=_obj, depth = depth -1)
                mark_dfs(x+1,y, _obj=_obj, depth = depth -1)
                mark_dfs(x-1,y, _obj=_obj, depth = depth -1)
                mark_dfs(x,y-1, _obj=_obj, depth = depth -1)
        
        return _obj

    objects = []
    for (x,y), value in frame_iterator(frame):
        if not flags[x][y]:
            __obj = mark_dfs(x,y, _obj = [])
            if len(__obj) > 0 :
                objects.append( __obj)

    return objects

def rect_object(_obj, frameRGB):
    
    vertex = []
    for func in [ min, max ]:
        for _ in [ 0 , 1 ]:  
            vertex.append(func( _obj, key = lambda u : u[_] )[_])
    x1, y1, x2, y2 = vertex

  

    for x in range(x1, x2):
        set_red(frameRGB[HISTS[0] + x][HISTS[1] + y1]) 
        set_red(frameRGB[HISTS[0] + x][HISTS[1] + y2])
    
    for y in range(y1, y2):
        set_red(frameRGB[HISTS[0] + x1][HISTS[1] + y]) 
        set_red(frameRGB[HISTS[0] + x2][HISTS[1] + y])

def find_matching(prev_particals, particals):
    '''
        find match between pair of ascending frames 
    '''    
    for prev in prev_particals:
        option = min(particals, key= lambda current: prev.distance(current))
        if np.sqrt(prev.distance( option )) < EPS : 
            if (option.prev is None) or option.distance(option.prev ) >  prev.distance( option ):
                prev.concate( option )



def plot_graphs():
    '''
    '''
    pass


def test_read():
    _arr = [ ]
    for w, testcases in  enumerate(testcases_exp):
        particales_frames = [ ]
        if not Path("pickleout{0}.pkl".format(w)).exists():
            for j, testcase in enumerate( testcases ):
                if Path(testcase).exists():
                    _arr.append( imread(testcase))
                    _objs = marks_objects(_arr[-1][:,HISTS[1]:1400])
                    particales_frames.append(  create_particales(_objs)  )
                    # for _obj in _objs:
                    #     rect_object(_obj, _arr[-1])
                    #     imwrite( testcasesout[j],  _arr[-1])
                    if len( particales_frames ) > 1:
                        find_matching( particales_frames[-2], particales_frames[-1] )
                else:
                    print("path {0} doesn't exists".format(testcase))

            for particale_frame in particales_frames:
                for particale in particale_frame:  
                    particale.draw_arrow(_arr[-1])


            plt.imshow(_arr[-1])
            imwrite( "out{0}.png".format(w) ,  _arr[-1])
            dump(particales_frames,open("pickleout{0}.pkl".format(w) , "wb+"))
        else:
            particales_frames = load(open("pickleout{0}.pkl".format(w) , "rb"))
            print(particales_frames[0][-1].CM)
            
            reasonable = list(map(reasonable_kernel, particales_frames[0]))
            reasonable = list(filter( lambda x : x != None, reasonable))
            # print(reasonable)
            _temp_ = list(map(calculate_time_distance, reasonable))
            # print(_temp_)
            distance_time = list(filter(lambda x: len(x[0]) > 30 ,  _temp_))
            if len(distance_time) == 0:
                continue 
            cut  = len(min (distance_time, key= lambda x: len(x[0]) )[0])
            print(distance_time[0])
            distance_time = map( lambda x : x[0][:cut], distance_time)
 
            distance_time = np.array( list(distance_time) )
            print(distance_time.shape)
            if len(distance_time) < 9:
                continue

            fig, axs = plt.subplots(3, 3)
            # axs[0, 0].plot(x, y)
            # axs[0, 0].set_title('Axis [0, 0]')
            # axs[0, 1].plot(x, y, 'tab:orange')
            # axs[0, 1].set_title('Axis [0, 1]')
            # axs[1, 0].plot(x, -y, 'tab:green')
            # axs[1, 0].set_title('Axis [1, 0]')
            # axs[1, 1].plot(x, -y, 'tab:red')
            # axs[1, 1].set_title('Axis [1, 1]')

            # for ax in axs.flat:
            #     ax.set(xlabel='x-label', ylabel='y-label')

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            # for ax in axs.flat:
            #     ax.label_outer()
            
            # import matplotlib 
            # matplotlib.rcParams['text.usetex'] = True

            fig.suptitle( r'graphs for first (make sense) nine particales, $ | X_{t+1}  - X_{t} | < \varepsilon $' )
            
            for _ in range(3):
                axs[2, _].set_xlabel(r'time [ 4 - frame ]')
                axs[_, 0].set_ylabel(r'$r$ [px]')

            for i in range(3):
                for j in range(3):
                    axs[i, j].plot(distance_time[i*3 + j].copy())
            

            fig.savefig("./fig/9-{0}.png".format(testcases[0]))
            # plt.show()
            # plt.close()
            plt.clf()
            # fig  = plt.gcf()
            plt.plot(np.mean( distance_time, axis=0))
            plt.title(  r' $ E [ r ] $ as function of time '  )
            plt.xlabel(r'time [ 4 - frame ]')
            plt.ylabel(r'$r$ [px]')
            fig.savefig("./fig/E-{0}.png".format(testcases[0]))
            # plt.show()


if __name__ == "__main__":
    test_read()



