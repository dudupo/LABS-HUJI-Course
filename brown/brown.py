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

testcases_exp = [[ "t" + clip +"{0}{1}.png".format("0" * (5 - len(str(1*_)))   , 1 * _)\
     for _ in range(1, 100) ] for clip in\
         [ 
            "10p0g22c.avi.avi",
            "10p0gsec.avi",
            "10p10g.avi.avi",
            "10p20g.avi",
            "10p35g.avi",
            "10p50g.avi",
            "10p60g.avi",
            "10p75g.avi",
            "10p85g.avi",
            "10p95g22.7c.avi.avi",
            "10p98g22.6.avi.avi",
            "10ppure.avi.avi",
            "10ppuresec2.avi",
            "10ppuresec3.avi",
            "10ppuresec4.avi",
            "10ppuresec5.avi",
            "straigh2t10ppure.avi",
            "straight10ppure.avi.avi"
         ]\
        ]


# testcasesout = [ "rect10p20g.avi000{0}.png".format(_) for _ in range(10) ]

EPS = 9

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

    # def matrix_representation(self):
    #     _maxx, _maxy = np.max( self.x ), np.max( self.y ) 
    #     _minx, _miny = np.min( self.x ), np.min( self.y )
        
    #     self._matrix =  
        
        # pass
    # def __mul__(self, grades):
    
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
        # return np.sqrt(self.distance())

    def calculate_total_grad(self):
        i = 0 
        temp = self
        while temp.next != None:
            temp = temp.next
            i += 1
        return temp .CM[0] -  self.CM[0], temp.CM[1] -  self.CM[1]
    
    def reduce_mean(self , _mean):
        temp = self
        while temp != None:
            temp.CM[0] -= _mean[0]
            temp.CM[1] -= _mean[1]
            temp = temp.next
    
    def shellcopy(self):
        return particale( zip( self.x.tolist(), self.y.tolist() ))
         



def leaves_generator( particales_frames_array ):
        
        # flatten the list : 
        _map =  { _particale.id : _particale for  particales_frame in particales_frames_array for _particale in particales_frame  }

        while len(_map) > 0:
            _id, _particale = _map.popitem() 
            while _particale.prev != None and _particale.id in _map:
                del _map [ _particale.id ] 
                _particale = _particale.prev
            if _particale.id in _map:
                del _map [ _particale.id ]
            yield _particale            

def reduce_mean_mean(particales):
    X_mean, Y_mean = [], [] 
    for _particale in particales:
        x, y = _particale.calculate_total_grad()
        X_mean.append(x)
        Y_mean.append(y)
    np.array(X_mean)
    np.array(Y_mean)
    x_mean, y_mean  = np.mean(X_mean), np.mean(Y_mean)
    print(x_mean, y_mean)
    for _particale in particales:
        _particale.reduce_mean((x_mean, y_mean))
    
    return particales

def reasonable_kernel(_particale):
    temp = _particale
    while temp.next != None: 
        if temp.distance( temp.next  ) > 20: 
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


MASSEPS = 50
def find_matching(prev_particals, particals):
    '''
        find match between pair of ascending frames 
    '''    
    for prev in prev_particals:

        options = list(filter( lambda current : prev.distance( current) < EPS, particals))
        not_matched = True

        massdiff = lambda x, y : np.abs( len(x.x) - len(y.x) ) 
        
        if len(options) > 0 :
            option = min( options , key= lambda current: massdiff( current, prev) )
            # if prev.distance( option ) < EPS : 
                # if massdiff(option.prev, option) >  prev.distance( option ):
                    #(option.prev is None) or
            prev.concate( option )
            not_matched = False
        if not_matched:
            shallcopy = prev.shellcopy()
            # prev.concate( shallcopy )
            # particals.append( shallcopy )

def create_matching_net(particales_frames_array, _estimate_func):
    
    def handle_adjacent_frames(prev_particals, particals):
        ret = np.zeros( shape = ( len(prev_particals), len(particals) ) )
        for (i,j), _val in np.ndenumerate(ret): 
            ret[i,j] = _estimate_func(  prev_particals[i], particals[j] )
        return ret
    prev_particals_lists, particals_lists = particales_frames_array[:-2], particales_frames_array[1]
    for prev_particals, particals in zip ( prev_particals_lists, particals_lists ):
        _matrix = handle_adjacent_frames(prev_particals, particals)
        for (i,j), _val in np.ndenumerate(_matrix):
            pass
    pass

def plot_graphs():
    '''
    '''
    pass


def mapnumpy( _function, _list ):
    return np.array( list(map(_function , _list)) )

def quantenize_mass(particals):
    masses = mapnumpy( lambda p: len(p.x),  particals )
    bins = np.quantile(masses , [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 0.95])
    inds = np.digitize(masses, bins)
    rets = [ list() for _ in range(len(bins) +1 ) ] 
    for  j, index in enumerate(inds):
        rets[index-1].append( particals[j] )
        # print(rets)
    # print(rets)
    return rets, bins


def padding_list_with_nan(lists):
    maxlen = max(lists, key = len(lists) )


def calculate_average( distance_time ):
    '''
        when our limits of the arrays are not equal

        E [ x ... x_n  ], E [ x ... x_n-1  ], ..
    '''
    _indices = sorted( [ len(_arr[0]) for _arr in distance_time ] )

    indices = [ _indices[0] ]
    for val in _indices[1:]:
        if val != indices[-1]:
            indices.append( val ) 
     
    left, right = indices[:-2] , indices[1:] 
    ret = [ ]
    for x,y in zip(left, right):
        N = 0
        temp = np.zeros(y-x)
        for case in distance_time:
            if len(case[0]) >= y:
                temp += np.array(case[0][x:y])
                N += 1
        if N > 0:
            ret += (temp/N).tolist() 

    # try:
    T = np.array(list(map(lambda c : c[0][:20],\
        list(filter( lambda x : len(x[0]) > 20, distance_time)))))
    
    return np.mean(T, axis=0)

    return np.array(ret)


def centerofmass_time( particales_frames ):
    def calc_center_of_mass( particales_frame ):
        CMx, CMy = 0, 0
        mass =  0
        for particale in particales_frame:
            CMx, CMy = CMx + particale.CM[0] * len(particale.x) , CMy + particale.CM[1] * len(particale.y) 
            mass += len(particale.x)
        if mass != 0:
            return CMx / mass, CMy / mass 
        else:
            return 0,0 
    return list(map( calc_center_of_mass, particales_frames))

def shift_center_mass( particales_frames ):
    center_list = centerofmass_time( particales_frames )
    
    def shift_particale( particale, center ):
        for _ in range(2):
            particale.CM[_] -= center[_] 

    for center, frame in zip(center_list, particales_frames):
        for particale in frame:
            shift_particale( particale, center)
        
def filternoise(particales_frames):
    ret = [ ]
    for particales_frame in particales_frames:
        ret.append( \
            list(filter(lambda p: len(p.x) > 30, particales_frame )) )
    return ret

def reset( particales_frames):
    for particales_frame in particales_frames:
        for particale in particales_frame:
            particale.next = None
            particale.prev = None
        
        # reset_frame( particales_frame )

def test_read():
    _arr = [ ]
    for w, testcases in  enumerate(testcases_exp[1:]):
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
            reset(particales_frames)
            particales_frames = load(open("pickleout{0}.pkl".format(w) , "rb"))
            particales_frames = filternoise(particales_frames )
            shift_center_mass(particales_frames)

            for i in range(len(particales_frames)-2):
                find_matching( particales_frames[i], particales_frames[i+1] )
            
            
            # print(particales_frames[0][-1].CM)
            # particales_mass_list, massbins  = quantenize_mass( list(leaves_generator(particales_frames)))  
            
            # for massindex, mass in enumerate(massbins):
            
            reasonable = list(leaves_generator(particales_frames)) #list(map(reasonable_kernel, particales_frames[0]))
            reasonable = list(filter( lambda x : x != None, reasonable))
            # reasonable = list(filter( lambda p: len(p.x) > 5, reasonable))
            print("reasonable size = ", len(reasonable) )

            # reasonable = reduce_mean_mean( reasonable )  


            # distance_time 
            _temp_= list(map(calculate_time_distance, reasonable))
            # print(_temp_)
            # print(len(_temp_))

            distance_time = list(filter(lambda x: len(x[0]) > 10 ,  _temp_))

            print(len(distance_time))
            if len(distance_time) == 0:
                continue 


            # print(distance_time[0])

            # print(distance_time.shape)
            # if len(distance_time) < 9:
            #     continue

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
                axs[2, _].set_xlabel(r'time [ frames ]')
                axs[_, 0].set_ylabel(r'$ r^2 $ [px]')

            for i in range(3):
                for j in range(3):
                    if i*3 + j < len(distance_time):
                        axs[i, j].plot(distance_time[i*3 + j][0].copy())
            

            fig.savefig("./fig/9-{0}.png".format(testcases[0]))
            # plt.show()
            # plt.close()
            plt.clf()
            # fig  = plt.gcf()
            plt.plot( calculate_average(distance_time))


            plt.title(  r' $ E [ r^2 ] $ as function of time '  )
            plt.xlabel(r'time [ frames ]')
            plt.ylabel(r'$r$ [px]')
            plt.axis('equal')
            fig.savefig("./fig/E-{0}.png".format(testcases[0]))
            # plt.show()


if __name__ == "__main__":
    test_read()



