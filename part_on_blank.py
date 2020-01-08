import mrcfile as mrc
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from scipy.spatial import KDTree

parts = ['1837_0001.mrcs',
         '1837_0002.mrcs',
         '1837_0003.mrcs',
         '1837_0004.mrcs',
         '1837_0005.mrcs',
         '1837_0006.mrcs',
         '1837_0007.mrcs',
         '1837_0008.mrcs',
         '1837_0009.mrcs']

def put_parts(locs,canvas_n, canvas_b):

    data_check = True
    for groups in parts:
        with mrc.open(groups) as f:
            part_data = f.data

        if data_check:
            data_check = False
            data = part_data
        else:
            data = np.concatenate((data, part_data), axis=0)

    print(data.shape)
    print(f'mean of parts: {np.mean(data)}')
    print(f'std of parts: {np.std(data)}')
    for loc in locs:
        one_part = data[randint(0,data.shape[0]-1)][:][:]
        partx, party = one_part.shape
        canvas_n[loc[0]:loc[0]+partx, loc[1]:loc[1]+party] = one_part
        canvas_b[loc[0]:loc[0]+partx, loc[1]:loc[1]+party] = one_part
        canvas_n = np.float32(canvas_n)
        canvas_b = np.float32(canvas_b)
    return canvas_n, canvas_b

def make_blank():
    canvas = np.zeros((4256,4256))
    return canvas

def rand_points():
    N   = 90
    pts = 4000*np.random.random((N,2))
    pts = pts.astype(int)

    tree = KDTree(pts)
    too_close = tree.query_pairs(220)
    too_close = [i[0] for i in too_close]
    pts = np.delete(pts,too_close,0)
    print(f'Number of particles: {pts.shape[0]}')
    return(pts)
    
def write_coords(points,i):
    with open(f'{i}.star', 'w+') as f:
        f.write('''
data_

loop_
_rlnCoordinateX #1 
_rlnCoordinateY #2
_rlnClassNumber #3
_rlnAnglePsi #4
_rlnAutopickFigureOfMerit  #5
        ''')
        for point in points:
            f.write(f'\n{point[1]+128} {point[0]+128} {-999} {-999} {-999}')

'''
data_

loop_
_rlnCoordinateX #1 
_rlnCoordinateY #2
_rlnClassNumber #3
_rlnAnglePsi #4
_rlnAutopickFigureOfMerit  #5
2596.0  4034.0  -999    -999.0  -999.0
3981.5  3946.5  -999    -999.0  -999.0
4145.5  3798.0  -999    -999.0  -999.0
3343.5  3764.0  -999    -999.0  -999.0
606.0   3683.0  -999    -999.0  -999.0
1848.0  3626.0  -999    -999.0  -999.0
893.0   3575.0  -999    -999.0  -999.0
3245.0  3325.0  -999    -999.0  -
'''
    
def new_mrc(img_data, i):
    with mrc.new(f'{i}.mrc', overwrite=True) as f:
        f.set_data(img_data)

def add_noise(image):
    row,col= image.shape
    mean = 0
    var = 1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy
    
def add_marker(image):
    mask1 = np.ones((30,60))
    mask2 = np.ones((30,30))
    image[0:30,0:60] = mask1
    image[-31:-1,0:30] = mask2
    return image
    
if __name__ == '__main__':
    for i in range(100):
        locs = rand_points()
        write_coords(locs,i)
        blank_canv = make_blank()
        noisy_canv = add_noise(blank_canv)
        pop_conv_n, pop_conv_b = put_parts(locs,noisy_canv, blank_canv)
        pop_conv_n = add_marker(pop_conv_n)
        new_mrc(pop_conv_n, i)

        '''
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(pop_conv_b, origin='lower', cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(pop_conv_n, origin='lower', cmap='gray')
        plt.show()
        '''
    
