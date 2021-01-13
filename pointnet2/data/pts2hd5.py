import os
import sys
import numpy as np
import math
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

GRID_DIMENTION = [0.5, 0.5]
GRID_SAMPLE_NUM = 4096

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue] for x,y,z,red,green,blue in pc])
    a, b = np.shape(pc_array)
    print('point cloud dimention: %s x %s' % (a, b))
    return pc_array

def getGridPos(pt, biased_centered_min, grid_dim):
    x = pt[0]
    y = pt[1]
    rows = []
    cols = []
    N1 = grid_dim[0]
    N2 = grid_dim[1]

    row_num = math.floor((x - biased_centered_min[0]) / (GRID_DIMENTION[0] / 2))
    col_num = math.floor((y - biased_centered_min[1]) / (GRID_DIMENTION[1] / 2))

    if row_num >= N1:
        row_num = N1 - 1
    if col_num >= N2:
        col_num = N2 - 1

    rows.append(row_num)
    cols.append(col_num)

    if row_num > 0:
        rows.append(row_num - 1)
    if col_num > 0:
        cols.append(col_num - 1)
    return rows, cols

def getLocalOrigins(pc_array, centered_min, centered_max):
    biased_centered_min = [centered_min[0] - 0.01, centered_min[1] - 0.01, centered_min[2]]
    grid_dim = [0,0]
    grid_dim[0] = math.ceil((centered_max[0] - biased_centered_min[0]) / GRID_DIMENTION[0]) * 2 - 1
    grid_dim[1] = math.ceil((centered_max[1] - biased_centered_min[1]) / GRID_DIMENTION[1]) * 2 - 1
    N = grid_dim[0] * grid_dim[1] # group numbers

    pc_origins = np.tile(biased_centered_min, (N, 1))

    for idy in range(grid_dim[1]):
        for idx in range(grid_dim[0]):
            pc_origins[idy * grid_dim[0] + idx] = [biased_centered_min[0] + idx * GRID_DIMENTION[0] / 2, # not origin but corner
                                          biased_centered_min[1] + idy * GRID_DIMENTION[1] / 2, # not origin but corner
                                          0.0]

    # get the capacity of each grid
    val_res = np.zeros(N)
    for idx in range(len(pc_array)):
        rows, cols = getGridPos(pc_array[idx], biased_centered_min, grid_dim)

        for ida in range(len(rows)):
            for idb in range(len(cols)):
                # put the points inside
                val_res[cols[idb] * grid_dim[0] + rows[ida]] += 1

    # switch (val_res)
    # case val_res < GRID_SAMPLE_NUM
    #   get the points in the neighboring eight grids
    #   if neighbors and val_res more than GRID_SAMPLE_NUM
    #       random_shuffle only neighbors and make the 'val_res' enough
    #   if neiighbors and val_res less than GRID_SAMPLE_NUM
    #       random_shuffle all points, add to val_res, until more than GRID_SAMPLE_NUM, and dispose the rest
    # case val_res > GRID_SAMPLE_NUM
    #   random shuffle and devide into different groups
    #   definitely the last group less than GRID_SAMPLE_NUM (if equal then fine)
    #   random shuffle the all points in the original grid and make the 'val_res' enough

    print('results:')
    res = 0
    res_nonempty = 0
    res_half = 0
    for idx in range(len(val_res)):
        print(val_res[idx])
        if val_res[idx] > GRID_SAMPLE_NUM:
            res += 1
        if val_res[idx] > 0:
            res_nonempty += 1
        if val_res[idx] > GRID_SAMPLE_NUM * 0.5:
            res_half += 1

    print('qualified: %s %s %s %s' % (res, res_nonempty, res_half, len(val_res)))
    print('%.4f %.4f %.4f' % (res / len(val_res), res_nonempty / len(val_res), res_half  / len(val_res)))
    return pc_origins

def getNormalizedHD5s(pc_array, pc_origins):
    print('not finished')

    pc_array = np.c_[pc_array, pc_array[..., 0:3]]
    return

def center2origin(pc_array):# move center to zero origin
    #pc_cen = pc_array
    n = len(pc_array)
    sums = pc_array.sum(axis=0)    # axis = 0 get all sums in all columns
    center = [0, 0, 0]
    center[0] = sums[0] / n
    center[1] = sums[1] / n
    center[2] = sums[2] / n

    for idi in range(n):
        pc_array[idi, 0] -= center[0]
        pc_array[idi, 1] -= center[1]
        pc_array[idi, 2] -= center[2]
    centered_max = pc_array.max(axis=0)[0:3]
    centered_min = pc_array.min(axis=0)[0:3]
    return center, centered_max, centered_min

def convertHD5(pc_array):


    print('not finished')
    return

def main():
    filename = '/home/en1060/Desktop/classroom.ply'
    pc_array = read_ply(filename)
    center, centered_max, centered_min = center2origin(pc_array)
    getLocalOrigins(pc_array, centered_min, centered_max)
    getNormalizedHD5s(pc_array, any)
    print(0)


if __name__ == '__main__':
    main()