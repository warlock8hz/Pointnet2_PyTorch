import os
import sys
import numpy as np
import math
import random
import h5py
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

    # get origins IN THE FORM OF CORNER FOR FASTER CHECKING
    for idy in range(grid_dim[1]):
        for idx in range(grid_dim[0]):
            pc_origins[idy * grid_dim[0] + idx] = [biased_centered_min[0] + idx * GRID_DIMENTION[0] / 2,
                                          biased_centered_min[1] + idy * GRID_DIMENTION[1] / 2, 0.0]

    # get the capacity of each grid
    pt2grid = []    # point in which grids
    ptIdInGrid = [[]] * N   # points in the corresponding grid

    # val_res = np.zeros(N)
    for idx in range(len(pc_array)):
        rows, cols = getGridPos(pc_array[idx], biased_centered_min, grid_dim)
        pt2grid.append([rows, cols])

        for ida in range(len(rows)):
            for idb in range(len(cols)):
                idc = cols[idb] * grid_dim[0] + rows[ida]
                # val_res[idc] += 1
                ptIdInGrid[idc] = ptIdInGrid[idc] + [idx]

    # from corner points to origins in the grids
    for ida in range(len(pc_origins)):
        pc_origins[ida][0] += GRID_DIMENTION[0] / 2
        pc_origins[ida][1] += GRID_DIMENTION[0] / 2

    return pc_origins, ptIdInGrid, grid_dim

def getPtsFromId(pc_array, ptIds):
    ptSelected = np.zeros((GRID_SAMPLE_NUM, 9), dtype=np.float32)
    for ida in range(len(ptIds)):
        ptSelected[ida] = pc_array[ptIds[ida]]
    #ptSelected[ida] = np.float32(ptSelected)
    return ptSelected

def getPtIdsIn8Neighbors(ptIdInGrid, centerGridId, grid_dim):
    ptIdsInNeighbor = []
    localSearchCtr = centerGridId

    # same row
    idx = localSearchCtr % grid_dim[0]
    idy = math.ceil(localSearchCtr / grid_dim[0])
    if idx > 0:
        ptIdsInNeighbor = ptIdInGrid[localSearchCtr - 1].copy()
    if idx < grid_dim[0] - 1:
        ptIdsInNeighbor = ptIdsInNeighbor + ptIdInGrid[localSearchCtr + 1]
    # previous row
    if idy > 0:
        localSearchCtr = centerGridId - grid_dim[0]
        if idx > 0:
           ptIdsInNeighbor = ptIdsInNeighbor + ptIdInGrid[localSearchCtr - 1]
        if idx < grid_dim[0] - 1:
            ptIdsInNeighbor = ptIdsInNeighbor + ptIdInGrid[localSearchCtr + 1]
    # next row
    if idy < grid_dim[1] - 1:
        localSearchCtr = centerGridId + grid_dim[0]
        if idx > 0:
            ptIdsInNeighbor = ptIdsInNeighbor + ptIdInGrid[localSearchCtr - 1]
        if idx < grid_dim[0] - 1:
            ptIdsInNeighbor = ptIdsInNeighbor + ptIdInGrid[localSearchCtr + 1]
    return ptIdsInNeighbor

def getNormalized(a_data, pc_origin):
    for ida in range(len(a_data)):
        a_data[ida][0] -= pc_origin[0]
        a_data[ida][1] -= pc_origin[1]
        a_data[ida][2] -= pc_origin[2]
    return

def getNormalizedHD5s(pc_array, pc_origins, ptIdInGrid, grid_dim):
    pc_localarray = pc_array.copy()
    pc_localarray = np.c_[pc_localarray, pc_localarray[..., 0:3]]

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

    totalLayers = 0
    for ida in range(len(ptIdInGrid)):
        totalLayers += math.ceil(len(ptIdInGrid[ida]) / GRID_SAMPLE_NUM)
    print('%s x %s in HDF5' % (totalLayers, GRID_SAMPLE_NUM))

    a_data = np.zeros((totalLayers, GRID_SAMPLE_NUM, 9), dtype=np.float32)
    a_pid = np.zeros((totalLayers, GRID_SAMPLE_NUM), dtype=np.uint8)

    # a_data_tempInLoop = np.zeros((GRID_SAMPLE_NUM, 9), dtype=np.float32)
    ids_InLoop = []
    num_layer = 0
    for ida in range(len(ptIdInGrid)):
        num_layer_in_grid = 0
        random.shuffle(ptIdInGrid[ida])
        if len(ptIdInGrid[ida]) == 0:
            continue
        elif len(ptIdInGrid[ida]) >= GRID_SAMPLE_NUM:
            num_processed = 0
            num_processed_InLoop = 0
            num_processed_max = len(ptIdInGrid[ida])
            while num_processed_InLoop < GRID_SAMPLE_NUM:
                ids_InLoop.append(ptIdInGrid[ida][num_processed])
                num_processed += 1
                num_processed_InLoop += 1
                if num_processed_InLoop == GRID_SAMPLE_NUM:
                    a_data[num_layer] = getPtsFromId(pc_localarray, ids_InLoop)
                    getNormalized(a_data[num_layer], pc_origins[ida])
                    num_layer_in_grid += 1
                    num_layer += 1
                    num_processed_InLoop = 0
                    ids_InLoop = []
                if num_processed == num_processed_max:
                    break
            if num_processed_InLoop > 0 and num_processed_InLoop < GRID_SAMPLE_NUM:
                ptIdSubset = ptIdInGrid[ida][0:num_layer_in_grid * GRID_SAMPLE_NUM]
                random.shuffle(ptIdSubset)
                ids_InLoop = ptIdSubset[0:GRID_SAMPLE_NUM - num_processed_InLoop] \
                             + ptIdInGrid[ida][num_layer_in_grid * GRID_SAMPLE_NUM:num_processed_max]
                a_data[num_layer] = getPtsFromId(pc_localarray, ids_InLoop)
                getNormalized(a_data[num_layer], pc_origins[ida])
                num_layer += 1
                num_processed_InLoop = 0
                ids_InLoop = []
                ptIdSubset = []
        else:
            ptIdInNeighbor = getPtIdsIn8Neighbors(ptIdInGrid, ida, grid_dim)
            random.shuffle(ptIdInNeighbor)
            ids_InLoop = ptIdInGrid[ida] + ptIdInNeighbor[0:GRID_SAMPLE_NUM - len(ptIdInGrid[ida])]
            if len(ids_InLoop) == 0:
                continue
            if len(ids_InLoop) < GRID_SAMPLE_NUM:
                local_duplicate = ids_InLoop.copy()
                while len(ids_InLoop) + len(local_duplicate) < GRID_SAMPLE_NUM:
                    ids_InLoop = ids_InLoop + local_duplicate
                random.shuffle(local_duplicate)
                ids_InLoop = ids_InLoop + local_duplicate[0:GRID_SAMPLE_NUM - len(ids_InLoop)]
            a_data[num_layer] = getPtsFromId(pc_localarray, ids_InLoop)
            getNormalized(a_data[num_layer], pc_origins[ida])
            num_layer += 1
            ids_InLoop = []
            ptIdInNeighbor = []

    return a_data, a_pid

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

def ply2hdf5(filename, hdf5_filename):
    #filename = '/home/en1060/Desktop/classroom-ss.ply'
    pc_array = read_ply(filename)
    center, centered_max, centered_min = center2origin(pc_array)
    pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_array, centered_min, centered_max)
    a_data, a_pid = getNormalizedHD5s(pc_array, pc_origins, ptIdInGrid, grid_dim)

    #f = h5py.File("/home/en1060/Desktop/classroom-ss.h5", 'w')
    f = h5py.File(hdf5_filename, 'w')
    data = f.create_dataset("data", data=a_data)
    pid = f.create_dataset("label", data=a_pid)

    return

def main():
    filename = '/home/en1060/Desktop/classroom-ss.ply'
    pc_array = read_ply(filename)
    center, centered_max, centered_min = center2origin(pc_array)
    pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_array, centered_min, centered_max)
    a_data, a_pid = getNormalizedHD5s(pc_array, pc_origins, ptIdInGrid, grid_dim)

    f = h5py.File("/home/en1060/Desktop/classroom-ss.h5", 'w')
    #f = h5py.File(hdf5_filename, 'w')
    data = f.create_dataset("data", data=a_data)
    pid = f.create_dataset("label", data=a_pid)

    return


if __name__ == '__main__':
    main()