import os
import sys
sys.path.append(".")

import numpy as np
import math
import random
import h5py
import codecs

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

GRID_DIMENTION = [1.0, 1.0]
GRID_SAMPLE_NUM = 4096

def returnIndoor3DLabels(label):
    # if label == 0:
    #     return 'ceiling'
    # elif label == 1:
    #     return 'floor'
    # elif label == 2:
    #     return 'wall'
    # elif label == 3:
    #     return 'column'
    # elif label == 4:
    #     return 'beam'
    # elif label == 5:
    #     return 'window'
    # elif label == 6:
    #     return 'door'
    # elif label == 7:
    #     return 'table'
    # elif label == 8:
    #     return 'chair'
    # elif label == 9:
    #     return 'bookcase'
    # elif label == 10:
    #     return 'sofa'
    # elif label == 11:
    #     return 'board'
    # elif label == 12:
    #     return 'clutter'
    if label == 0:
        return 'pipe'
    elif label == 1:
        return 'bend-box'
    elif label == 2:
        return 'valve'
    elif label == 3:
        return '90d-elbow'
    elif label == 4:
        return '22.5d-elbow'
    elif label == 5:
        return '45d-elbow'
    elif label == 6:
        return 'tee'
    elif label == 7:
        return 'fence'
    elif label == 8:
        return 'ladder'
    elif label == 9:
        return 'ground'
    elif label == 10:
        return 'framework'
    elif label == 11:
        return 'barrier'
    else: # should not happen
        return 'unknownError'
    return 'unknownError' # if error

def returnIndoor3DLabelIds(strName):
    # if strName.lower() == 'ceiling':
    #     return 0
    # elif strName.lower() == 'floor':
    #     return 1
    # elif strName.lower() == 'wall':
    #     return 2
    # elif strName.lower() == 'column':
    #     return 3
    # elif strName.lower() == 'beam':
    #     return 4
    # elif strName.lower() == 'window':
    #     return 5
    # elif strName.lower() == 'door':
    #     return 6
    # elif strName.lower() == 'table':
    #     return 7
    # elif strName.lower() == 'chair':
    #     return 8
    # elif strName.lower() == 'bookcase':
    #     return 9
    # elif strName.lower() == 'sofa':
    #     return 10
    # elif strName.lower() == 'board':
    #     return 11
    # elif strName.lower() == 'clutter':
    #     return 12
    if strName.lower() == 'pipe':
        return 0
    elif strName.lower() == 'bend-box':
        return 1
    elif strName.lower() == 'valve':
        return 2
    elif strName.lower() == '90d-elbow':
        return 3
    elif strName.lower() == '22.5d-elbow':
        return 4
    elif strName.lower() == '45d-elbow':
        return 5
    elif strName.lower() == 'tee':
        return 6
    elif strName.lower() == 'fence':
        return 7
    elif strName.lower() == 'ladder':
        return 8
    elif strName.lower() == 'ground':
        return 9
    elif strName.lower() == 'framework':
        return 10
    elif strName.lower() == 'barrier':
        return 11
    else: # should not happen
        return 255
    return 255#should be error


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
    biased_centered_min = np.float32(biased_centered_min)
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
    ptIdInGrid = [[-1] * len(pc_array)] * N   # points in the corresponding grid
    ptNumInGrid = [0] * N

    # val_res = np.zeros(N)
    # get the statistics
    print('first pass of grouping %s points for memory allocation' % len(pc_array))
    for idx in range(len(pc_array)):
        if idx % 500000 == 0:
            print('%s of %s finished'%(idx, len(pc_array)))
        rows, cols = getGridPos(pc_array[idx], biased_centered_min, grid_dim)
        pt2grid.append([rows, cols])

        for ida in range(len(rows)):
            for idb in range(len(cols)):
                idc = cols[idb] * grid_dim[0] + rows[ida]
                # val_res[idc] += 1
                #ptIdInGrid[idc] = ptIdInGrid[idc] + [idx]
                #ptIdInGrid[idc][ptNumInGrid[idc]] = idx
                ptNumInGrid[idc] += 1
    print('%s of %s finished'%(len(pc_array), len(pc_array)))

    # get the result list of lists prepared
    print('memory allocation...')
    for idb in range(grid_dim[1]):
        for ida in range(grid_dim[0]):
            idc = idb * grid_dim[0] + ida
            ptIdInGrid[idc] = [0] * ptNumInGrid[idc]

    # save to the list of lists
    print('second pass: grouping %s points' % len(pc_array))
    ptNumInGrid2save = [0] * N
    for idx in range(len(pc_array)):
        if idx % 500000 == 0:
            print('%s of %s finished'%(idx, len(pc_array)))
        rows = pt2grid[idx][0].copy() # retrieve the pos
        cols = pt2grid[idx][1].copy() # retrieve the pos

        for ida in range(len(rows)):
            for idb in range(len(cols)):
                idc = cols[idb] * grid_dim[0] + rows[ida]
                #ptIdInGrid[idc] = ptIdInGrid[idc] + [idx]
                ptIdInGrid[idc][ptNumInGrid2save[idc]] = idx
                ptNumInGrid2save[idc] += 1
    print('%s of %s finished'%(len(pc_array), len(pc_array)))

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
def getLabelsFromId(label_list, ptIds):
    labelSelected = []
    for ida in range(len(ptIds)):
        labelSelected.append(label_list[ptIds[ida]])
    return labelSelected

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

def getNormalizedVal(rawVal, range, minVal):
    normalizedVal = (rawVal - minVal)/range
    return normalizedVal

def getNormalized(a_data, pc_origin):
    for ida in range(len(a_data)):
        a_data[ida][0] -= pc_origin[0]
        a_data[ida][1] -= pc_origin[1]
        a_data[ida][2] -= pc_origin[2]
        a_data[ida][3] = a_data[ida][3] / 255
        a_data[ida][4] = a_data[ida][4] / 255
        a_data[ida][5] = a_data[ida][5] / 255

    # get normalized
    centered_max = a_data.max(axis=0)[0:3]
    #centered_max = np.float32(centered_max)
    centered_min = a_data.min(axis=0)[0:3]
    centered_range = [centered_max[0] - centered_min[0],
                      centered_max[1] - centered_min[1],
                      centered_max[2] - centered_min[2]]

    for ida in range(len(a_data)):
        a_data[ida][6] = getNormalizedVal(a_data[ida][0], centered_range[0], centered_min[0])
        a_data[ida][7] = getNormalizedVal(a_data[ida][1], centered_range[1], centered_min[1])
        a_data[ida][8] = getNormalizedVal(a_data[ida][2], centered_range[2], centered_min[2])

    return

def getNormalizedHD5s(pc_array, label_list, pc_origins, ptIdInGrid, grid_dim):
    # uint8 label can only be generate after all classes and hdf5 databases generated
    D2label_list = [] # will change from 1D to 2D
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
    a_origins = np.zeros((totalLayers, 3), dtype=np.float32)

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
                    D2label_list.append(getLabelsFromId(label_list, ids_InLoop))
                    getNormalized(a_data[num_layer], pc_origins[ida])
                    a_origins[num_layer] = pc_origins[ida]
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
                D2label_list.append(getLabelsFromId(label_list, ids_InLoop))
                getNormalized(a_data[num_layer], pc_origins[ida])
                a_origins[num_layer] = pc_origins[ida]
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
            D2label_list.append(getLabelsFromId(label_list, ids_InLoop))
            getNormalized(a_data[num_layer], pc_origins[ida])
            a_origins[num_layer] = pc_origins[ida]
            num_layer += 1
            ids_InLoop = []
            ptIdInNeighbor = []

    return a_data, a_pid, a_origins, D2label_list

def getNormalizedDSHD5s(pc_array, label_list, pc_origins, ptIdInGrid, grid_dim):
    # uint8 label can only be generate after all classes and hdf5 databases generated
    D2label_list = [] # will change from 1D to 2D
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
        if len(ptIdInGrid[ida]) > 0:
            totalLayers += 1
        #totalLayers += math.ceil(len(ptIdInGrid[ida]) / GRID_SAMPLE_NUM)
    print('%s x %s in HDF5' % (totalLayers, GRID_SAMPLE_NUM))

    a_data = np.zeros((totalLayers, GRID_SAMPLE_NUM, 9), dtype=np.float32)
    a_pid = np.zeros((totalLayers, GRID_SAMPLE_NUM), dtype=np.uint8)
    a_origins = np.zeros((totalLayers, 3), dtype=np.float32)

    # a_data_tempInLoop = np.zeros((GRID_SAMPLE_NUM, 9), dtype=np.float32)
    ids_InLoop = []
    num_layer = 0
    for ida in range(len(ptIdInGrid)):
        num_layer_in_grid = 0
        random.shuffle(ptIdInGrid[ida])
        if len(ptIdInGrid[ida]) == 0:
            continue
        elif len(ptIdInGrid[ida]) >= GRID_SAMPLE_NUM:
            ids_InLoop = ptIdInGrid[ida][0:GRID_SAMPLE_NUM]
            a_data[num_layer] = getPtsFromId(pc_localarray, ids_InLoop)
            D2label_list.append(getLabelsFromId(label_list, ids_InLoop))
            getNormalized(a_data[num_layer], pc_origins[ida])
            a_origins[num_layer] = pc_origins[ida]
            num_layer += 1
            ids_InLoop = []
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
            D2label_list.append(getLabelsFromId(label_list, ids_InLoop))
            getNormalized(a_data[num_layer], pc_origins[ida])
            a_origins[num_layer] = pc_origins[ida]
            num_layer += 1
            ids_InLoop = []
            ptIdInNeighbor = []

    return a_data, a_pid, a_origins, D2label_list

def addBias2pts(data, origins):
    for idx in range(len(data)):
        data[idx][0] += origins[0]
        data[idx][1] += origins[1]
        data[idx][2] += origins[2]
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

def ply2hdf5data(filename):#ply file only has no label information
    #filename = '/home/en1060/Desktop/classroom-ss.ply'
    pc_array = read_ply(filename)
    center, centered_max, centered_min = center2origin(pc_array)
    pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_array, centered_min, centered_max)
    label_list = [""]*len(pc_array)
    a_data, a_label, a_origins, D2label_list = getNormalizedHD5s(pc_array, label_list, pc_origins, ptIdInGrid, grid_dim)

    return a_data, a_label, a_origins, center
def ply2hdf5file(filename):#ply file only has no label information
    #filename = '/home/en1060/Desktop/classroom-ss.ply'
    pc_array = read_ply(filename)
    center, centered_max, centered_min = center2origin(pc_array)
    pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_array, centered_min, centered_max)
    label_list = [""]*len(pc_array)
    a_data, a_label, a_origins, D2label_list = getNormalizedHD5s(pc_array, label_list, pc_origins, ptIdInGrid, grid_dim)

    f = h5py.File(filename + ".h5", 'w')
    #f = h5py.File(hdf5_filename, 'w')
    data = f.create_dataset("data", data=a_data)
    pid = f.create_dataset("label", data=a_label)
    return #a_data, a_label, a_origins

def name2class(name):
    # no dash at all
    pos = name.find('_', 0, len(name))
    if pos == -1:
        pos = name.find('.asc', 0, len(name))
        className = name[0:pos]
    # only one dash allowed
    else:
        className = name[0:pos]
    return className
def scanAscFile(path, fList, nameList):
    # List all files in a directory using scandir()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.asc'):
                fList.append(os.path.join(path, entry.name))
                nameList.append(name2class(entry.name))
    return
def folderPly2HDF5(path):
    fList = []
    nameList = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            scanAscFile(os.path.join(path, entry), fList, nameList)
    return
def scanFolderInFolder(path):
    folderList = []
    sceneName = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            folderList.append(os.path.join(path, entry))
            sceneName.append(entry)
    return folderList, sceneName
def checkAddNewCat(nameList, catList):
    for name in nameList:
        res = -1
        for idx, cat in enumerate(catList):
            if name.lower() == cat.lower():
                res = idx
        if res < 0:
            catList.append(name)
    return
def checkNameInCatList(name, catList):
    res = -1
    for idx, cat in enumerate(catList):
        if name.lower() == cat.lower():
            res = idx
    if res < 0:
        res = 255
    return res
def initialNassignLabelVal2HD5labels(label_vals, catList, all_labelstrlist, all_labelvallist):
    for id in range(len(all_labelvallist)):
        for idx in range(len(all_labelvallist[id])):
            for idy in range(len(all_labelvallist[id][idx])):
                # pending label value: all_labelvallist[id][idx][idy] vs label string all_labelstrlist[id][idx][idy]
                res = checkNameInCatList(all_labelstrlist[id][idx][idy], catList)
                all_labelvallist[id][idx][idy] = label_vals[res]
    return
def assignLabelVal2HD5labels(all_labelstrlist, all_labelvallist):
    for id in range(len(all_labelvallist)):
        for idx in range(len(all_labelvallist[id])):
            for idy in range(len(all_labelvallist[id][idx])):
                res = returnIndoor3DLabelIds(all_labelstrlist[id][idx][idy])
                all_labelvallist[id][idx][idy] = res    #check if uint8
    return
def hd5prediction2txt(data, label, origins, filename, center):
    pt_data = data[:, :, 0:3]  # convert to three-column shape for open3D

    with open(filename + '.txt', 'a') as the_file:
        the_file.write('Hello\n')
        pt_data = np.float64(pt_data)
        for i in range(len(pt_data)):
            addBias2pts(pt_data[i], origins[i])
            #pt_data[i] = np.float64(pt_data[i]) # prepare to recover the coordinates
            addBias2pts(pt_data[i], center)
            for j in range(len(pt_data[i])):
                the_file.write('%.5f, %.5f, %.5f, %s, ' % (pt_data[i][j][0], pt_data[i][j][1],
                                                         pt_data[i][j][2], label[i][j]))
                the_file.write(returnIndoor3DLabels(label[i][j]))
                the_file.write('\n')
    return
def asc2hdf5data_prediction(path):  # single asc file to single hdf5 data
    D2labelVal_list = []
    pc_list = np.zeros((0, 6), dtype=np.float64)
    label_list = []
    fList = []
    nameList = []
    scanAscFile(path, fList, nameList) # nameList is useless in prediction
    nameList = ['unknown'] * len(fList)
    for ida, file in enumerate(fList):
        with codecs.open(file, encoding='utf-8-sig') as f:
            pcLocalList = np.loadtxt(f)
            pcLocalList = np.delete(pcLocalList, 3, 1)
        nameLocalList = [nameList[ida]] * len(pcLocalList)
        pc_list = np.concatenate((pc_list, pcLocalList), axis=0)
        label_list = label_list + nameLocalList
    print('%s points added'%len(pc_list))
    # pc and labels to hd5
    center, centered_max, centered_min = center2origin(pc_list) # centerlized, need to be recovered in the future
    pc_list = np.float32(pc_list)
    pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_list, centered_min, centered_max)
    # label_list from 1D to 2D, a_label reserved the space
    a_data, a_label, a_origins, D2label_list = \
        getNormalizedHD5s(pc_list, label_list, pc_origins, ptIdInGrid, grid_dim) # string list cannot be mutable

    D2label_list = [D2label_list]
    D2labelVal_list = [a_label]
    assignLabelVal2HD5labels(D2label_list, D2labelVal_list)
    D2labelVal_list = D2labelVal_list[0]

    return a_data, D2labelVal_list, a_origins, center
def asc2hdf5dataSingleScene(path):
    #path = "/home/en1060/Desktop/importh5/utility-test"
    folders, sceneNames = scanFolderInFolder(path)
    all_filelist = []
    all_pclist = []
    all_labelvallist = []
    all_labelstrlist = []
    all_originlist = []
    all_gridoriginlist = []
    all_centers = []
    room_filelist = []
    catList = []
    for idx, folder in enumerate(folders):
        all_filelist.append(sceneNames[idx])
        pc_list = np.zeros((0, 6), dtype=np.float64)
        label_list = []
        fList = []
        nameList = []
        scanAscFile(folder, fList, nameList)
        for ida, file in enumerate(fList):
            with codecs.open(file, encoding='utf-8-sig') as f:
                pcLocalList = np.loadtxt(f)
            nameLocalList = [nameList[ida]] * len(pcLocalList)
            pc_list = np.concatenate((pc_list, pcLocalList), axis=0)
            label_list = label_list + nameLocalList
        # pc and labels to hd5
        center, centered_max, centered_min = center2origin(pc_list) # centerlized, need to be recovered in the future
        pc_list = np.float32(pc_list)
        pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_list, centered_min, centered_max)
        # label_list from 1D to 2D, a_label reserved the space
        a_data, a_label, a_origins, D2label_list = \
            getNormalizedHD5s(pc_list, label_list, pc_origins, ptIdInGrid, grid_dim) # string list cannot be mutable
        for idy in range(len(a_data)):
            room_filelist.append('Area_' + str(idx) + '_site_' + str(idx))
            with open(os.path.join(path, "room_filelist.txt"), "a") as text_file:
                print(room_filelist[len(room_filelist) - 1], file=text_file)
        with open(os.path.join(path, "all_files.txt"), "a") as text_file:
            print(all_filelist[len(all_filelist) - 1] + ".h5", file=text_file)

        checkAddNewCat(nameList, catList)

        all_pclist.append(a_data)
        all_labelvallist.append(a_label)
        all_labelstrlist.append(D2label_list)
        all_originlist.append(pc_origins)
        all_gridoriginlist.append(a_origins)
        all_centers.append(center)

    # assign uint8 value to labels
    print("labels are assigned after indoor S3D labels (start from 14)")
    if False: # only used when dont know if any new labels
        num_known_cats = 13
        uint8_label = np.zeros(len(catList), dtype=np.uint8)
        for idx in range(len(uint8_label)):
            uint8_label[idx] = 13 + idx
        initialNassignLabelVal2HD5labels(uint8_label, catList, all_labelstrlist, all_labelvallist)
    else:
        assignLabelVal2HD5labels(all_labelstrlist, all_labelvallist)

    for idx in range(len(all_pclist)):
        #f = h5py.File(os.path.join(path, sceneNames[idx] + ".h5"), 'w')
        #data = f.create_dataset("data", data=all_pclist[idx])
        #pid = f.create_dataset("label", data=all_labelvallist[idx])

        # check res here
        hd5prediction2txt(all_pclist[idx], all_labelvallist[idx], all_gridoriginlist[idx],
                          os.path.join(path, sceneNames[idx] + ".rawLabel.txt"), all_centers[idx])

    return all_pclist, all_labelvallist, all_gridoriginlist, all_centers, all_filelist
def asc2hdf5file(path):
    #path = "/home/en1060/Desktop/importh5/utility"
    folders, sceneNames = scanFolderInFolder(path)
    all_filelist = []
    all_pclist = []
    all_labelvallist = []
    all_labelstrlist = []
    all_originlist = []
    all_gridoriginlist = []
    all_centers = []
    room_filelist = []
    catList = []
    for idx, folder in enumerate(folders):
        all_filelist.append(sceneNames[idx])
        pc_list = np.zeros((0, 6), dtype=np.float64)
        label_list = []
        fList = []
        nameList = []
        scanAscFile(folder, fList, nameList)
        for ida, file in enumerate(fList):
            with codecs.open(file, encoding='utf-8-sig') as f:
                pcLocalList = np.loadtxt(f)
            nameLocalList = [nameList[ida]] * len(pcLocalList)
            pc_list = np.concatenate((pc_list, pcLocalList), axis=0)
            label_list = label_list + nameLocalList
        # pc and labels to hd5
        center, centered_max, centered_min = center2origin(pc_list) # centerlized, need to be recovered in the future
        pc_list = np.float32(pc_list)
        pc_origins, ptIdInGrid, grid_dim = getLocalOrigins(pc_list, centered_min, centered_max)
        # label_list from 1D to 2D, a_label reserved the space
        a_data, a_label, a_origins, D2label_list = \
            getNormalizedHD5s(pc_list, label_list, pc_origins, ptIdInGrid, grid_dim) # string list cannot be mutable
        for idy in range(len(a_data)):
            room_filelist.append('Area_' + str(idx) + '_site_' + str(idx))
            with open(os.path.join(path, "room_filelist.txt"), "a") as text_file:
                print(room_filelist[len(room_filelist) - 1], file=text_file)
        with open(os.path.join(path, "all_files.txt"), "a") as text_file:
            print(all_filelist[len(all_filelist) - 1] + ".h5", file=text_file)

        checkAddNewCat(nameList, catList)

        all_pclist.append(a_data)
        all_labelvallist.append(a_label)
        all_labelstrlist.append(D2label_list)
        all_originlist.append(pc_origins)
        all_gridoriginlist.append(a_origins)
        all_centers.append(center)

    # assign uint8 value to labels
    print("labels are assigned after indoor S3D labels (start from 14)")
    if False: # only used when dont know if any new labels
        num_known_cats = 13
        uint8_label = np.zeros(len(catList), dtype=np.uint8)
        for idx in range(len(uint8_label)):
            uint8_label[idx] = 13 + idx
        initialNassignLabelVal2HD5labels(uint8_label, catList, all_labelstrlist, all_labelvallist)
    else:
        assignLabelVal2HD5labels(all_labelstrlist, all_labelvallist)

    for idx in range(len(all_pclist)):
        f = h5py.File(os.path.join(path, sceneNames[idx] + ".h5"), 'w')
        # f = h5py.File(hdf5_filename, 'w')
        data = f.create_dataset("data", data=all_pclist[idx])
        pid = f.create_dataset("label", data=all_labelvallist[idx])

        # check res here
        if True:#False:
            hd5prediction2txt(all_pclist[idx], all_labelvallist[idx], all_gridoriginlist[idx],
                              os.path.join(path, sceneNames[idx] + ".txt"), all_centers[idx])

    return
def main():
    path = '/home/en1060/Desktop/importh5/utility_raw_resolution'
    asc2hdf5file(path)
    return

if __name__ == '__main__':
    main()