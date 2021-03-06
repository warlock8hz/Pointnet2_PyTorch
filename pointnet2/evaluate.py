# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# the file is modified based on demo.py in https://github.com/facebookresearch/votenet.git

""" Demo of using pointnet++
"""

import os
import sys
sys.path.append(".")
import numpy as np
import argparse
import importlib
import time
from data import Indoor3DSemSegLoader as i3ssLoader
import math
import h5py
from data import pts2hd5

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', default='cls', help='Dataset: cls for classification or seg for segmentation [default: cls]')
#parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
#FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

N_GRAM = 256 #196 for 6 GB GRAM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/en1060/Projects/Pointnet2_PyTorch'
CKPT_DIR = os.path.join(ROOT_DIR, 'CKPT')
H5_DIR = os.path.join(BASE_DIR, 'data/indoor3d_sem_seg_hdf5_data')
PC_DIR = os.path.join(BASE_DIR, 'data/modelnet40_normal_resampled')
PC_SCALED_VALVE_DIR = os.path.join(BASE_DIR, 'data/scaled_valve')
pc_path = os.path.join(BASE_DIR, 'data/modelnet40_normal_resampled/', 'xbox/xbox_0013.txt')
cls_name_path = os.path.join(PC_DIR, 'modelnet40_shape_names.txt')
#sys.path.append(os.path.join(ROOT_DIR, 'utils'))
#sys.path.append(os.path.join(ROOT_DIR, 'models'))

import hydra
import omegaconf
#import argparse

#from pointnet2.data import Indoor3DSemSeg
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG # for validation functions
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG

def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)

#################################
# pcd io
#################################
def read_pcd_txt(filename):
    pc_array = np.genfromtxt(filename,delimiter=',')
    return pc_array

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def preprocess_point_cloud(point_cloud, num_point):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    #point_cloud = point_cloud[:,0:3] # do not use color for now --> same as training data
    #floor_height = np.percentile(point_cloud[:,2],0.99)
    #height = point_cloud[:,2] - floor_height
    #point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7) --> for votenet
    point_cloud = random_sampling(point_cloud, num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def scanFile(path, fList):
    # List all files in a directory using scandir()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                fList.append(os.path.join(path, entry.name))

def scanH5File(path, fList):
    # List all files in a directory using scandir()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.h5'):
                fList.append(os.path.join(path, entry.name))

def scanFolder(path):
    fList = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            scanFile(os.path.join(path, entry), fList)
    return fList

@hydra.main("config/config.yaml")
#@pytest.mark.parametrize("use_xyz", ["True", "False"])
#@pytest.mark.parametrize("model", ["ssg", "msg"])
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))
    #model = pytest.helpers.get_model(
    #    ["task=cls", f"model={model}", f"model.use_xyz={use_xyz}"]
    #)

    device = torch.device("cuda:0" if torch.cuda.is_available() else sys.exit(-1))

    checkpoint_folder = 'cls-ssg' # default cls
    checkpoint_path = os.path.join(CKPT_DIR, checkpoint_folder, 'best39_valve.ckpt')#'best39_valve_no_scale.ckpt')
    fList = scanFolder(PC_SCALED_VALVE_DIR)#(PC_DIR)
    fList.sort()
    B = 1

    if cfg.task_model.name == 'sem-ssg':
        checkpoint_folder = 'sem-ssg'
        checkpoint_path = os.path.join(CKPT_DIR, checkpoint_folder, 'utility_DS_epoch=46-val_loss=2.52-val_acc=0.616.ckpt')#'eutility_poch=34-val_loss=0.78-val_acc=0.859.ckpt')#'best869.ckpt')#'best39_valve_no_scale.ckpt')
        fList = []
        #scanH5File(H5_DIR, fList)
        scanH5File('/home/en1060/Desktop/importh5', fList)
        fList.sort()

    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
    model.eval()  # set model to eval mode (for bn and dp)

    model.cuda()  # move to cuda to work
    total = len(fList)
    correct = 0
    sequenceID = 0

    if cfg.task_model.name == 'sem-ssg':
        if False: # regular check for hdf5 scenes
            #for item in fList:
            item = "/home/en1060/Desktop/importh5/_1-PTS_CHD0130.5-0130.6.ply"
            print(sequenceID)
            #pts, labels = i3ssLoader._load_data_file(item)
            #pts, labels, origins, center = pts2hd5.ply2hdf5data(item)
            #pts2hd5.asc2hdf5dataSingleScene(item)
            pts, labels, origins, center = pts2hd5.ply2hdf5data(item)
            group_GRAM = math.floor(len(pts) / N_GRAM)

            for i_GRAM in range(group_GRAM + 1):
                correct = 0
                print('\t%d.%d' % (sequenceID, i_GRAM))

                N_start = i_GRAM * N_GRAM
                N_end = (i_GRAM + 1) * N_GRAM
                if i_GRAM == group_GRAM:
                    N_end = len(pts)

                pts_GRAM = pts[N_start:N_end, ...]
                labels_GRAM = labels[N_start:N_end, ...]
                #data_batches = np.concatenate(pts, 0) # 3D to 2D
                #labels_batches = np.concatenate(labels, 0) # 3D to 2D

                # Model inference
                inputs = {'point_clouds': torch.from_numpy(pts_GRAM).to(device)}
                tic = time.time()
                with torch.no_grad():
                    optimizer.zero_grad()
                    predicted_labels = torch.from_numpy(np.random.randint(10, 11, size=(len(pts_GRAM), len(pts_GRAM[0])))).to(device)
                    res = model.validate_once((inputs['point_clouds'], predicted_labels), None)
                    predicted_labels = res['log']['label']

                toc = time.time()
                pl = predicted_labels.cpu().detach().numpy()

                for idx in range(len(pl)):
                    for idy in range(len(pl[idx])):
                        if pl[idx][idy] == labels_GRAM[idx][idy]:
                            correct += 1

                total = pts_GRAM.shape[0] * pts_GRAM.shape[1]

                print(correct / total)

                labels[N_start:N_end, ...] = pl

            # f = h5py.File("/home/en1060/Desktop/importh5/classroom_predict.h5", 'w')
            # f = h5py.File(item + 'predict.h5', 'w')
            # data = f.create_dataset("data", data=pts)
            # pid = f.create_dataset("label", data=labels)
            pts2hd5.hd5prediction2txt(pts, labels, origins, item, center)

            sequenceID += 1
        elif False: # scene/folder-based check
            path = '/home/en1060/Desktop/importh5'#/utility-test'
            all_pclist, all_labelvallist, all_gridoriginlist, all_centers, all_filelist = \
                pts2hd5.asc2hdf5dataSingleScene(path)
            for id_file in range(len(all_pclist)):
                print(sequenceID)
                # pts, labels = i3ssLoader._load_data_file(item)
                # pts, labels, origins, center = pts2hd5.ply2hdf5data(item)
                group_GRAM = math.floor(len(all_pclist[id_file]) / N_GRAM)
                labels = all_labelvallist[id_file].copy()

                for i_GRAM in range(group_GRAM + 1):
                    correct = 0
                    print('\t%d.%d' % (sequenceID, i_GRAM))

                    N_start = i_GRAM * N_GRAM
                    N_end = (i_GRAM + 1) * N_GRAM
                    if i_GRAM == group_GRAM:
                        N_end = len(all_pclist[id_file])

                    pts_GRAM = all_pclist[id_file][N_start:N_end, ...]
                    labels_GRAM = all_labelvallist[id_file][N_start:N_end, ...]
                    # data_batches = np.concatenate(pts, 0) # 3D to 2D
                    # labels_batches = np.concatenate(labels, 0) # 3D to 2D

                    # Model inference
                    inputs = {'point_clouds': torch.from_numpy(pts_GRAM).to(device)}
                    tic = time.time()
                    with torch.no_grad():
                        optimizer.zero_grad()
                        predicted_labels = torch.from_numpy(
                            np.random.randint(10, 11, size=(len(pts_GRAM), len(pts_GRAM[0])))).to(device)
                        res = model.validate_once((inputs['point_clouds'], predicted_labels), None)
                        predicted_labels = res['log']['label']

                    toc = time.time()
                    pl = predicted_labels.cpu().detach().numpy()

                    for idx in range(len(pl)):
                        for idy in range(len(pl[idx])):
                            if pl[idx][idy] == labels_GRAM[idx][idy]:
                                correct += 1

                    total = pts_GRAM.shape[0] * pts_GRAM.shape[1]

                    print(correct / total)

                    labels[N_start:N_end, ...] = pl

                # f = h5py.File("/home/en1060/Desktop/importh5/classroom_predict.h5", 'w')
                # f = h5py.File(item + 'predict.h5', 'w')
                # data = f.create_dataset("data", data=pts)
                # pid = f.create_dataset("label", data=labels)
                pts2hd5.hd5prediction2txt(all_pclist[id_file], labels, all_gridoriginlist[id_file],
                                          os.path.join(path, all_filelist[id_file] + '.pred.txt'), all_centers[id_file])

                sequenceID += 1
        elif True: # single ply/ascii validation
            #path = '/home/en1060/Desktop/importh5'#/utility-test'
            path = '/home/en1060/Desktop/importh5'
            a_data, D2labelVal_list, a_origins, center = pts2hd5.asc2hdf5data_prediction(path)
            print(sequenceID)
            # pts, labels = i3ssLoader._load_data_file(item)
            # pts, labels, origins, center = pts2hd5.ply2hdf5data(item)
            group_GRAM = math.floor(len(a_data) / N_GRAM)
            labels = D2labelVal_list.copy()

            for i_GRAM in range(group_GRAM + 1):
                correct = 0
                print('\t%d.%d' % (sequenceID, i_GRAM))

                N_start = i_GRAM * N_GRAM
                N_end = (i_GRAM + 1) * N_GRAM
                if i_GRAM == group_GRAM:
                    N_end = len(a_data)

                pts_GRAM = a_data[N_start:N_end, ...]
                labels_GRAM = D2labelVal_list[N_start:N_end, ...]
                # data_batches = np.concatenate(pts, 0) # 3D to 2D
                # labels_batches = np.concatenate(labels, 0) # 3D to 2D

                # Model inference
                inputs = {'point_clouds': torch.from_numpy(pts_GRAM).to(device)}
                tic = time.time()
                with torch.no_grad():
                    optimizer.zero_grad()
                    predicted_labels = torch.from_numpy(
                        np.random.randint(10, 11, size=(len(pts_GRAM), len(pts_GRAM[0])))).to(device)
                    res = model.validate_once((inputs['point_clouds'], predicted_labels), None)
                    predicted_labels = res['log']['label']

                toc = time.time()
                pl = predicted_labels.cpu().detach().numpy()

                for idx in range(len(pl)):
                    for idy in range(len(pl[idx])):
                        if pl[idx][idy] == labels_GRAM[idx][idy]:
                            correct += 1

                total = pts_GRAM.shape[0] * pts_GRAM.shape[1]

                print(correct / total)

                labels[N_start:N_end, ...] = pl

            # f = h5py.File("/home/en1060/Desktop/importh5/classroom_predict.h5", 'w')
            # f = h5py.File(item + 'predict.h5', 'w')
            # data = f.create_dataset("data", data=pts)
            # pid = f.create_dataset("label", data=labels)
            pts2hd5.hd5prediction2txt(a_data, labels, a_origins, os.path.join(path, 'res.pred.txt'), center)

            sequenceID += 1

        else:
            sequenceAll = len(fList)
            correctNum = [0] * 12
            type1errorNum = [0] * 12 # pl != val while label = val
            type2errorNum = [0] * 12 # pl = val while label != val
            correctRate = [0.0] * 12
            type1errorRate = [0.0] * 12
            type2errorRate = [0.0] * 12

            for item in fList:
                print("{} of {}".format(sequenceID, sequenceAll))
                pts, labels = i3ssLoader._load_data_file(item)
                # pts, labels, origins, center = pts2hd5.ply2hdf5data(item)
                group_GRAM = math.floor(len(pts) / N_GRAM)
                labels = labels.copy()

                for i_GRAM in range(group_GRAM + 1):
                    correct = 0
                    print('\t%d.%d' % (sequenceID, i_GRAM))

                    N_start = i_GRAM * N_GRAM
                    N_end = (i_GRAM + 1) * N_GRAM
                    if i_GRAM == group_GRAM:
                        N_end = len(pts)

                    pts_GRAM = pts[N_start:N_end, ...]
                    labels_GRAM = labels[N_start:N_end, ...]
                    # data_batches = np.concatenate(pts, 0) # 3D to 2D
                    # labels_batches = np.concatenate(labels, 0) # 3D to 2D

                    # Model inference
                    inputs = {'point_clouds': torch.from_numpy(pts_GRAM).to(device)}
                    tic = time.time()
                    with torch.no_grad():
                        optimizer.zero_grad()
                        predicted_labels = torch.from_numpy(
                            np.random.randint(10, 11, size=(len(pts_GRAM), len(pts_GRAM[0])))).to(device)
                        res = model.validate_once((inputs['point_clouds'], predicted_labels), None)
                        predicted_labels = res['log']['label']

                    toc = time.time()
                    pl = predicted_labels.cpu().detach().numpy()

                    for idx in range(len(pl)):
                        for idy in range(len(pl[idx])):
                            if pl[idx][idy] == labels_GRAM[idx][idy]:
                                correct += 1
                                correctNum[labels_GRAM[idx][idy]] += 1
                            else:
                                type1errorNum[pl[idx][idy]] += 1
                                type2errorNum[labels_GRAM[idx][idy]] += 1

                    total = pts_GRAM.shape[0] * pts_GRAM.shape[1]

                    for idx in range(12):
                        if correctNum[idx] + type1errorNum[idx] > 0:
                            correctRate[idx] = correctNum[idx] / (correctNum[idx] + type1errorNum[idx])
                        if correctNum[idx] + type1errorNum[idx] > 0:
                            type1errorRate[idx] = type1errorNum[idx] / (correctNum[idx] + type1errorNum[idx])
                        if correctNum[idx] + type2errorNum[idx] > 0:
                            type2errorRate[idx] = type2errorNum[idx] / (correctNum[idx] + type2errorNum[idx])
                    print(correct / total)

                    #labels[N_start:N_end, ...] = pl

                # f = h5py.File("/home/en1060/Desktop/importh5/classroom_predict.h5", 'w')
                # f = h5py.File(item + 'predict.h5', 'w')
                # data = f.create_dataset("data", data=pts)
                # pid = f.create_dataset("label", data=labels)
                #pts2hd5.hd5prediction2txt(all_pclist[id_file], labels, all_gridoriginlist[id_file],
                #                          os.path.join(path, all_filelist[id_file] + '.pred.txt'), all_centers[id_file])

                sequenceID += 1
            print("finished")

    else:#classification
        for item in fList:
            # Load and preprocess input point cloud
            point_cloud = read_pcd_txt(item)
            pc = preprocess_point_cloud(point_cloud, 10000)
            #print('Loaded point cloud data: %s' % (item))

            # Model inference
            inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
            tic = time.time()
            with torch.no_grad():
                optimizer.zero_grad()
                labels = torch.from_numpy(np.random.randint(10, 11, size=B)).to(device)
                res = model.validate_once((inputs['point_clouds'], labels), None)

            toc = time.time()
            if res['log']['label'] != 37:
                print('Type: %d' % (res['log']['label']))
                print('Inference time: %f' % (toc - tic))
            if res['log']['label'] == 37: #valve
                correct += 1

    print('Total Correct: %d' % (correct))
    print('Correct rate: %.3f' % (correct/total))

        #res['point_clouds'] = inputs['point_clouds']
    #end_points['point_clouds'] = inputs['point_clouds']
    #pred_map_cls = parse_predictions(end_points, eval_config_dict) bounding box generation
    #print('Finished detection. %d object detected.' % (len(pred_map_cls[0])))
    print('finished!')

    #dump_dir = os.path.join(demo_dir, '%s_results' % (FLAGS.dataset))
    #if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    #MODEL.dump_results(end_points, dump_dir, DC, True)
    #print('Dumped detection results to folder %s' % (dump_dir))

if __name__ == '__main__':
    main()

