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
import numpy as np
import argparse
import importlib
import time
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', default='cls', help='Dataset: cls for classification or seg for segmentation [default: cls]')
#parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
#FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/g800g1/Projects/Pointnet2_PyTorch'
CKPT_DIR = os.path.join(ROOT_DIR, 'CKPT')
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
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG # for loading classification data

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

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
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
    if cfg.task_model == 'semseg':
        checkpoint_folder = 'sem-ssg'

    checkpoint_path = os.path.join(CKPT_DIR, checkpoint_folder, 'best39_valve.ckpt')#'best39_valve_no_scale.ckpt')
    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
    model.eval()  # set model to eval mode (for bn and dp)

    fList = scanFolder(PC_SCALED_VALVE_DIR)#(PC_DIR)
    fList.sort()
    B = 1
    model.cuda()  # move to cuda to work
    total = len(fList)
    correct = 0;

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

