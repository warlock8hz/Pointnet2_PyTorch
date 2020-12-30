#
import os
import argparse
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    import open3d as o3d
except ImportError:
    raise ImportError( 'Please install open3d-python with `pip install open3d-python`.')
import h5py as h5py

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

parser = argparse.ArgumentParser()
#DATA_PATH = os.path.join(ROOT_DIR, 'data')
#parser.add_argument('--file_name', type=str, default=os.path.join(BASE_DIR, 'pointnet2/data/indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5'))#'scene0000_00_vh_clean.pcd')
parser.add_argument('--file_name', type=str, default=os.path.join(BASE_DIR, 'pointnet2/data/indoor3d_sem_seg_hdf5_data_visualization'))
parser.add_argument('--batch', type = int, default = 0)
config = parser.parse_args()

pathName = os.path.join(BASE_DIR, config.file_name)
fList = []
scanFile(pathName, fList)  # get all files in the folder
fList.sort()

for item in fList:

    f = h5py.File(item, 'r')

    #id = config.batch
    print("Keys %s" % f.keys())
    a_group_key = list(f.keys())[0]
    data = f['data'][:]
    pt_data = data[:, :, 6:10]   # convert to three-column shape for open3D
    label = f['label'][:]
    #print(data[999][4095])
    rgb = np.repeat(label[:, :, np.newaxis], 3, axis=2)
    rgb = rgb / np.amax(label)

    pcd_each = o3d.geometry.PointCloud()
    pcd_comb = o3d.geometry.PointCloud()

    #nD1, nD2, nD3 = pt_data.shape
    for i in range(len(pt_data)):
        pcd_each.points = o3d.utility.Vector3dVector(pt_data[i])
        pcd_each.colors = o3d.utility.Vector3dVector(rgb[i])
        pcd_comb += pcd_each

    o3d.io.write_point_cloud(item+'.ply', pcd_comb, True) # True for write in ASCII

    #o3d.visualization.draw_geometries([pcd])
#o3d.draw_geometries(pcdMerged)

