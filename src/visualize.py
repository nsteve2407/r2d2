#! /usr/bin/env/python3
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

dataset_path = ""
sequences = os.listdir(dataset_path)


class Dataset():
    def __init__(self) -> None:
        self. dataset_path = ""
        sequences = os.listdir(dataset_path)


    def open_pointcloud(self,full_file_path):
        if os.path.isfile(full_file_path):
                # if all goes well, open pointcloud
            pcd = np.fromfile(full_file_path, dtype=np.float32)
            pcd = pcd.reshape((-1, 4))

            # put in attribute
            points = pcd[:, 0:3]    # get xyz
            remissions = pcd[:, 3]  # get remission
            
            return points,remissions
        else:
            print("Error file doesn not exist:{}".format(full_file_path))
    
    def draw_bbox(img,label_file_name):
        if os.isfile(label_file_name):
            file = open(label_file_name,'r')
            labels = file.readlines()
            for label in labels:
                l = label.split(" ")

        return img





