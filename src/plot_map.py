#! /usr/bin/env/python3
import os
import yaml
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import cv2
import time
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy import app,scene
import vispy.io as io
import utm

class Dataset():
    def __init__(self,ds_path,seq_num=None) -> None:
        self. dataset_path = ds_path
        if seq_num ==None:
            self.sequences = os.listdir(ds_path)
        else:
            self.sequences = [seq_num]
        self.canvas = SceneCanvas(keys='interactive', show=True,size=(1920,1080))
        self.timer = app.Timer()

        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='black',bgcolor="white", parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0,0,col_span=3)

        self.scan_vis = visuals.Markers()
        self.scan_view.camera = scene.cameras.FlyCamera()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)
        self.seq = 0
        self.file_num = 0
        self.files = os.listdir(os.path.join(self.dataset_path,self.sequences[self.seq],'velodyne'))
        self.files = [s.strip(".bin") for s in self.files]
        self.files.sort()
        print("Loaded {} files".format(len(self.files)))

            # make semantic colors
        self.cfg = yaml.safe_load(open("/home/mkz/git/r2d2/config/r2d2.yaml",'r'))
        sem_color_dict = self.cfg["color_map"]
        max_sem_key = 0
        for key, data in sem_color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in sem_color_dict.items():
            self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

        self.sem_label_color = []
        self.map = []
        self.pose_file = None

        self.clock = time.time()
        print("\nLoading map...")
        finished = False
        while not finished:
            finished = self.load_map()

        self.scan_vis.set_data(self.map,face_color=self.sem_label_color[...,::-1],edge_color=self.sem_label_color[...,::-1],size=1)


    def open_pointcloud(self,full_file_path):
        if os.path.isfile(full_file_path):
                # if all goes well, open pointcloud
            pcd = np.fromfile(full_file_path, dtype=np.float32)
            pcd = pcd.reshape((-1, 4))

            # put in attribute
            points = pcd[:, 0:3]    # get xyz
            remissions = pcd[:, 3]  # get remission
            
            return np.array(points),remissions
        else:
            print("Error file doesn not exist:{}".format(full_file_path))
    
    def draw_bbox(self,img,img_d,labels):
        # if os.path.isfile(label_file_name):
            # file = open(label_file_name,'r')
            # labels = file.readlines()
        if len(labels)==0:
            return img,img_d
        for label in labels:
            l = label.split()
            print(l)
            xmin,ymin,xmax,ymax = float(l[-12]),float(l[-11]),float(l[-10]),float(l[-9])
            print(img.shape)
            xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            print(xmin,ymin,xmax,ymax)
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),3)
            cv2.rectangle(img_d,(xmin,ymin),(xmax,ymax),(0,0,255),3)



        return img,img_d

    def open_label(self, filename):

        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        # label = label[self.idx_valid]

        # set it
        sem_label = label & 0xFFFF  # semantic label in lower half  
        sem_label_color = self.sem_color_lut[sem_label]
        sem_label_color = sem_label_color.reshape((-1, 3))     

        return sem_label_color



    def play_sequence(self):
        st = 0.2
        fig = plt.figure(figsize=(16,24))
        grid = plt.GridSpec(3,3,wspace =0.1, hspace = 0.1)

        for seq in self.sequences:

            files = os.listdir(os.path.join(self.dataset_path,seq,'velodyne'))
            base_path = os.path.join(self.dataset_path,seq)
            files = [s.strip(".bin") for s in files]
            gps_file = open(os.path.join(self.dataset_path,seq,'gps',"gps.txt"),'r')
            gps_log = gps_file.readlines() 

            for frame in range(0,len(files)):
                time.sleep(st)
                pc,r = self.open_pointcloud(os.path.join(base_path,"velodyne",files[frame]+".bin"))
                # label

                # print(gps)
                # Plot point cloud
                print(pc[0])
                ax = plt.subplot(grid[:2,:],projection="3d")
                ax.scatter(pc[:,0],pc[:,1],pc[:,2])
                r = np.linalg.norm(pc,axis=1)
                pc = pc[r>0.1]
                ax = plt.subplot(grid[:2,:])
                ax.scatter(pc[:,0],pc[:,1],s=0.5)

                # Plot images
                img_left = cv2.resize(img_left, (960,540), interpolation = cv2.INTER_AREA)
                img_left = cv2.flip(img_left,0)
                img_d = cv2.resize(img_d, (960,540), interpolation = cv2.INTER_AREA)
                ax = plt.subplot(grid[2,0])
                ax.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
                ax = plt.subplot(grid[2,1])
                ax.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))

                # Plot gps coordinates
                # print(gps)
                lat,lon = float(gps[1]),float(gps[2])
                ax = plt.subplot(grid[2,2])
                ax.scatter(lat,lon,c="blue")
                plt.pause(0.1)

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0  

    def increase_brightness(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img 

    def load_map(self):
        # viridis_map = self.get_mpl_colormap("viridis")
        # viridis_colors = viridis_map[viridis_range]

        
        base_path = os.path.join(self.dataset_path,self.sequences[self.seq])

        if self.file_num==0:
            self.pose_file = open(os.path.join(base_path,"poses.txt"))
            self.poses = self.pose_file.readlines()

        pose = self.poses[self.file_num]
        pose = pose.split()
        a,b,c,d,e,f,g,h,i,j,k,l = float(pose[0]),float(pose[1]),float(pose[2]),float(pose[3]),float(pose[4]),float(pose[5]),float(pose[6]),float(pose[7]),float(pose[8]),float(pose[9]),float(pose[10]),float(pose[11])
        tf = np.eye(4,dtype=np.float32)
        tf_s = np.array([[a,b,c,d,e,f,g,h,i,j,k,l]])
        tf_s = tf_s.reshape((3,4))
        tf[:3,:] = tf_s

        pc,r = self.open_pointcloud(os.path.join(base_path,"velodyne",self.files[self.file_num]+".bin"))
        norm = np.linalg.norm(pc,axis=1)
        valid = norm>0.5
        pc = pc[valid]
        pc_H = pc.T
        ones = np.ones((1,pc.shape[0]),dtype=np.float32)
        pc_H = np.vstack([pc_H,ones])
        pc_global = tf @ pc_H
        pc_global = pc_global.T
        if self.file_num == 0:
            self.map = pc_global[:,:3]
        else:
            self.map = np.vstack([self.map,pc_global[:,:3]])

        sem_label_color = self.open_label(os.path.join(base_path,"labels",self.files[self.file_num]+".label"))
        sem_label_color = sem_label_color[valid]
        if self.file_num==0:
            self.sem_label_color = sem_label_color
        else:
            self.sem_label_color = np.vstack([self.sem_label_color,sem_label_color])


        if self.file_num%10==0:
            percentage = self.file_num/len(self.files)
            print("\nLoading {} percent complete".format(percentage))
        self.file_num +=1

        if self.file_num>= len(self.files):
            # self.seq +=1
            # if self.seq >= len(self.sequences):
            #     self.seq = 0
            # self.file_num = 0
            # self.gps_logs = None
            # if os.path.isfile(os.path.join(self.dataset_path,self.sequences[self.seq],'gps',"gps.txt")):
            #     self.gps_file = open(os.path.join(self.dataset_path,self.sequences[self.seq],'gps',"gps.txt"),'r')
            #     self.gps_log = self.gps_file.readlines() 
            # else:
            #     self.gps_file = []
            #     self.gps_log = []
            return True
        return False

    def vispy_update(self,event):
        now = time.time()
        if now - self.clock>10.0:
            img = self.canvas.render()
            name = str(now)+".png"
            io.write_png(os.path.join("./images",name),img)
            self.clock = now

        

    def run(self):
        self.timer.connect(self.vispy_update)
        self.timer.start()
        vispy.app.run()





def main():
    viz = Dataset("/home/mkz/git/lidar-bonnetal/train/tasks/semantic/dataset/sequences",seq_num="17")
    viz.run()


if __name__ == "__main__":
    main()