import pyrealsense2 as rs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.chdir("d:/github/AR-project/darkflow_master")
from darkflow.net.build import TFNet
import cv2
import pyximport
pyximport.install()

pipeline = rs.pipeline()
pipeline.start()

options = {"model": "cfg/tiny-yolo.cfg", "load":"bin/yolov3-tiny.weights", "threshold":0.6}
tfnet = TFNet(options)

while 1:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    img_data = frames.get_color_frame().as_frame().get_data()
    depth_data = depth.as_frame().get_data()
    np_image = np.asanyarray(img_data)
    np_depth = np.asanyarray(depth_data)
    
    
    
    imgcv = cv2.imread(img_data)
    res = tfnet.return_predict(imgcv)
    print(res)
    
