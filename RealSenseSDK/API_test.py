### pyrealsense2 INSTRUCTION ###
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

pipeline = rs.pipeline()
pipeline.start()

#try:
#    while True:
#        frames = pipeline.wait_for_frames()
#        depth = frames.get_depth_frame()
#        
#        if not depth:
#            continue
#        
#        coverage = [0] * 64
#        for y in range(480):
#            for x in range(640):
#                dist = depth.get_distance(x, y)
#                if 0 < dist and dist < 1:
#                    coverage[x//10] += 1
#            
#            if y % 20 == 19:
#                line = ""
#                for c in coverage:
#                    line += " .:nhBXWW"[c//25]
#                coverage = [0]*64
#                print(line)
#
#finally:
#    pipeline.stop()
    
### numpy INSTRUCTION ###
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
img_data = frames.get_color_frame().as_frame().get_data()
depth_data = depth.as_frame().get_data()
np_image = np.asanyarray(img_data)
np_depth = np.asanyarray(depth_data)
plt.imshow(np_image)
plt.imshow(np_depth)