### point cloud API ###
import pyrealsense2 as rs
import numpy as np

pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()
pipeline.start()

try:
    frames = pipeline.wait_for_frames()
    
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    
    pc.map_to(color)
    points = pc.calculate(depth)
    points.export_to_ply("1.ply", color)


finally:
    pipeline.stop()