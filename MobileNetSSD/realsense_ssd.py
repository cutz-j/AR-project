### TENSORFLOW API: MobilnetV2_SSD ###
## Import packages ##
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
## 시각화 Tool ##
from matplotlib import pyplot as plt
from PIL import Image
import cv2
## 통신 LIB ##
import mmap as mp
## API Tool ##
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

################ tensorflow version check ####################
sys.path.append("..")
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

## share memory part ##
width = 640
height = 480
img_size = width * height * 3
depth_size = width * height
row = 40
col = 7
info_size = row * col
memory = np.zeros(shape=[row, col], dtype=float)

################# object detection part #####################
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

## Grab path to current working directory ##
CWD_PATH = os.getcwd()
NUM_CLASSES = 80

## Load the label map. ##
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

## Load the Tensorflow model into memory ##
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image #
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') # confidence
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') # 80classes
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

while(True):
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    readMap_image = mp.mmap(0, img_size*4, "IMAGE") # IMAGE READ
    readMap_info = mp.mmap(0, info_size*4, "INFO") # Detection Info WRITE
    readData = readMap_image.read(img_size * 4) # share memory read
    image = np.frombuffer(readData, dtype=np.int, count=img_size)
    image = image.copy().reshape(height, width, 3)
    color_image = image.astype('uint8')
    res_image = color_image.copy()
    # cv image 변환 bgr --> rgb 변환 #
    res_image[:,:,0] = color_image[:,:,2]
    res_image[:,:,2] = color_image[:,:,0]
    frame_expanded = np.expand_dims(res_image, axis=0)
    
    # signal memory --> point cloud stage: not detecting #
    readMap_signal = mp.mmap(0, 8, "SIGNAL")
    signal = np.zeros(shape=[1,1], dtype=np.int32)
    readMap_signal.write(signal)
    memory = np.zeros_like(memory) # memory 초기화 #

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        res_image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    
    # memory write --> detect num이 없을 경우 0 전송 #
    if int(num[0]) == 0:
        memory = np.zeros_like(memory)
    
    else:
        # detect이 됐을 경우, memory에 저장 후 공유메모리 저장 #
        for i in range(int(num[0])):
            memory[i, 0], memory[i, 1], memory[i, 2], memory[i, 3] = boxes[0, i, 0] * height, boxes[0, i, 1] * width, boxes[0, i, 2] * height, boxes[0, i, 3] * width
            memory[i, 4] = scores[0, i]
            memory[i, 5] = float(classes[0, i])
            memory[i, 6] = float(num[0])
    np_arr = memory.ravel().astype(np.float32) 
    readMap_info.write(np_arr) # 공유메모리 저장
    
    ## detect 시각화 ##
    cv2.imshow('Object detector', res_image)

############## POINT CLOUD STAGE ######################
    # 파이썬 cv에서 q를 눌렀을 경우 stage 전환 #
    if cv2.waitKey(1) == ord('q'):
        signal[0] = 1 # signal 변환
        cv2.destroyAllWindows() # 시각화 종료
        readMap_signal = mp.mmap(0, 8, "SIGNAL")
        readMap_signal.write(signal) # signal 전송
        while(signal[0] == 1):
            ## signal이 1인 동안에는 공유메모리 읽기만을 반복: NOT detecting ##
            readMap_signal = mp.mmap(0, 8, "SIGNAL")
            read_signal = readMap_signal.read(4)
            r_signal = np.frombuffer(read_signal, dtype=np.int, count=1)
            if (r_signal[0] == 0):
                signal[0] = 0

    # Press 't' to transform

# Clean up

readMap_image.close()
readMap_info.close()
cv2.destroyAllWindows()

