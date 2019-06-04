# Import packages
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import mmap as mp
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

''' share memory '''
width = 640
height = 480
img_size = width * height * 3
row = 40
col = 7
info_size = row * col
memory = np.zeros(shape=[row, col], dtype=float)

''' TF_object_detection API '''
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
#PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file

# Number of classes the object detector can identify
NUM_CLASSES = 80

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    readMap_image = mp.mmap(0, img_size*4, "IMAGE") # IMAGE READ
    readMap_info = mp.mmap(0, info_size*8, "INFO") # Detection Info WRITE
    readData = readMap_image.read(img_size * 4) # share memory read
    image = np.frombuffer(readData, dtype=np.int, count=img_size)
    image = image.copy().reshape(height, width, 3)
    color_image = image.astype('uint8')
    res_image = color_image.copy()
    res_image[:,:,0] = color_image[:,:,2]
    res_image[:,:,2] = color_image[:,:,0]
    frame_expanded = np.expand_dims(res_image, axis=0)

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
    
    # memory write #
    if int(num[0]) == 0:
        memory = np.zeros_like(memory)
    
    else:
        for i in range(int(num[0])):
            memory[i, 0], memory[i, 1], memory[i, 2], memory[i, 3] = boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3]
            memory[i, 4] = scores[0, i]
            memory[i, 5] = float(classes[0, i])
            memory[i, 6] = float(num[0])
   
      
    np_arr = memory.ravel().astype(np.float32)
    readMap_info.write(np_arr)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', res_image)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Press 't' to transform

# Clean up
readMap_image.close()
readMap_info.close()
cv2.destroyAllWindows()

