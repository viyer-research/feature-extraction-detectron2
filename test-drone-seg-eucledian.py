#############################################################
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from copyreg import pickle
###########################################################
from copyreg import pickle

from sqlalchemy import null
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

import pickle

from utils_seg import *
import os
##########################################
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

############################################
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog

train_dataset_name = "Drone_LP_train"
train_image_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/train"
train_json_annot_path = "//mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/train/_annotations.coco.json"

test_dataset_name = "Drone_LP_test"
test_images_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/test"
test_json_annot_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/test/_annotations.coco.json"

cfg_save_path = "OS_cfg.pickle"
###################################
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_image_path)
register_coco_instances(name = test_dataset_name, metadata={}, json_file=test_json_annot_path, image_root=test_images_path)

metadata = MetadataCatalog.get(train_dataset_name)
###############################################################



cfg_save_path = "OS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
   cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

predictor = DefaultPredictor(cfg)

image_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/test/seq3-drone_0000010_jpg.rf.1d1c9cb09ed588cbe7fa1eecc0947021.jpg"

#image_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/test/2012-09-11_15_53_00_jpg.rf.8282544a640a23df05bd245a9210e663.jpg"
#image_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/test/2012-09-12_06_36_36_jpg.rf.08869047c7e9f62f5ce9334546b52958.jpg"
#image_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/test/2012-09-17_12_29_06_jpg.rf.52b6b85f338407b7647edd791f4e67d7.jpg"

#image_path = "/mnt/d/SFFP-2022/RoboFlow/CarParkProject/carParkImg.png"

#videoPath = "/mnt/d/SFFP-2022/RoboFlow/CarParkProject/carPark.mp4" # give a video here
#test_dataset_name = "LP_test"
test_dataset_name = "Drone_LP_train"


###################################
#Tracker Initialization
###################################
# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
    
# initialize deep sort
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric, 15, 10)

print("create tracker")

videoPath = "/mnt/d/SFFP-2022/RoboFlow/Drone-Traffic/frame-130.mp4" # give a video here

#on_image(image_path, predictor, dataset_name=test_dataset_name)
on_video_ReID(videoPath, predictor, tracker, Detection, dataset_name=test_dataset_name)

    
