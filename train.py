from copyreg import pickle
from pickle import Pickler
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog

import os 
import pickle

from utils import *

config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/object_detection"
num_classes = 3    # set the no. of classes according to the dataset
device = "cuda" # "cpu"

train_dataset_name = "LP_train"
train_image_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/train"
train_json_annot_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/train/_annotations.coco.json"

test_dataset_name = "LP_test"
test_images_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/test"
test_json_annot_path = "/mnt/d/SFFP-2022/RoboFlow/PKLot.v2-640.coco/test/_annotations.coco.json"

cfg_save_path = "OD_cfg.pickle"
###################################
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_image_path)
register_coco_instances(name = test_dataset_name, metadata={}, json_file=test_json_annot_path, image_root=test_images_path)

metadata = MetadataCatalog.get(train_dataset_name)
#MetadataCatalog.get(train_dataset_name).thing_classes = ["occupied", "unoccupied"]
print("Meta:", metadata.as_dict())

plot_samples(dataset_name=train_dataset_name, n=2)

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
   main()

