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
#https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/drone_seg_object_detection"
num_classes = 5    # set the no. of classes according to the dataset
device = "cuda" # "cpu"

train_dataset_name = "Drone_LP_train"
train_image_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/train"
train_json_annot_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/train/_annotations.coco.json"

test_dataset_name = "Drone_LP_test"
test_images_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/test/_annotations.coco.json"
test_json_annot_path = "/mnt/d/SFFP-2022/RoboFlow/Drone-COCO-Seg/test/_annotations.coco.json"

cfg_save_path = "OS_cfg.pickle"
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

