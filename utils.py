
from matplotlib.pyplot import get_current_fig_manager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUPUT_DIR = output_dir

    return cfg

def on_image_reid(image_path, predictor, dataset_name):
        im = cv2.imread(image_path)
        outputs = predictor(im)
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)

        v = Visualizer(im[:,:,::-1], metadata=dataset_custom_metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        #v = print("instace-masks:", outputs["instances"].to("cpu"))

###################################################################################
# Pick an item to mask
        masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))

        item_mask = masks[2]

        # Get the true bounding box of the mask (not the same as the bbox prediction)
        segmentation = np.where(item_mask == True)
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        print(x_min, x_max, y_min, y_max)

        # Create a cropped image from just the portion of the image we want
        cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')
        cropped.save('temp.png')
        cropped.show()
        data = np.array(cropped)
        vec = data.flatten()
        print("Vec:", vec)

        key = cv2.waitKey(1) 
#####################################################################################

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(14,10))
        plt.imshow(v.get_image())
        plt.show()
    
def on_image(image_path, predictor, dataset_name):
        im = cv2.imread(image_path)
        outputs = predictor(im)
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)

        v = Visualizer(im[:,:,::-1], metadata=dataset_custom_metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        #v = print("instace-masks:", outputs["instances"].to("cpu"))

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(14,10))
        plt.imshow(v.get_image())
        plt.show()
    
def on_video(videoPath, predictor, dataset_name):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()==False):
            print("Error opening file...")
            return

        (success, image) = cap.read()
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)

        while success:
            predictions = predictor(image)
            v = Visualizer(image[:,:,::-1], metadata=dataset_custom_metadata, instance_mode=ColorMode.SEGMENTATION)
            output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

            cv2.imshow("Result", output.get_image()[:,:,::-1]) 

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()








