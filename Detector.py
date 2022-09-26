from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

class Detector:
    def __init__(self):
        self.cfg = get_cfg()


        #Load model config and model
        #https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        #self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        #https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        #self.cfg.MODEL.DEVICE = "cpu" #cpu or cuda
        self.cfg.MODEL.DEVICE = "cuda" #cpu or cuda
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)      
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW)
              

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)
