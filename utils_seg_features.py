
from matplotlib.pyplot import get_current_fig_manager
from torch import float64
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
import cv2
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, ssim
from detectron2.structures import Boxes, BoxMode

#deepsort
from deep_sort import preprocessing, nn_matching
nms_max_overlap = 1.0
DETCTION_THRESHOLD = 0.7
import pickle
import os
import torch
###################################
#
#  feature vector from Detectron2
#################################
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference_single_image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.modeling import build_backbone
cfg_save_path = "OS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
   cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
#########################################################

# Supress/hide the warning  # IMPORTANT please check logic
np.seterr(invalid='ignore')

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
   
def IsReID(image1, image2):
    in_img1 = cv2.imread(image1)
    in_img2 = cv2.imread(image2)

    out_ssim = ssim(in_img1, in_img2)
    return out_ssim
    
# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes   

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
    
def on_video_ReID(videoPath, predictor, curr_tracker, curr_tracker_dets, dataset_name):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()==False):
            print("Error opening file...")
            return

        (success, image) = cap.read()
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)
        #Tracker
        bboxes = []
        scores = []
        names = []
        cosine_features = []

        while success:
            bboxes.clear()
            scores.clear()
            names.clear()
            cosine_features.clear()

            count = 0
            detections = []
            prev_detections = []
            
            predictions = predictor(image)
            v = Visualizer(image[:,:,::-1], metadata=dataset_custom_metadata, instance_mode=ColorMode.SEGMENTATION)
            output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    
            #cv2.imshow("Result", output.get_image()[:,:,::-1]) 
            # Call the tracker
            boxes = predictions["instances"].pred_boxes.to('cpu')
            instances = predictions["instances"]
            if len(instances) > 0:
                bboxes = BoxMode.convert(
                    instances.pred_boxes.tensor.cpu(),
                    BoxMode.XYXY_ABS,
                    BoxMode.XYXY_ABS,
                    ).tolist()    
            print("Detections-BBox:", boxes)
            print("Converted-Box:", bboxes)        
            det_bboxes = np.zeros([len(boxes), 4], dtype=float)
            #det_score = np.zeros([len(boxes), 4], dtype=float)
            det_cls = np.zeros([len(boxes), 4], dtype=float)
            cols = 512

            class_scores = predictions["instances"].scores.to('cpu')
            count = 0
            for det_scores in class_scores:
                if (det_scores > DETCTION_THRESHOLD):
                    scores.append(det_scores)
                else:
                    scores.append(det_scores)
                    print("low-score:", det_scores)

            print("Scores:", len(scores))
            print("scores-list:", scores)
            #feature_cosine_ndarray = np.random.randint([len(boxes), cols], dtype='int64')
            
            count = 0
            for box in bboxes:
                if (scores[count] > DETCTION_THRESHOLD):
                    bboxes.append(box)
                    det_bboxes[count] = box
                    #cols = 512
                    #feature_cosine_ndarray[count] = np.zeros([len(boxes), cols], dtype=float)
                    count +=1
             
            class_labels = predictions["instances"].pred_classes.to('cpu')
            count = 0
            
            masks_array = np.asarray(predictions["instances"].pred_masks.to("cpu"))

            #RGB features
            #feature_cosine_ndarray = np.zeros([len(boxes), 3], dtype=float)
            
            for labels in class_labels:
                if (scores[count] > DETCTION_THRESHOLD):
                    #print("curr-labels:", float(labels))
                    #print("curr-type:", type(labels))
                    names.append(int(labels))
                    #det_cls[count] = float(labels)
                    #feature_cosine_ndarray[count] = masks_array[count+1]
                    count +=1
            count = 0

            print("class-scores:", class_scores)
            #print("Summary:", count, len(det_bboxes))
            #print("det_cls",names, len(names))
            #print("det_boxes",det_bboxes, len(det_bboxes))
            #print("score",scores, len(scores))
            #print("feature_cosine_ndarray",feature_cosine_ndarray, len(feature_cosine_ndarray))
            
            #post detection
            #det_bboxes[count] = boxes[]
            #feature_cosine_ndarray[count] = reid_result
            #arrbboxes = np.array(bboxes)
            arrbboxes = det_bboxes
            #arrbboxes = format_boxes(det_bboxes, 1, 1)
            #print("det_bboxes:", det_bboxes)
            #print("det_bboxes_type:", type(det_bboxes))
            #print("arrbboxes:", arrbboxes)
            #arrbboxes = bboxes[0:int(num_objects)]
            #arrbboxes = arrbboxes[arrbboxes != 0.0]
            arrbboxes_nz = arrbboxes[~np.all(arrbboxes == 0, axis=1)]
            print("arrbboxes:", arrbboxes_nz)
            
            scores = [x for x in scores if x > DETCTION_THRESHOLD]
             
            feature_cosine_ndarray = np.random.randint(0,5,size = (len(names),cols))
 
            #feature_cosine_ndarray = np.zeros([len(names), cols], dtype=float)
            
            #########################################
            ## ADD feature vector here############
            #######################################
            # Pick an item to mask
            masks = np.asarray(predictions["instances"].pred_masks.to("cpu"))
            print(">>>>>>Total: {} No. of objects: {}", len(class_labels), len(names))
            #for item_mask in masks[0]:
            for item_mask in range(len(names)):
                print("item_mask:", item_mask)
                curr_item_mask = masks[item_mask+1]

                # Get the true bounding box of the mask (not the same as the bbox prediction)
                segmentation = np.where(curr_item_mask == True)
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
                print(x_min, x_max, y_min, y_max)

                #image = cv2.imread('my_image.jpg')
                cropped = Image.fromarray(image[y_min:y_max, x_min:x_max, :], mode='RGB')
                cropped_cv2 = np.asarray(cropped)  
                      
                height, width = cropped_cv2.shape[:2]
                print("Cropped Width:{} Height:{}", width, height)
                #width = x_max-x_min
                #height = y_max-y_min
                image_torch = torch.as_tensor(cropped_cv2.astype("float32").transpose(2, 0, 1))
                inputs = [{"image": image_torch, "height": height, "width": width}]
                with torch.no_grad():
                     trained_model = build_model(cfg)
                     trained_model.eval()
                     pre_images = trained_model.preprocess_image(inputs)  # don't forget to preprocess
            
                     detectron2_features = trained_model.backbone(pre_images.tensor)    
                
                     print("after image_torch", len(detectron2_features), type(detectron2_features))
                     proposals, _ = trained_model.proposal_generator(pre_images, detectron2_features)  # RPN

                     features_ = [detectron2_features[f] for f in trained_model.roi_heads.box_in_features]
                     box_features = trained_model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                     box_features = trained_model.roi_heads.box_head(box_features)  # features of all 1k candidates
                     print("box_features", box_features.shape)
                 
                 
                     ###################################################################################
                     predictions = trained_model.roi_heads.box_predictor(box_features)
                     pred_instances, pred_inds = trained_model.roi_heads.box_predictor.inference(predictions, proposals)
                     pred_instances = trained_model.roi_heads.forward_with_given_boxes(detectron2_features, pred_instances)

                     # output boxes, masks, scores, etc
                     pred_instances = trained_model._postprocess(pred_instances, inputs, pre_images.image_sizes)  # scale box to orig size
                     # features of the proposed boxes
                     feats = box_features[pred_inds]            
                     print("extracted features:", len(feats))
            
            
#########################################################            
            #with torch.no_grad():
            #images = model.preprocess_image(inputs)  # don't forget to preprocess

            #image_features = ImageList.from_tensors([image])
            #trained_model = build_model(cfg)
            #features = trained_model.backbone(image_features.tensor)

            # Create a cropped image from just the portion of the image we want
            cropped = Image.fromarray(image[y_min:y_max, x_min:x_max, :], mode='RGB')
            #print("cropped:", image.shape)
            cropped_vec = cropped
            #cropped.save('temp.png')
            #cropped.show()
            data = np.asarray(cropped_vec, dtype="int32")
            data = data.reshape(-1)
            print("data{} shape{}:", data, data.shape)
            print("Vec-cropped{} type{}:", cropped_vec, type(cropped_vec))

            #feature_cosine_ndarray = np.random.randint(0,5,size = (len(names),cols))
            #feature_cosine_ndarray = np.array(cropped)
            
            print("len:", type(arrbboxes), len(arrbboxes_nz), len(scores), len(names), feature_cosine_ndarray.shape)
            print("feature_cosine_ndarray:", feature_cosine_ndarray)
            ''' 
            detections = [curr_tracker_dets(tracker_bbox, score, class_name, feature) for
                                 tracker_bbox, score, class_name, feature in
                                  zip(arrbboxes_nz, scores, names, feature_cosine_ndarray)]
            '''
            detections = [curr_tracker_dets(tracker_bbox, score, class_name, feature) for
                                 tracker_bbox, score, class_name, feature in
                                  zip(arrbboxes_nz, scores, names, feature_cosine_ndarray)]


            # run non-maxima supression
            det_boxs = np.array([d.tlwh for d in detections])
            det_scores = np.array([d.confidence for d in detections])
            det_classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(det_boxs, det_scores, nms_max_overlap, det_scores)
            detections = [detections[i] for i in indices]       

            del feature_cosine_ndarray       #check for erros    
            if detections != prev_detections:    
                print("Before update Detections:", len(detections))

                curr_tracker.predict()
                curr_tracker.update(detections)
                print("After update Detections:", len(detections))
                    # update tracks
                for track in curr_tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
		    #draw tracker box
		    #v.draw_box(bbox)
                    v.draw_text(str(track.track_id), (int(bbox[0]), int(bbox[1])))
                    #plt.imshow(output.get_image())
                    cv2.imshow("Result", output.get_image()[:,:,::-1]) 
                    #plt.show()

		    
                    class_name = track.get_class()
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            else:
                print("Same detection")

            prev_detections = detections
           
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()








