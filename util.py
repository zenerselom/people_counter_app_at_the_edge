import logging
import cv2
import os
import sys
import numpy as np
from time import time
from math import exp as exp

#logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
#log = logging.getLogger()

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.    
    #----------------------------------------------Layers parameters display-------------------------------------------
    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        #[log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]
        
def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)    


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)        
        
def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(1): # to only detect human
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects            
def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    else:
        return area_of_overlap / area_of_union 

def process_result(frame,p_frame,output,reseau,prob_threshold,log,iou_threshold):
    """
    return bounding box for human only
    input: frame,p_frame,output,network,prob_threshold,log,iou_threshold
    output: dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence) 
    """   
    
    objects = list()  
    for layer_name, out_blob in output.items():
        out_blob = out_blob.reshape(reseau.network.layers[reseau.network.layers[layer_name].parents[0]].shape)
        layer_params = YoloParams(reseau.network.layers[layer_name].params, out_blob.shape[2])
        #log.info("Layer {} parameters: ".format(layer_name))
        #layer_params.log_params()
        objects += parse_yolo_region(out_blob, p_frame.shape[2:],frame.shape[:-1], layer_params,prob_threshold)
    

    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]
    #if len(objects) :
        #log.info("\nDetected boxes for batch {}:".format(1))
        #log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
        
    origin_im_size = frame.shape[:-1]
    #for obj in objects:
        # Validation bbox of detected object
        
        #if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            #continue
     #   color = (int(min(obj['class_id'] * 12.5, 255)),min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
     #   det_label = str(obj['class_id'])
        #log.info("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],obj['ymin'], obj['xmax'], obj['ymax'],color))

      #  cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
      #  cv2.putText(frame,"#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',(obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    

    return objects
    
def process_result_cv(frame,output,prob_threshold,iou_threshold):
    """
    return bounding box for human only
    input: frame,output,prob_threshold,log,iou_threshold
    output: dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence) 
    """      
    origin_im_size = frame.shape[:-1]
    objects = list()  
    for out_blob in output:
        for detection in out_blob:
            #print (detection)
            #a = input('GO!')
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[0] #scores[classid]
            # Consider only the predictions that are above a certain confidence level
            if confidence > prob_threshold:
                # TODO Check detection
                box = detection[0:4] * np.array([origin_im_size[1], origin_im_size[0], origin_im_size[1], origin_im_size[0]])
                x, y, w, h = box.astype('int')
                #cv2.putText(frame, "x:{}  y:{}  w:{}  h:{}  confidence:{}".format(x,y,w,h,confidence), (15, 145), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                #x = int(x - (w / 2))
                #y = int(y - (h / 2))
        #log.info("Layer {} parameters: ".format(layer_name))
        #layer_params.log_params()
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=0, confidence=confidence,h_scale= 1, w_scale=1))
    

    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]
    #if len(objects) :
        #log.info("\nDetected boxes for batch {}:".format(1))
        #log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
        
    
    #for obj in objects:
        # Validation bbox of detected object
        
        #if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            #continue
        #color = (int(min(obj['class_id'] * 12.5, 255)),min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
        #det_label = str(obj['class_id'])
        #log.info("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],obj['ymin'], obj['xmax'], obj['ymax'],color))

        #cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        #cv2.putText(frame,"#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',(obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    

    return objects
