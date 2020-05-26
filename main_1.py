"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
from time import time
from util import*
from sort import*
# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Initialise the class
    #log.info("Creating Inference Engine...")
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    ### TODO: Load the model through `infer_network` ###
    #log.info("Loading network files:\n\t{}".format(args.model))
    #log.info("Loading model to the plugin")
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    #log.info("Preparing inputs")
    infer_network.network.batch_size = 1
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    #check for input stream  is a cam?
    input_stream = 0 if  args.input == "cam" else  args.input
    try:
         cap = cv2.VideoCapture(input_stream)
    except FileNotFoundError:
         print("File {} not available".format(input_stream))
    except Exception as e:
         print("error on loading file:{}".format(e))
         exit(1)
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
    current_count = 0
    total_count = 0
    duration = 0
    count_start_time = 0
    person_num_trigger = 3 # maximum total number before alarm
    min_duration = 5 # minimum time (s) before alarm
    k_ref = 5 #  number of frames before account for non detection
    k = 0
    det_time = [] #inference times array
    
    cap.open(input_stream)
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        #log.info("Starting inference...")
        start_time = time()
        infer_network.exec_net(p_frame,0)
        det_time.append( time() - start_time)
        ### TODO: Wait for the result ###
        # Collecting object detection results
        objects = list()
        if infer_network.wait(0) == 0:
            det_time.append( time() - start_time)
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(0)
            #print(key_pressed)
            ### TODO: Extract any desired stats from the results ###pip 
            start_time = time()
            objects = process_result(frame,p_frame,result,infer_network,prob_threshold,log,prob_threshold)
            parsing_time = time() - start_time
            # Draw performance stats over frame
            inf_time_message = "Inference time: {:.3f} ms **** Inference mean time :{:.3f} ms ".format(det_time[-1] * 1e3,sum(det_time)*1e3/len(det_time))
            parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ####
            num_detection = len(objects)
            delta = num_detection - current_count
            if delta >0 and k == 0:
                 count_start_time = time()
                 current_count = num_detection
                 k = 0
            elif delta < 0 and  k >= k_ref:
                 duration = time()- count_start_time
                 total_count += current_count
                 current_count = num_detection
                 MQTT_MSG_DURATION=json.dumps({"duration": duration})
                 client.publish("person/duration",  MQTT_MSG_DURATION)
                 k= 0
                 #cv2.putText(frame, "past person duration: {}".format(time()- count_start_time), (15, 115), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            elif delta < 0 and k < k_ref:
                  k += 1
            if delta == 0 and k > 0:
                  k = 0
            MQTT_MSG_PERSON=json.dumps({"count": current_count })
            client.publish("person", MQTT_MSG_PERSON)  
            
            if total_count > person_num_trigger :
                  cv2.putText(frame, "number: {} of total people more than trigger limit:{}".format(total_count,person_num_trigger), (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
                  
            if (time()- count_start_time)> min_duration and current_count >=1  :
                  cv2.putText(frame, "Duration: {:2f} s of this person more than trigger limit: {} s".format(time()- count_start_time,min_duration), (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
                  
            for obj in objects:
                  color = (int(min(obj['class_id'] * 12.5, 255)),min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                  det_label = str(obj['class_id'])
                  cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                  cv2.putText(frame,"#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',(obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)      
                  
            
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if number_input_frames == 1:
            cv2.imwrite("out.png", frame)
            
        # Break if escape key pressed
        
        if key_pressed == 27:
            break
        
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    # Disconnect from MQTT
    client.disconnect()

if __name__ == '__main__':
    main()
