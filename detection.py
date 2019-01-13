"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Most of these codes are based on 
# OpenVINO python sample code "object_detection_demo_ssd_async.py" 
# https://software.intel.com/en-us/openvino-toolkit/
# and
# Object detection with deep learning and OpenCV
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

# import the necessary packages
import numpy as np
import cv2
from logging import getLogger, basicConfig, DEBUG, INFO
from timeit import default_timer as timer
from openvino.inference_engine import IENetwork, IEPlugin

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")


class ObjectDetection(object):
    def __init__(self, frame_prop, model_xml, model_bin, device,
                 prob_threshold, cpu_extension, is_async_mode):
        self.prob_threshold = prob_threshold
        # Plugin initialization for specified device and load extensions library if specified
        logger.info("Initializing plugin for {} device...".format(device))
        self.plugin = IEPlugin(device=device, plugin_dirs=None)
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)
        # Read IR
        logger.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        logger.info("Loading IR to the plugin...")
        self.exec_net = self.plugin.load(network=self.net, num_requests=2)
        # Read and pre-process input image
        n, c, h, w = self.net.inputs[self.input_blob].shape
        logger.info("net.inpute.shape(n, c, h, w):{}".format(self.net.inputs[
            self.input_blob].shape))
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1
        logger.info(
            "Starting inference in async mode is {}".format(is_async_mode))

        self.w = frame_prop[0]
        self.h = frame_prop[1]

        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def start_inference(self, frame, next_frame, is_async_mode):
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = timer()
        logger.info("cur_request_id:{} next_request_id:{} is_async:{}".format(
            self.cur_request_id, self.next_request_id, is_async_mode))
        if is_async_mode:
            in_frame = cv2.dnn.blobFromImage(
                cv2.resize(next_frame, (300, 300)), 0.007843, (300, 300),
                127.5)
            self.exec_net.start_async(
                request_id=self.next_request_id,
                inputs={self.input_blob: in_frame})
        else:
            in_frame = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            # is_async_mode change false from true, wait request
            if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
                self.exec_net.start_async(
                    request_id=self.cur_request_id,
                    inputs={self.input_blob: in_frame})
        logger.debug("in_frame shape:{}".format(in_frame.shape))
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_end = timer()
            det_time = inf_end - inf_start
            # Parse detection results of the current request
            logger.debug("computing object detections...")
            detections = self.exec_net.requests[self.cur_request_id].outputs[
                self.out_blob]
            logger.debug("detections shape:{}".format(detections.shape))
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > self.prob_threshold:
                    # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array(
                        [self.w, self.h, self.w, self.h])
                    (startX, startY, endX, endY) = box.astype("int")
                    logger.debug("startX, startY, endX, endY: {}".format(
                        box.astype("int")))

                    # display the prediction
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    logger.info("{} {}".format(self.cur_request_id, label))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else "Inference time: {:.3f} ms".format(
                det_time * 1000)
            async_mode_message = "Async mode is {}. Processing request {}".format(
                is_async_mode, self.cur_request_id)

            if is_async_mode:
                color = (200, 10, 10)
            else:
                color = (10, 10, 200)

            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, async_mode_message, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Calculate FPS
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1
        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

        # Draw FPS in top left corner
        cv2.rectangle(frame, (self.w - 50, 0), (self.w, 17), (255, 255, 255),
                      -1)
        cv2.putText(frame, self.fps, (self.w - 50 + 3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
            ##self.frame = next_frame

        return frame
