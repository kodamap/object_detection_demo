""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import numpy as np
import math
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from timeit import default_timer as timer

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

resize_prop = (640, 480)


class VideoCamera(object):
    def __init__(self, input, detections, no_v4l):
        if input == 'cam':
            self.input_stream = 0
            if no_v4l:
                self.cap = cv2.VideoCapture(self.input_stream)
            else:
                # for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
                try:
                    self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
                except:
                    import traceback
                    traceback.print_exc()
                    print(
                        "\nPlease try to start with command line parameters using --no_v4l\n"
                    )
                    os._exit(0)
        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        ret, self.frame = self.cap.read()
        cap_prop = self._get_cap_prop()
        logger.info("cap_pop:{}, frame_prop:{}".format(cap_prop, resize_prop))

        self.detections = detections

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, is_async_mode, flip_code, is_object_detection,
                  is_face_detection, is_age_gender_detection,
                  is_emotions_detection, is_head_pose_detection,
                  is_facial_landmarks_detection):

        if is_async_mode:
            ret, next_frame = self.cap.read()
            next_frame = cv2.resize(next_frame, resize_prop)
            if self.input_stream == 0 and flip_code is not None:
                next_frame = cv2.flip(next_frame, int(flip_code))
        else:
            ret, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, resize_prop)
            next_frame = None
            if self.input_stream == 0 and flip_code is not None:
                self.frame = cv2.flip(self.frame, int(flip_code))

        if is_object_detection:
            frame = self.detections.object_detection(self.frame, next_frame,
                                                     is_async_mode)
        if is_face_detection:
            frame = self.detections.face_detection(
                self.frame, next_frame, is_async_mode, is_age_gender_detection,
                is_emotions_detection, is_head_pose_detection,
                is_facial_landmarks_detection)

        # The first detected frame is None
        if frame is None:
            ret, jpeg = cv2.imencode('1.jpg', self.frame)
        else:
            ret, jpeg = cv2.imencode('1.jpg', frame)

        if is_async_mode:
            self.frame = next_frame

        return jpeg.tostring()
