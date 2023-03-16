###############################################################################
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Miguel Grinberg
#
# Released under the MIT license
# https://github.com/miguelgrinberg/flask-video-streaming/blob/master/LICENSE
#
###############################################################################

from flask import Flask, Response, render_template, request, jsonify
from libs.camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import cv2
import json
from libs.interactive_detection import Detections
from libs.argparser import build_argparser

app = Flask(__name__)
logger = getLogger(__name__)

is_async = True
is_object_det = True
is_face_det = False
is_age_gender = False
is_emotions = False
is_head_pose = False
is_facial_landmarks = False
flip_code = None  # filpcode: 0,x-axis 1,y-axis -1,both axis
resize_width = 1280


def gen(camera):
    frame_id = 0
    while True:
        frame_id += 1
        frame = camera.get_frame(flip_code)
        if frame is None:
            logger.info("video finished. exit...")
            os._exit(0)
        if is_object_det:
            frame = detections.object_detection(
                frame, is_async, args.prob_threshold)
        else:
            frame = detections.face_detection(
                frame, is_async, args.prob_threshold_face, is_age_gender, is_emotions, is_head_pose, is_facial_landmarks)
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template(
        'index.html',
        is_async=is_async,
        flip_code=flip_code,
        is_object_det=is_object_det,
        devices=devices,
        models=models)


@app.route('/video_feed')
def video_feed():
    return Response(
        gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection', methods=['POST'])
def detection():
    global is_async
    global is_object_det
    global is_face_det
    global is_age_gender
    global is_emotions
    global is_head_pose
    global is_facial_landmarks

    command = request.json['command']
    if command == "async":
        is_async = True
    elif command == "sync":
        is_async = False

    if command == "object_det":
        is_object_det = True
        is_face_det = False
    if command == "face_det":
        is_face_det = True
        is_object_det = False
    if command == "age_gender" and not is_object_det:
        is_age_gender = not is_age_gender
    if command == "emotions" and not is_object_det:
        is_emotions = not is_emotions
    if command == "head_pose" and not is_object_det:
        is_head_pose = not is_head_pose
    if command == "facial_landmarks" and not is_object_det:
        is_facial_landmarks = not is_facial_landmarks

    result = {
        "command": "is_async",
        "is_async": is_async,
        "flip_code": flip_code,
        "is_object_det": is_object_det,
        "is_face_det": is_face_det,
        "is_age_gender": is_age_gender,
        "is_emotions": is_emotions,
        "is_head_pose": is_head_pose,
        "is_facial_landmarks": is_facial_landmarks
    }
    logger.info(
        "command:{} is_async:{} flip_code:{} is_obj_det:{} is_face_det:{} is_ag_det:{} is_em_det:{} is_hp_det:{} is_lm_det:{}".
        format(command, is_async, flip_code, is_object_det,
               is_face_det, is_age_gender,
               is_emotions, is_head_pose,
               is_facial_landmarks))
    return jsonify(ResultSet=json.dumps(result))


@app.route('/flip', methods=['POST'])
def flip_frame():
    global flip_code

    command = request.json['command']

    if command == "flip" and flip_code is None:
        flip_code = 0
    elif command == "flip" and flip_code == 0:
        flip_code = 1
    elif command == "flip" and flip_code == 1:
        flip_code = -1
    elif command == "flip" and flip_code == -1:
        flip_code = None

    result = {
        "command": "is_async",
        "is_async": is_async,
        "flip_code": flip_code,
        "is_object_det": is_object_det,
        "is_face_det": is_face_det,
        "is_age_gender": is_age_gender,
        "is_emotions": is_emotions,
        "is_head_pose": is_head_pose,
        "is_facial_landmarks": is_facial_landmarks
    }
    logger.info(
        f"command:{command} is_async:{is_async} flip_code:{flip_code} is_obj_det:{is_object_det} is_face_det:{is_face_det} is_ag:{is_age_gender} is_em:{is_emotions} is_hp:{is_head_pose} is_lm:{is_facial_landmarks}")
    return jsonify(ResultSet=json.dumps(result))


if __name__ == '__main__':

    # arg parse
    args = build_argparser().parse_args()

    # logging
    level = INFO
    if args.verbose:
        level = DEBUG

    basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s",
    )

    devices = [
        args.device, args.device, args.device_age_gender, args.device_emotions,
        args.device_head_pose, args.device_facial_landmarks
    ]

    models = [
        args.model_ssd, args.model_face, args.model_age_gender,
        args.model_emotions, args.model_head_pose, args.model_facial_landmarks
    ]

    camera = VideoCamera(args.input, resize_width, args.v4l)
    logger.info(
        f"input:{args.input} v4l:{args.v4l} frame shape: {camera.frame.shape}"
    )
    detections = Detections(camera.frame, devices)

    app.run(host='0.0.0.0', threaded=True)
