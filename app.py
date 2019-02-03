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
from camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import json
import interactive_detection

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_async_mode = True
is_object_detection = True
is_face_detection = False
is_age_gender_detection = False
is_emotions_detection = False
is_head_pose_detection = False
is_facial_landmarks_detection = False
flip_code = None  # filpcode: 0,x-axis 1,y-axis -1,both axis


def gen(camera):
    while True:
        frame = camera.get_frame(is_async_mode, flip_code, is_object_detection,
                                 is_face_detection, is_age_gender_detection,
                                 is_emotions_detection, is_head_pose_detection,
                                 is_facial_landmarks_detection)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template(
        'index.html',
        is_async_mode=is_async_mode,
        flip_code=flip_code,
        is_object_detection=is_object_detection,
        devices=devices,
        models=models)


@app.route('/video_feed')
def video_feed():
    camera = VideoCamera(args.input, detections, args.no_v4l)
    return Response(
        gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection', methods=['POST'])
def detection():
    global is_async_mode
    global is_object_detection
    global is_face_detection
    global is_age_gender_detection
    global is_emotions_detection
    global is_head_pose_detection
    global is_facial_landmarks_detection

    command = request.json['command']
    if command == "async":
        is_async_mode = True
    elif command == "sync":
        is_async_mode = False

    if command == "object_detection":
        is_object_detection = True
        is_face_detection = False
    if command == "face_detection":
        is_face_detection = True
        is_object_detection = False
    if command == "age_gender_detection" and not is_object_detection:
        is_age_gender_detection = not is_age_gender_detection
    if command == "emotions_detection" and not is_object_detection:
        is_emotions_detection = not is_emotions_detection
    if command == "head_pose_detection" and not is_object_detection:
        is_head_pose_detection = not is_head_pose_detection
    if command == "facial_landmarks_detection" and not is_object_detection:
        is_facial_landmarks_detection = not is_facial_landmarks_detection

    result = {
        "command": "is_async_mode",
        "is_async_mode": is_async_mode,
        "flip_code": flip_code,
        "is_object_detection": is_object_detection,
        "is_face_detection": is_face_detection,
        "is_age_gender_detection": is_age_gender_detection,
        "is_emotions_detection": is_emotions_detection,
        "is_head_pose_detection": is_head_pose_detection,
        "is_facial_landmarks_detection": is_facial_landmarks_detection
    }
    logger.info(
        "command:{} is_async:{} flip_code:{} is_obj_det:{} is_face_det:{} is_ag_det:{} is_em_det:{} is_hp_det:{} is_lm_det:{}".
        format(command, is_async_mode, flip_code, is_object_detection,
               is_face_detection, is_age_gender_detection,
               is_emotions_detection, is_head_pose_detection,
               is_facial_landmarks_detection))
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
        "command": "is_async_mode",
        "is_async_mode": is_async_mode,
        "flip_code": flip_code,
        "is_object_detection": is_object_detection,
        "is_face_detection": is_face_detection,
        "is_age_gender_detection": is_age_gender_detection,
        "is_emotions_detection": is_emotions_detection,
        "is_head_pose_detection": is_head_pose_detection,
        "is_facial_landmarks_detection": is_facial_landmarks_detection
    }
    logger.info(
        "command:{} is_async:{} flip_code:{} is_obj_det:{} is_face_det:{} is_ag_det:{} is_em_det:{} is_hp_det:{} is_lm_det:{}".
        format(command, is_async_mode, flip_code, is_object_detection,
               is_face_detection, is_age_gender_detection,
               is_emotions_detection, is_head_pose_detection,
               is_facial_landmarks_detection))
    return jsonify(ResultSet=json.dumps(result))


if __name__ == '__main__':

    # arg parse
    args = interactive_detection.build_argparser().parse_args()
    devices = [
        args.device, args.device, args.device_age_gender, args.device_emotions,
        args.device_head_pose, args.device_facial_landmarks
    ]
    models = [
        args.model_ssd, args.model_face, args.model_age_gender,
        args.model_emotions, args.model_head_pose, args.model_facial_landmarks
    ]
    if "CPU" in devices and args.cpu_extension is None:
        print(
            "\nPlease try to specify cpu extensions library path in demo's command line parameters using -l "
            "or --cpu_extension command line argument")
        sys.exit(1)

    # Create detectors class instance
    detections = interactive_detection.Detections(
        devices, models, args.cpu_extension, args.plugin_dir,
        args.prob_threshold, args.prob_threshold_face, is_async_mode)
    models = detections.models  # Get models to display WebUI.

    app.run(host='0.0.0.0', threaded=True)
