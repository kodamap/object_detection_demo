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

# Sample Usage:
# python3 app.py -i cam -m IR/MobileNetSSD_FP16/MobileNetSSD_deploy.xml -d MYRIAD

from flask import Flask, Response, render_template, request, jsonify
from camera import VideoCamera
from argparse import ArgumentParser
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import json

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_async_mode = True
#filpcode: 
#  0 : flipping around x-axis
#  1 : flipping around y-axis
# -1 : flipping around both axis
flip_code = "reset"


# construct the argument parse and parse the arguments
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Path to an .xml file with a trained model.",
        required=True,
        type=str)
    parser.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str)
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
        "impl.",
        type=str,
        default=None)
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
        "will look for a suitable plugin for device specified (CPU by default)",
        default="CPU",
        type=str)
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        help="Probability threshold for detections filtering",
        default=0.2,
        type=float)
    parser.add_argument(
        '--no_v4l',
        help='cv2.VideoCapture without cv2.CAP_V4L',
        action='store_true')

    return parser


def gen(camera):
    while True:
        frame = camera.get_frame(is_async_mode, flip_code)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template(
        'index.html', is_async_mode=is_async_mode, flip_code=flip_code)


@app.route('/video_feed')
def video_feed():
    camera = VideoCamera(input, model_xml, model_bin, device, prob_threshold,
                         cpu_extention, is_async_mode, no_v4l)
    return Response(
        gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection', methods=['POST'])
def detection():
    global is_async_mode
    global flip_code

    command = request.json['command']
    if command == "async":
        is_async_mode = True
    elif command == "sync":
        is_async_mode = False

    if command == "flip-x":
        flip_code = "0"
    elif command == "flip-y":
        flip_code = "1"
    elif command == "flip-xy":
        flip_code = "-1"
    elif command == "flip-reset":
        flip_code = "reset"

    result = {
        "command": "is_async_mode",
        "is_async_mode": is_async_mode,
        "flip_code": flip_code
    }
    logger.info("command:{} is_async:{} flip_code:{}".format(
        command, is_async_mode, flip_code))
    return jsonify(ResultSet=json.dumps(result))


if __name__ == '__main__':

    args = build_argparser().parse_args()
    input = args.input
    cpu_extention = args.cpu_extension
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    device = args.device
    prob_threshold = args.prob_threshold
    no_v4l = args.no_v4l

    if device == "CPU" and cpu_extention is None:
        print(
            "\nPlease try to specify cpu extensions library path in demo's command line parameters using -l "
            "or --cpu_extension command line argument")
        sys.exit(1)

    app.run(host='0.0.0.0', threaded=True)