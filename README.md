# Object Detection MobileNet-SSD on NCS2 with Raspberry Pi

## What's this

This is Object detection demo for MobileNet-SSD on NCS2 with Raspberry Pi.

You can do followings:
* Detection result streaming via browser.
* Change inference mode (asyn/sync)
* Rotate frame (x-axis, y-axis, bot-axis)

Youtube Link

[![](https://img.youtube.com/vi/Ey78julifqw/0.jpg)](https://www.youtube.com/watch?v=Ey78julifqw)

**Note:**
Most of these codes are based on OpenVINO python sample code "object_detection_demo_ssd_async.py"
and [Object detection with deep learning and OpenCV]( https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/) 

## Reference

Object detection with deep learning and OpenCV

https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

Intel® Neural Compute Stick 2

https://software.intel.com/en-us/neural-compute-stick

Flask Video streaming

http://blog.miguelgrinberg.com/post/video-streaming-with-flask

https://github.com/ECI-Robotics/opencv_remote_streaming_processing/

## Environment

* Raspberry Pi 3B
* Intel® Neural Compute Stick 2
* Raspbian Stretch with desktop  (2018-11-13-raspbian-stretch)
* OpenVINO Toolkit 2018 R5 (l_openvino_toolkit_ie_p_2018.5.445.tgz)

## Prerequisite

Install OpenVINO Toolkit on Raspberry Pi following articles bellows.
You need Raspbian* 9 OS (Stretch).

* Install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian* OS

https://software.intel.com/en-us/articles/OpenVINO-Install-Raspberrypi


## Required Packages

Install packages on Raspberry Pi

```sh
sudo apt install python3-pip -y
pip3 install flask
```

## How to use

Make sure to be enable picamera(rasp-config) and modprobe bcm2835-v412.

```sh
sudo modprobe bcm2835-v4l2
```

Show help
```sh
$ python3 app.py -h
usage: app.py [-h] -m MODEL -i INPUT [-d DEVICE] [-pt PROB_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video
                        stream from camera
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA
                        or MYRIAD is acceptable. Demo will look for a suitable
                        plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering
```

Run app

```sh
$ python3 app.py -i cam -m IR\MobileNetSSD_FP16\MobileNetSSD_deploy.xml -d MYRIAD
```

access to the streaming url with your browser

```txt
http://<your raspberryPi ip addr>:5000/
```
