# Object Detection MobileNet-SSD on NCS2 with Raspberry Pi

## What's this

This is Object detection demo for MobileNet-SSD on NCS2 with Raspberry Pi.

You can do followings:
* Detection result streaming via browser.
* Change inference mode (async/sync)
* Rotate frame (x-axis, y-axis, both-axis)

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

![DSC_5548.JPG](https://qiita-image-store.s3.amazonaws.com/0/118309/75f9765f-d7c8-2c7e-7bd3-5817496656de.jpeg)


## Prerequisite

Install OpenVINO Toolkit on Raspberry Pi following articles bellows.
You need Raspbian* 9 OS (Stretch).

* Install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian* OS

https://software.intel.com/en-us/articles/OpenVINO-Install-Raspberrypi

To download IR files and cpu extension dll from google drive.

URL: https://drive.google.com/open?id=18e4fhpPCBrJR-MVpAGcx-3NtFxWTcEGk
  * File extension.zip
  * Size: 32,084,489 byte
  * MD5 hash : 8e46de84d6ee0b61ff9224e7087e01e7

* Extract extension.zip and store extension folder under tellooo(file lists)
```sh
extension/cpu_extension.dll
extension/IR/MobileNetSSD_FP16/MobileNetSSD_deploy.bin
extension/IR/MobileNetSSD_FP16/MobileNetSSD_deploy.mapping
extension/IR/MobileNetSSD_FP16/MobileNetSSD_deploy.xml
extension/IR/MobileNetSSD_FP32/MobileNetSSD_deploy.bin
extension/IR/MobileNetSSD_FP32/MobileNetSSD_deploy.mapping
extension/IR/MobileNetSSD_FP32/MobileNetSSD_deploy.xml
```



## Required Packages

Install packages on Raspberry Pi

```sh
sudo apt install python3-pip -y
pip3 install flask
```

## How to use

Make sure that picamera is enabled (rasp-config) and modprobe bcm2835-v412.

```sh
sudo modprobe bcm2835-v4l2
```

Show help

```sh
$ python3 app.py -h
usage: app.py [-h] -m MODEL -i INPUT [-l CPU_EXTENSION] [-d DEVICE]
              [-pt PROB_THRESHOLD] [--no_v4l]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video
                        stream from camera
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA
                        or MYRIAD is acceptable. Demo will look for a suitable
                        plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering
  --no_v4l              cv2.VideoCapture without cv2.CAP_V4L
```

Run app

```sh
$ python3 app.py -i cam -m IR\MobileNetSSD_FP16\MobileNetSSD_deploy.xml -d MYRIAD
```

access to the streaming url with your browser

```txt
http://<your raspberryPi ip addr>:5000/
```

## Misc

Test on PC(Windows10)

* specify cpu_extension.dll with "-l" option.

```sh
$ python app.py -i cam -l extension\cpu_extension.dll -m extension\IR\MobileNetSSD_FP32\MobileNetSSD_deploy.xml --no_v4l
```