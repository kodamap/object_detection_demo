# Object Detection MobileNet-SSD on NCS2 with Raspberry Pi

## What's this

This is Object detection(MobileNet-SSD) and Face detection demo on NCS2 with Raspberry Pi.

You can do followings:
* Object Detection(MobileNet-SSD)
* Face Detection and Face analytics (Age/Gender, Emotion, Head Pose, Facial Landmarks)
* Switch inference mode (async/sync)
* Flip frame (x-axis, y-axis, both-axis)

Youtube Link

[![](https://img.youtube.com/vi/kVPuSOPvY3U/0.jpg)](https://www.youtube.com/watch?v=kVPuSOPvY3U)

**Note:**
Most of these codes are based on OpenVINO python sample code "object_detection_demo_ssd_async.py"
and [Object detection with deep learning and OpenCV]( https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/) 

## Reference

Object detection with deep learning and OpenCV
https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

Use the Deep Learning Recognition Models in the Intel® Distribution of OpenVINO™ Toolkit
https://software.intel.com/en-us/articles/use-the-deep-learning-recognition-models-in-the-intel-distribution-of-openvino-toolkit

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

1. Install OpenVINO Toolkit on Raspberry Pi following articles bellows.
You need Raspbian* 9 OS (Stretch).

* Install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian* OS

https://software.intel.com/en-us/articles/OpenVINO-Install-Raspberrypi

2. Download MobileNet-SSD IR files from google drive.

* URL: https://drive.google.com/open?id=1YKbwy9W0MZObls9dy_0n90MQoRq0RdOB
  * File extension.zip
  * Size: 32,084,333 byte
  * MD5 hash : 31d7c77ade31fd1cb9cca6c9a92128f3

* Extract extension.zip and store extension folder under the "object_detection_demo"

```sh
extension/cpu_extension.dll
extension/IR/FP16/MobileNetSSD_deploy.bin
extension/IR/FP16/MobileNetSSD_deploy.mapping
extension/IR/FP16/MobileNetSSD_deploy.xml
extension/IR/FP32/MobileNetSSD_deploy.bin
extension/IR/FP32/MobileNetSSD_deploy.mapping
extension/IR/FP32/MobileNetSSD_deploy.xml
```

3. Download Face detection models IR files

```sh
cd extension/IR/FP16/
models="face-detection-retail-0004 age-gender-recognition-retail-0013 emotions-recognition-retail-0003 head-pose-estimation-adas-0001 landmarks-regression-retail-0009"
for model in $models
do
wget --no-check-certificate https://download.01.org/openvinotoolkit/2018_R5/open_model_zoo/${model}/FP16/${model}.xml
wget --no-check-certificate https://download.01.org/openvinotoolkit/2018_R5/open_model_zoo/${model}/FP16/${model}.bin
done
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
usage: app.py [-h] -i INPUT [-m_ss MODEL_SSD] [-m_fc MODEL_FACE]
              [-m_ag MODEL_AGE_GENDER] [-m_em MODEL_EMOTIONS]
              [-m_hp MODEL_HEAD_POSE] [-m_lm MODEL_FACIAL_LANDMARKS]
              [-l CPU_EXTENSION] [-d {CPU,GPU,FPGA,MYRIAD}]
              [-d_ag {CPU,GPU,FPGA,MYRIAD}] [-d_em {CPU,GPU,FPGA,MYRIAD}]
              [-d_hp {CPU,GPU,FPGA,MYRIAD}] [-d_lm {CPU,GPU,FPGA,MYRIAD}]
              [-pp PLUGIN_DIR] [--labels LABELS] [-pt PROB_THRESHOLD]
              [-ptf PROB_THRESHOLD_FACE] [--no_v4l]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video
                        stream from camera
  -m_ss MODEL_SSD, --model_ssd MODEL_SSD
                        Required. Path to an .xml file with a trained
                        MobileNet-SSD model.
  -m_fc MODEL_FACE, --model_face MODEL_FACE
                        Optional. Path to an .xml file with a trained
                        Age/Gender Recognition model.
  -m_ag MODEL_AGE_GENDER, --model_age_gender MODEL_AGE_GENDER
                        Optional. Path to an .xml file with a trained
                        Age/Gender Recognition model.
  -m_em MODEL_EMOTIONS, --model_emotions MODEL_EMOTIONS
                        Optional. Path to an .xml file with a trained Emotions
                        Recognition model.
  -m_hp MODEL_HEAD_POSE, --model_head_pose MODEL_HEAD_POSE
                        Optional. Path to an .xml file with a trained Head
                        Pose Estimation model.
  -m_lm MODEL_FACIAL_LANDMARKS, --model_facial_landmarks MODEL_FACIAL_LANDMARKS
                        Optional. Path to an .xml file with a trained Facial
                        Landmarks Estimation model.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
  -d {CPU,GPU,FPGA,MYRIAD}, --device {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for MobileNet-SSSD / Face
                        Detection to infer on; CPU, GPU, FPGA or MYRIAD is
                        acceptable.
  -d_ag {CPU,GPU,FPGA,MYRIAD}, --device_age_gender {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for Age/Gender Recognition
                        to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.
  -d_em {CPU,GPU,FPGA,MYRIAD}, --device_emotions {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for for Emotions Recognition
                        to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.
  -d_hp {CPU,GPU,FPGA,MYRIAD}, --device_head_pose {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for Head Pose Estimation to
                        infer on; CPU, GPU, FPGA or MYRIAD is acceptable.
  -d_lm {CPU,GPU,FPGA,MYRIAD}, --device_facial_landmarks {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for Facial Landmarks
                        Estimation to infer on; CPU, GPU, FPGA or MYRIAD is
                        acceptable.
  -pp PLUGIN_DIR, --plugin_dir PLUGIN_DIR
                        Path to a plugin folder
  --labels LABELS       Labels mapping file
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for object detections filtering
  -ptf PROB_THRESHOLD_FACE, --prob_threshold_face PROB_THRESHOLD_FACE
                        Probability threshold for face detections filtering
  --no_v4l              cv2.VideoCapture without cv2.CAP_V4L
```

Run app

* Specify **MYRIAD** with "-d(device)" option.

```sh
$ python3 app.py -i cam -d MYRIAD -d_em MYRIAD -d_ag MYRIAD -d_hp MYRIAD -d_lm MYRIAD
```

access to the streaming url with your browser

```txt
http://<your raspberryPi ip addr>:5000/
```

## Misc

Test with PC(Windows10)

* Specify cpu_extension.dll with "-l" option.
* Select FP32 IR model.
* You might need to add "--no_v4l" option.

```sh
> python app.py -i cam -l extension\cpu_extension.dll --no_v4l
```
