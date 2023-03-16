# Object Detection MobileNet-SSD and Face analytics

## What's this

This is Object detection(MobileNet-SSD) and Face detection demo

You can do followings:
* Object Detection(MobileNet-SSD)
* Face Detection and Face analytics (Age/Gender, Emotion, Head Pose, Facial Landmarks)
* Switch inference mode (async/sync)
* Flip frame (x-axis, y-axis, both-axis)


**Object Detection and FaceDetection Demo - (YouTube Link)**

<a href="https://youtu.be/wfM4Vyteqmw">
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/118309/ef614609-d634-88cc-3dc7-f99abe5c2a02.gif" alt="TownCentre" width="%" height="auto"></a>

## Reference

- [Live Object Detection with OpenVINO™](https://github.com/openvinotoolkit/openvino_notebooks/blob/2022.1/notebooks/401-object-detection-webcam/401-object-detection.ipynb)

- [Flask Video streaming](http://blog.miguelgrinberg.com/post/video-streaming-with-flask)

## Environment

* Windows 11 
* Python 3.9.2
* OpenVINO Toolkit 2022.1 ~ 2022.3


## Prerequisite

Install OpenVINO Toolkit following articles bellows.

[Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

Packages

```sh
pip install -r requirements
```

## How to use

### Show help

```sh
$ python app.py -h
usage: app.py [-h] -i INPUT [-m_ss MODEL_SSD] [-m_fc MODEL_FACE] [-m_ag MODEL_AGE_GENDER] [-m_em MODEL_EMOTIONS]
              [-m_hp MODEL_HEAD_POSE] [-m_lm MODEL_FACIAL_LANDMARKS] [-d {CPU,GPU,MYRIAD,AUTO}]
              [-d_ag {CPU,GPU,MYRIAD,AUTO}] [-d_em {CPU,GPU,MYRIAD,AUTO}] [-d_hp {CPU,GPU,MYRIAD,AUTO}]
              [-d_lm {CPU,GPU,MYRIAD,AUTO}] [--labels LABELS] [-pt PROB_THRESHOLD] [-ptf PROB_THRESHOLD_FACE] [-v]
              [--v4l]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video stream from camera
  -m_ss MODEL_SSD, --model_ssd MODEL_SSD
                        Required. Path to an .xml file with a trained MobileNet-SSD model.
  -m_fc MODEL_FACE, --model_face MODEL_FACE
                        Optional. Path to an .xml file with a trained Age/Gender Recognition model.
  -m_ag MODEL_AGE_GENDER, --model_age_gender MODEL_AGE_GENDER
                        Optional. Path to an .xml file with a trained Age/Gender Recognition model.
  -m_em MODEL_EMOTIONS, --model_emotions MODEL_EMOTIONS
                        Optional. Path to an .xml file with a trained Emotions Recognition model.
  -m_hp MODEL_HEAD_POSE, --model_head_pose MODEL_HEAD_POSE
                        Optional. Path to an .xml file with a trained Head Pose Estimation model.
  -m_lm MODEL_FACIAL_LANDMARKS, --model_facial_landmarks MODEL_FACIAL_LANDMARKS
                        Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.
  -d {CPU,GPU,MYRIAD,AUTO}, --device {CPU,GPU,MYRIAD,AUTO}
                        Specify the target device for MobileNet-SSSD / Face Detection to infer on; CPU, GPU, MYRIAD or
                        AUTO is acceptable.
  -d_ag {CPU,GPU,MYRIAD,AUTO}, --device_age_gender {CPU,GPU,MYRIAD,AUTO}
                        Specify the target device for Age/Gender Recognition to infer on; CPU, GPU, MYRIAD or AUTO is
                        acceptable.
  -d_em {CPU,GPU,MYRIAD,AUTO}, --device_emotions {CPU,GPU,MYRIAD,AUTO}
                        Specify the target device for for Emotions Recognition to infer on; CPU, GPU, MYRIAD or AUTO is
                        acceptable.
  -d_hp {CPU,GPU,MYRIAD,AUTO}, --device_head_pose {CPU,GPU,MYRIAD,AUTO}
                        Specify the target device for Head Pose Estimation to infer on; CPU, GPU, MYRIAD or AUTO is
                        acceptable.
  -d_lm {CPU,GPU,MYRIAD,AUTO}, --device_facial_landmarks {CPU,GPU,MYRIAD,AUTO}
                        Specify the target device for Facial Landmarks Estimation to infer on; CPU, GPU, MYRIAD or AUTO
                        is acceptable.
  --labels LABELS       Labels mapping file
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for object detections filtering
  -ptf PROB_THRESHOLD_FACE, --prob_threshold_face PROB_THRESHOLD_FACE
                        Probability threshold for face detections filtering
  -v, --verbose         set logging level Debug
  --v4l                 cv2.VideoCapture with cv2.CAP_V4L
```

### Run app

test video: 

https://github.com/openvinotoolkit/openvino_notebooks/tree/2022.1/notebooks/202-vision-superresolution/data

```sh
python app.py -i ".\video\CEO Pat Gelsinger on Leading Intel.mp4"
```

### Access to the url on your browser

```txt
http://localhost:5000/
```

# Known Issue

## Only CPU works.

When using GPU

```sh
python app.py -i ".\video\CEO Pat Gelsinger on Leading Intel.mp4" -d GPU
```

Error :
```sh
File "openvino\object_detection_demo\libs\detectors.py", line 121, in get_results
    results = self.curr_request.get_output_tensor(0).data
RuntimeError: invalid vector subscript
```

When using MYRIAD

```sh
python app.py -i "\video\CEO Pat Gelsinger on Leading Intel.mp4" -d MYRIAD 
```

Error :
```sh
File "C:\Program Files (x86)\Intel\openvino_2022\python\python3.9\openvino\runtime\ie_api.py", line 266, in compile_model
    super().compile_model(model, device_name, {} if config is None else config)
RuntimeError: Failed to allocate graph: MYRIAD device is not opened.
```
