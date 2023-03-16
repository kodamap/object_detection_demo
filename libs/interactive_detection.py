from logging import getLogger
import libs.detectors as detectors
import cv2
import math
import numpy as np
from timeit import default_timer as timer

logger = getLogger(__name__)

model_path = "model/intel"
model_ss = "mobilenet-ssd"
model_fc = "face-detection-retail-0004"
model_ag = "age-gender-recognition-retail-0013"
model_em = "emotions-recognition-retail-0003"
model_hp = "head-pose-estimation-adas-0001"
model_lm = "facial-landmarks-35-adas-0002"

# https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/voc_20cl_bkgr.txt
CLASSES = ["background",
           "aeroplane",
           "bicycle",
           "bird",
           "boat",
           "bottle",
           "bus",
           "car",
           "cat",
           "chair",
           "cow",
           "diningtable",
           "dog",
           "horse",
           "motorbike",
           "person",
           "pottedplant",
           "sheep",
           "sofa",
           "train",
           "tvmonitor"
           ]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class Detectors(object):

    def __init__(self, devices):
        self.device_ss, self.device_fc, self.device_ag, self.device_em, self.device_hp, self.device_lm = devices
        self._define_models()
        self._load_detectors()

    def _define_models(self):

        # mobilenet ssd
        fp_path = "FP16" if self.device_ss == "CPU" else "FP16"
        self.model_ss = f"{model_path}/{model_ss}/{fp_path}/{model_ss}.xml"
        # face detection
        fp_path = "FP16-INT8" if self.device_fc == "CPU" else "FP16"
        self.model_fc = f"{model_path}/{model_fc}/{fp_path}/{model_fc}.xml"
        # age gender
        fp_path = "FP16-INT8" if self.device_ag == "CPU" else "FP16"
        self.model_ag = f"{model_path}/{model_ag}/{fp_path}/{model_ag}.xml"
        # emotions
        fp_path = "FP16-INT8" if self.device_em == "CPU" else "FP16"
        self.model_em = f"{model_path}/{model_em}/{fp_path}/{model_em}.xml"
        # head pose
        fp_path = "FP16-INT8" if self.device_hp == "CPU" else "FP16"
        self.model_hp = f"{model_path}/{model_hp}/{fp_path}/{model_hp}.xml"
        # landmarks
        fp_path = "FP16-INT8" if self.device_lm == "CPU" else "FP16"
        self.model_lm = f"{model_path}/{model_lm}/{fp_path}/{model_lm}.xml"

        self.models = [self.model_ss, self.model_fc, self.model_ag,
                       self.model_em, self.model_hp, self.model_lm]

    def _load_detectors(self):

        # Create MobileNet-SSD detection class instance
        self.mobilenet_ssd = detectors.MobilenetSSD(
            self.device_ss, self.model_ss)
        # Create face_detection class instance
        self.face_detector = detectors.FaceDetection(
            self.device_fc, self.model_fc)
        # Create face_analytics class instances
        self.age_gender = detectors.AgeGender(self.device_ag, self.model_ag)
        self.emotions = detectors.Emotions(self.device_em, self.model_em)
        self.headpose = detectors.HeadPose(self.device_hp, self.model_hp)
        self.facial_landmarks = detectors.FacialLandmarks(
            self.device_lm, self.model_lm)


class Detections(Detectors):
    def __init__(self, frame, devices):
        super().__init__(devices)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()
        self.prev_frame = frame

    def _calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps += 1

        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

    def draw_perf_stats(self, det_time, det_time_txt, frame, is_async):

        # Draw FPS in top left corner
        self._calc_fps()
        cv2.rectangle(frame, (frame.shape[1] - 50, 0), (frame.shape[1], 17),
                      (255, 255, 255), -1)
        cv2.putText(frame, self.fps, (frame.shape[1] - 50 + 3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw performance stats
        if is_async:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for async mode"
            )
        else:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for sync mode"
            )
        cv2.putText(
            frame,
            inf_time_message,
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (135, 206, 235),
            1,
        )

        if det_time_txt:
            inf_time_message = (f"Detection time: {det_time_txt}")
            cv2.putText(
                frame,
                inf_time_message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (135, 206, 235),
                1,
            )
        return frame

    def draw_object_boxes(self, frame, boxes):
        for label, score, box in boxes:
            # Choose color for the label.
            color = tuple(map(int, COLORS[label]))
            # Draw a box.
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]
            cv2.rectangle(img=frame, pt1=box[:2], pt2=(
                x2, y2), color=color, thickness=3)

            # Draw a label name inside the box.
            cv2.putText(
                img=frame,
                text=f"{CLASSES[label]} {score:.2f}",
                org=(box[0] + 10, box[1] + 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=frame.shape[1] / 1000,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix

    def draw_headpose_axes(self, frame, center_of_face, yaw, pitch, roll, scale):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])

        Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx  # R = np.dot(Rz, np.dot(Ry, Rx))

        camera_matrix = self.build_camera_matrix(center_of_face, 950.0)

        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o

        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)

        return frame

    def object_detection(self, frame, is_async, prob_threshold):
        # ----------- Start Face Detection ---------- #
        logger.debug("** face_detection start **")

        det_time = 0
        det_time_txt = ""
        prev_frame = None

        if is_async:
            prev_frame = frame.copy()
        else:
            self.prev_frame = frame.copy()

        # Do inference.
        inf_start = timer()
        self.mobilenet_ssd.infer(self.prev_frame, frame, is_async)
        inf_end = timer()
        det_time = inf_end - inf_start

        # Get boxes from network results.
        boxes = self.mobilenet_ssd.get_results(frame, is_async, prob_threshold)

        if boxes is None:
            return self.prev_frame

        det_time_txt = f"object detection:{det_time * 1000:.3f} ms "

        # Draw boxes on a frame.
        self.draw_object_boxes(frame=self.prev_frame, boxes=boxes)
        frame = self.draw_perf_stats(
            det_time, det_time_txt, self.prev_frame, is_async)

        if is_async:
            self.prev_frame = prev_frame

        return frame

    def face_detection(self, frame, is_async, prob_threshold_face,
                       is_age_gender, is_emotions,
                       is_head_pose, is_facial_landmarks):

        # ----------- Start Face Detection ---------- #
        logger.debug("** face_detection start **")
        color = (0, 255, 0)
        det_time = 0
        det_time_fc = 0
        det_time_txt = ""
        prev_frame = None

        if is_async:
            prev_frame = frame.copy()
        else:
            self.prev_frame = frame.copy()

        frame_h, frame_w = frame.shape[:2]  # shape (h, w, c)
        is_face_analytics = True if is_age_gender or is_emotions else False

        inf_start = timer()
        self.face_detector.infer(self.prev_frame, prev_frame, is_async)
        faces = self.face_detector.get_results(is_async, prob_threshold_face)
        inf_end = timer()
        det_time = inf_end - inf_start
        det_time_fc = det_time

        if faces is None:
            return self.prev_frame

        det_time_txt = f"face_cnt:{faces.shape[0]} face:{det_time * 1000:.3f} ms "

        # ----------- Start Face Analytics ---------- #

        face_w, face_h = 0, 0
        face_frame = None
        det_time_ag = 0
        det_time_em = 0
        det_time_hp = 0
        det_time_lm = 0

        for i, face in enumerate(faces):
            face_analytics = ""
            age_gender = ""
            emotion = ""

            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")
            # class_id = int(face[1])
            result = str(i) + " " + str(round(face[2] * 100, 1)) + '% '

            if xmin < 0 or ymin < 0:
                logger.info(
                    f"Rapid motion returns negative value(xmin and ymin) which make face_frame None. xmin:{xmin} xmax:{xmax} ymin:{ymin} ymax:{ymax}")
                return frame

            # Start face analytics
            # prev_box is previous box(faces), which is None at the first time
            # will be updated with prev face box in async mode
            face_frame = frame[ymin:ymax, xmin:xmax]

            # Check face frame.
            # face_fame is None at the first time with async mode.
            if face_frame is not None:
                face_w, face_h = face_frame.shape[:2]
                # Resizing face_frame will be failed when witdh or height of the face_fame is 0 ex. (243, 0, 3)
                if face_w == 0 or face_h == 0:
                    logger.info(
                        f"Unexpected shape of face frame. face_frame.shape:{face_h} {face_w}")
                    return frame

            # ----------- Start Age/Gender detection ---------- #
            if is_age_gender:
                inf_start = timer()
                self.age_gender.infer(face_frame)
                age, gender = self.age_gender.get_results()
                age_gender = str(int(round(age))) + " " + gender + " "
                inf_end = timer()
                det_time = inf_end - inf_start
                det_time_ag += det_time

            # ----------- Start Emotions detection ---------- #
            if is_emotions:
                inf_start = timer()
                self.emotions.infer(face_frame)
                emotion, _ = self.emotions.get_results()
                emotion = emotion + " "
                inf_end = timer()
                det_time = inf_end - inf_start
                det_time_em += det_time

            # ----------- Start Head Pose detection ---------- #
            if is_head_pose:
                inf_start = timer()
                self.headpose.infer(face_frame)
                yaw, pitch, roll = self.headpose.get_results()
                center_of_face = (xmin + face_h / 2, ymin + face_w / 2, 0)
                frame = self.draw_headpose_axes(
                    frame, center_of_face, yaw, pitch, roll, 50)
                inf_end = timer()
                det_time = inf_end - inf_start
                det_time_hp += det_time

            # ----------- Start facial landmarks detection ---------- #
            if is_facial_landmarks:
                inf_start = timer()
                self.facial_landmarks.infer(face_frame)
                normed_landmarks = self.facial_landmarks.get_results()
                n_lm = normed_landmarks.size
                for i in range(int(n_lm / 2)):
                    normed_x = normed_landmarks[2 * i]
                    normed_y = normed_landmarks[2 * i + 1]
                    x_lm = xmin + face_h * normed_x
                    y_lm = ymin + face_w * normed_y
                    cv2.circle(frame, (int(x_lm), int(y_lm)),
                               1 + int(0.012 * face_h), (0, 255, 255), -1)
                inf_end = timer()
                det_time = inf_end - inf_start
                det_time_lm += det_time

            # ---------- Draw results into frame ------------- #
            face_analytics = age_gender + emotion

            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin), color, -1)
            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin),
                          (255, 255, 255))
            # Draw box and label\class_id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
            if is_face_analytics:
                cv2.putText(frame, face_analytics, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            else:
                cv2.putText(frame, result, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            logger.debug(f"face_id:{i} confidence:{round(face[2] * 100)}%")

        det_time = det_time_fc + det_time_ag + det_time_em + det_time_hp + det_time_lm
        det_time_txt = det_time_txt + f"ag:{det_time_ag * 1000:.2f} "
        det_time_txt = det_time_txt + f"em:{det_time_em * 1000:.2f} "
        det_time_txt = det_time_txt + f"hp:{det_time_hp * 1000:.2f} "
        det_time_txt = det_time_txt + f"lm:{det_time_lm * 1000:.2f} "

        frame = self.draw_perf_stats(det_time, det_time_txt, frame,
                                     is_async)

        if is_async:
            self.prev_frame = prev_frame

        return frame
