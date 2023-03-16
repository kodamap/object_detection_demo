import cv2
import numpy as np
from logging import getLogger

from openvino.runtime import Core
import openvino.runtime as ov
from openvino.runtime import get_version
from timeit import default_timer as timer

logger = getLogger(__name__)


class BaseDetection(object):
    def __init__(self, device, model_xml, detection_of):

        ie = Core()
        # read the network and corresponding weights from file
        model = ie.read_model(model=model_xml)
        self.input_layer_ir = model.input(0)
        # compile the model for the CPU (you can choose manually CPU, GPU, MYRIAD etc.)
        # or let the engine choose the best available device (AUTO)
        self.compiled_model = ie.compile_model(model=model, device_name=device)
        self.input_layer = self.compiled_model.inputs
        self.output_layers = self.compiled_model.outputs

        logger.info(
            f"Loading {device} model to the {detection_of} ... version:{get_version()}"
        )

    def preprocess(self, frame, shape):
        """
         Define the preprocess function for input data

        :param: image: the orignal input frame
        :returns:
                resized_image: the image processed
        """
        resized_frame = cv2.resize(frame, shape)
        resized_frame = cv2.cvtColor(
            np.array(resized_frame), cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame.transpose((2, 0, 1))
        resized_frame = np.expand_dims(
            resized_frame, axis=0).astype(np.float32)

        return resized_frame


class MobilenetSSD(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "mobilenet-ssd"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        # Create 2 infer requests
        self.curr_request = self.compiled_model.create_infer_request()
        self.next_request = self.compiled_model.create_infer_request()

    def process_results(self, frame, results, prob_threshold):
        """
        Ref: object detection
        https://github.com/openvinotoolkit/openvino_notebooks/blob/2022.1/notebooks/401-object-detection-webcam/401-object-detection.ipynb
        """
        # The size of the original frame.
        h, w = frame.shape[:2]
        # The 'results' variable is a [1, 1, 200, 7] tensor.
        results = results.squeeze()
        boxes, labels, scores = [], [], []
        for _, label, score, xmin, ymin, xmax, ymax in results:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                tuple(
                    map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
            )
            labels.append(int(label))
            scores.append(float(score))

        # Apply non-maximum suppression to get rid of many overlapping entities.
        # See https://paperswithcode.com/method/non-maximum-suppression
        # This algorithm returns indices of objects to keep.
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes, scores=scores, score_threshold=prob_threshold, nms_threshold=0.6
        )

        # If there are no boxes.
        if len(indices) == 0:
            return []

        # Filter detected objects.
        return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

    def infer(self, frame, next_frame, is_async):
        """
        Ref: async api
        https://github.com/openvinotoolkit/openvino_notebooks/blob/2022.1/notebooks/115-async-api/115-async-api.ipynb
        """

        if is_async:
            resized_frame = self.preprocess(next_frame, self.shape)
            self.next_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the "next" inference request
            self.next_request.start_async()
        else:
            self.curr_request.wait_for(-1)
            resized_frame = self.preprocess(frame, self.shape)
            self.curr_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the current inference request
            self.curr_request.start_async()

    def get_results(self, frame, is_async, prob_threshold):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """
        boxes = None
        if self.curr_request.wait_for(-1) == 1:
            results = self.curr_request.get_output_tensor(0).data
            # Get poses from network results.
            boxes = self.process_results(frame,
                                         results=results, prob_threshold=prob_threshold)

        if is_async:
            self.curr_request, self.next_request = self.next_request, self.curr_request

        return boxes


class FaceDetection(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Face Detection"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        # Create 2 infer requests
        self.curr_request = self.compiled_model.create_infer_request()
        self.next_request = self.compiled_model.create_infer_request()

    def infer(self, frame, next_frame, is_async):
        """
        Ref: async api
        https://github.com/openvinotoolkit/openvino_notebooks/blob/2022.1/notebooks/115-async-api/115-async-api.ipynb
        """

        if is_async:
            resized_frame = self.preprocess(next_frame, self.shape)
            self.next_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the "next" inference request
            self.next_request.start_async()

        else:
            self.curr_request.wait_for(-1)
            resized_frame = self.preprocess(frame, self.shape)
            self.curr_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the current inference request
            self.curr_request.start_async()

    def get_results(self, is_async, prob_threshold):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """

        faces = None
        if self.curr_request.wait_for(-1) == 1:
            results = self.curr_request.get_output_tensor(0).data
            # Get rows whose confidence is larger than prob_threshold.
            # detected faces are also used by age/gender, emotion, landmark, head pose detection.
            faces = results[0][0][np.where(
                results[0][0][:, 2] > prob_threshold)]

        if is_async:
            self.curr_request, self.next_request = self.next_request, self.curr_request

        return faces


class AgeGender(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Age/Gender"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        self.label = ('F', 'M')
        self.request = self.compiled_model.create_infer_request()

    def infer(self, face_frame):
        resized_frame = self.preprocess(face_frame, self.shape)
        self.request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.request.infer()

    def get_results(self):
        """
        Output layer names in Inference Engine format:
         "age_conv3", shape: [1, 1, 1, 1] - Estimated age divided by 100.
         "prob", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male]
        """
        age = 0
        gender = ""
        # Age
        results = self.request.get_output_tensor(1).data
        age = results.squeeze() * 100
        # Gender
        results = self.request.get_output_tensor(0).data
        gender = self.label[np.argmax(results.squeeze())]
        return age, gender


class Emotions(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Emotion"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        self.request = self.compiled_model.create_infer_request()

    def infer(self, face_frame):
        resized_frame = self.preprocess(face_frame, self.shape)
        self.request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.request.infer()

    def get_results(self):
        """
        Output layer names in Inference Engine format:
         "prob_emotion", shape: [1, 5, 1, 1]
         - Softmax output across five emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').
        """

        emotion = ""
        results = self.request.get_output_tensor(0).data
        emotion = self.label[np.argmax(results.squeeze())]
        prob = np.max(results.squeeze())

        return emotion, prob


class HeadPose(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Head Pose"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        self.request = self.compiled_model.create_infer_request()

    def infer(self, face_frame):
        resized_frame = self.preprocess(face_frame, self.shape)
        self.request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.request.infer()

    def get_results(self):
        """
        Output layer names in Inference Engine format:
         "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
         "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
         "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        Each output contains one float value that represents value in 
        Tait-Bryan angles (yaw, pitсh or roll).
        """

        yaw = .0  # Axis of rotation: z
        pitch = .0  # Axis of rotation: y
        roll = .0  # Axis of rotation: x

        # Get result
        # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
        # roll:  Axis of rotation: z
        fc_r = self.request.get_output_tensor(0).data
        roll = fc_r.squeeze()
        # pitch: Axis of rotation: x
        fc_p = self.request.get_output_tensor(1).data
        pitch = fc_p.squeeze()
        # yaw:   Axis of rotation: y
        fc_y = self.request.get_output_tensor(2).data
        yaw = fc_y.squeeze()

        return yaw, pitch, roll


class FacialLandmarks(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Facial Landmarks"
        super().__init__(device, model_xml, detection_of)

        # get input node
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        self.request = self.compiled_model.create_infer_request()

    def infer(self, face_frame):
        resized_frame = self.preprocess(face_frame, self.shape)
        self.request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.request.infer()

    def get_results(self):
        """
        # Output layer names in Inference Engine format:
        # landmarks-regression-retail-0009:
        #   "95", [1, 10, 1, 1], containing a row-vector of 10 floating point values for five landmarks
        #         coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        #         All the coordinates are normalized to be in range [0,1]
        # facial-landmarks-35-adas-0001:
        #   "align_fc3", [1,70], the shape: [1, 70], containing row-vector of 70 floating point values for 35 landmarks'
        #    normed coordinates in the form (x0, y0, x1, y1, ..., x34, y34).
        """

        normed_landmarks = np.zeros(0)
        results = self.request.get_output_tensor(0).data
        normed_landmarks = results.squeeze()

        return normed_landmarks
