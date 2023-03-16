from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str)
    parser.add_argument(
        "-m_ss",
        "--model_ssd",
        help="Required. Path to an .xml file with a trained MobileNet-SSD model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_fc",
        "--model_face",
        help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_ag",
        "--model_age_gender",
        help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_em",
        "--model_emotions",
        help="Optional. Path to an .xml file with a trained Emotions Recognition model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_hp",
        "--model_head_pose",
        help="Optional. Path to an .xml file with a trained Head Pose Estimation model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_lm",
        "--model_facial_landmarks",
        help="Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.",
        type=str,
        default=None)
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device for MobileNet-SSSD / Face Detection to infer on; CPU, GPU, MYRIAD or AUTO is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
        type=str)
    parser.add_argument(
        "-d_ag",
        "--device_age_gender",
        help="Specify the target device for Age/Gender Recognition to infer on; CPU, GPU, MYRIAD or AUTO is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
        type=str)
    parser.add_argument(
        "-d_em",
        "--device_emotions",
        help="Specify the target device for for Emotions Recognition to infer on; CPU, GPU, MYRIAD or AUTO is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
        type=str)
    parser.add_argument(
        "-d_hp",
        "--device_head_pose",
        help="Specify the target device for Head Pose Estimation to infer on; CPU, GPU, MYRIAD or AUTO is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
        type=str)
    parser.add_argument(
        "-d_lm",
        "--device_facial_landmarks",
        help="Specify the target device for Facial Landmarks Estimation to infer on; CPU, GPU, MYRIAD or AUTO is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
        type=str)
    parser.add_argument(
        "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        help="Probability threshold for object detections filtering",
        default=0.3,
        type=float)
    parser.add_argument(
        "-ptf",
        "--prob_threshold_face",
        help="Probability threshold for face detections filtering",
        default=0.5,
        type=float)
    parser.add_argument(
        "-v", "--verbose", help="set logging level Debug", action="store_true"
    )
    parser.add_argument(
        '--v4l',
        help='cv2.VideoCapture with cv2.CAP_V4L',
        action='store_true')

    return parser
