# IMPORTS ----------------------------------------------------------------------------------------
import copy
import argparse

'''
IMPORTANT!!! READ HERE BEFORE YOU PROCEED WITH ANYTHING ELSE IN THIS FILE:
ENG: It has come to my attention that when you use Mediapipe Holistic to process a mirrored/flipped image,
it WILL RETURN the landmark of the opposite side of your body in general (Hands, Pose, Face).
This WON'T affect our trained models, since we kinda trained them using mirrored landmarks as well.

VIET: Nếu ta sử dụng Mediapipe Holistic để xử lý và lấy landmark từ hình đã được mirrored/flipped bằng cv2.flip,
Mediapipe sẽ trả về các landmark ngược bên với các bộ phận có thể nhận diện được (Hands, Pose, Face).
Việc này sẽ không làm ảnh hưởng tới mô hình được huấn luyện trước đó, vì ta đã luyện mô hình với tập dữ liệu
gồm các landmarks đã được mirrored.

EXAMPLE:  results = holistics.process(image)
- Left Hand will have results.right_hand_landmarks
- Right Hand will have results.left_hand_landmarks

- Right Shoulder will have PoseLandmarks.LEFT_SHOULDER landmarks (PoseLandmarks is imported from Mediapipe)
'''

'''
Since cv2.VideoCapture(0, CAP_MSMF) takes a REALLY LONG WHILE to start up (roughly 20s), I've decided to disable the HW Transforms.
This MUST be put before import cv2
'''
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

'''
IMPORTANT 2: App compiled with PyInstaller was having problems with Tensorflow logging. This occurs when:
 App is built with [noconsole / windowed] and tensorflow's logging is naively assuming that sys.stdout and sys.stderr are available, 
 but in fact they are None.
 
 One of the fixes for this is to add this snippet at the start of the app, like so:
 (If sys.stdout or sys.stderr are not available --> set them to os.devnull objects.)
 This is the most surefire way to counter this problem I think.
'''
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
'''
 The other way is to "Change the "verbose" argument of the predict function to 0".
 The predict function is in keras_classifier.py file
'''

import cv2 as cv
import numpy as np
import mediapipe as mp

# For fast append and pop in list, and to replace older elements in a list
# This is used in the "predictions generalization list"
# ELI5-VN: List được dùng để lấy prediction xuất hiện nhiều nhất trong vòng 10 predictions,
# nhằm loại bỏ các predictions nhỏ lẻ (có độ chính xác thấp) trong 1 chuỗi gesture
from collections import deque

# To check for most common prediction out of 3/5/10 predictions
from collections import Counter

# FPS Class
from utils import CvFpsCalc

# Steering Wheel Class
from utils import SteeringWheel

# KERAS model class
from models import KerasGestureClassifier
# from models.keras_gesture_classifier import KerasGestureClassifier

# Socket Import
import socket

# Defining socket attributes ----------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 27001
# This app will act as a Client so it doesn't need to bind itself with the IP and PORT
# We will only send data/messages to the specified IP and PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# UDP DATA Processing --------------------------------------------------------------------------
# Label/class is already a string, Angle is actually a float value rounded to 2 decimals
def process_udp_data(label, angle):
    processed_data = label + '$' + str(angle)
    return processed_data


# Sending data through UDP
def send_data(sock, UDP_IP, UDP_PORT, data, debug_var=True):
    sock.sendto(data.encode("UTF-8"), (UDP_IP, UDP_PORT))
    if (debug_var):
        print("Data: ", data, " sent!", "_" * 10)
        print(data.split('$'))

# MODIFYING Mediapipe Holistics for capturing Shoulder + Arms + Hands only ------------------------
from mediapipe.python.solutions.holistic import PoseLandmark

mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# List of landmarks to include on the image
included_landmarks = [
    # right hand set
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,

    # left hand set
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    # thumb, index and pinky are not required, since they are included in hand landmarks
]

# Actions list (List of gesture strings, we'll be using this later)
actions = np.array(["Idle", "StaticStraight", "LSteer", "RSteer", "Boost", "Brake", "BrakeHold", "Reverse", "None"])
NONE_ACTION_IDX = 8  # Yeah I made this up, purely for when there's nothing happening - "None" action

# Specifying FILE PATHS ------------------------------------------------------------------------
full_path = os.path.realpath(__file__)
dir_path = os.path.dirname(full_path)
current_model_path = os.path.join(dir_path, "models", "7-lstm_model_7.keras")


# Getting arguments ----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=720)  # 540

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument('--unuse_smooth_landmarks', action='store_true',
                        default=True)
    parser.add_argument('--enable_segmentation', action='store_true',
                        default=False)
    parser.add_argument('--smooth_segmentation', action='store_true',
                        default=False)
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true',
                        default=False)

    args = parser.parse_args()

    return args


# Holistics Visualization ----------------------------------------------------------------------
# Hand Landmark Visualization
def draw_hand_v2(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    if hand_landmarks:
        for index, landmark in enumerate(hand_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))

            if index == 0:  # 手首1
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # 手首2
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

            # if not upper_body_only:
            # if True:
            #     cv.putText(image, "z:" + str(round(landmark_z, 3)),
            #                (landmark_x - 10, landmark_y - 10),
            #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            #                cv.LINE_AA)
            #     cv.putText(image, "x:" + str(round(landmark_x, 3)),
            #                (landmark_x - 10, landmark_y - 10),
            #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            #                cv.LINE_AA)
            #     cv.putText(image, "y:" + str(round(landmark_y, 3)),
            #                (landmark_x - 10, landmark_y - 10),
            #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            #                cv.LINE_AA)

        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
            cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

            # 人差指
            cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
            cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
            cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

            # 中指
            cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
            cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
            cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

            # 薬指
            cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
            cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
            cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

            # 小指
            cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
            cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
            cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

            # 手の平
            cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
            cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
            cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
            cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
            cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
            cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
            cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    return image


# Shoulders + Arms
def draw_pose_landmarks_v2(
        image,
        landmarks,
        # upper_body_only,
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    if landmarks:
        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

            if landmark.visibility < visibility_th:
                continue

            #
            if index in included_landmarks:
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                # cv.putText(image, "z:" + str(round(landmark_z, 3)),
                #            (landmark_x - 10, landmark_y - 10),
                #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                #            cv.LINE_AA)

        if len(landmark_point) > 0:
            # 肩 - Shoulders
            if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[12][1],
                        (0, 255, 0), 2)

            # 右腕 - Right Arm
            if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[13][1],
                        (0, 255, 0), 2)
            if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
                cv.line(image, landmark_point[13][1], landmark_point[15][1],
                        (0, 255, 0), 2)

            # 左腕 - Left Arm
            if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
                cv.line(image, landmark_point[12][1], landmark_point[14][1],
                        (0, 255, 0), 2)
            if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
                cv.line(image, landmark_point[14][1], landmark_point[16][1],
                        (0, 255, 0), 2)

    return image


# Rect Drawing
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


# LANDMARK EXTRACTION method -------------------------------------------------------------------
'''
# We will only be taking the landmarks of Shoulders + Arms + Wrists + 2 Hands
# AND this time, the coords are shifted with Left Shoulder (11) being the base (0, 0, z)
# Yes, all z coords (and vis) are still intact. Since z coords aren't really affected by x and y.
'''


def extract_keypoints_v3(results):
    # Left Shoulder coord
    try:
        LeftSh_x = results.pose_landmarks.landmark[11].x
        LeftSh_y = results.pose_landmarks.landmark[11].y
    except:
        LeftSh_x = 0
        LeftSh_y = 0

    pose = []
    lh = []
    rh = []
    # Pose Landmarks
    if results.pose_landmarks:
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            if index in included_landmarks:
                pose = np.append(pose, [landmark.x - LeftSh_x, landmark.y - LeftSh_y, landmark.z, landmark.visibility])
                # This will be flattened upon appending
    else:
        pose = np.zeros(6 * 4)
    # Left Hand Landmarks
    if results.left_hand_landmarks:
        lh = np.array(
            [[res.x - LeftSh_x, res.y - LeftSh_y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)
    # Right Hand Landmarks
    if results.right_hand_landmarks:
        rh = np.array(
            [[res.x - LeftSh_x, res.y - LeftSh_y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])


# Main func (pretty much the core of this app to begin with) -----------------------------------
def main():
    ### Get the arguments
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    smooth_landmarks = not args.unuse_smooth_landmarks
    enable_segmentation = args.enable_segmentation
    smooth_segmentation = args.smooth_segmentation
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    ### Camera setup
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('m', 'j', 'p', 'g'))

    ### Mediapipe Holistics setup
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        smooth_segmentation=smooth_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    ### FPS Module
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    ### Steering Wheel module
    # I only needed to use the update() and the draw_steering_wheel()
    # Everything else is mostly handled in the class itself
    # Though I MIGHT need to use some of its ATTRIBUTES for UDP data sending
    steering_wheel = SteeringWheel()

    ### Gesture Prediction Variables
    pred_sequence = []  # THIS will be used to accumulate the landmark list from each frame.
                        # It's used as input for our prediction model. (5frames, 150 landmarks).
                        # Before that it will be reshaped as (1, 5, 150) though, since our model
                        # was trained using lists of (5, 150) - sequences of 5frames, 150landmarks each.
    current_action = "None"  # Basically a variable to store our prediction result. (We use our predicted index
                            # to get the corresponding string)
    predictions = deque(maxlen=3)
    # This is to filter out small/random results.
    # We'd store our predictions in this list, and we check within the 3/5/10 last predictions to
    # grab and return the most common element in that range.
    # (This would delay our output down somewhat though.)

    # THIS is our predictor/classifier
    gest_clsf = KerasGestureClassifier(model_path=current_model_path)

    ### !!!Webcam capturing loop
    while True:
        display_fps = cvFpsCalc.get()

        # Camera Capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror our captured image
        debug_image = copy.deepcopy(image)

        # Small note for future users ########################################
        debug_image = cv.putText(debug_image, "This app can be turned off using the 'M' key", (230, 30),
                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
        debug_image = cv.putText(debug_image, "This app can be turned off using the 'M' key", (230, 30),
                                 cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Holistics Processing ###############################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Pose ###############################################################
        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            ### brect = calc_bounding_rect(debug_image, pose_landmarks)
            debug_image = draw_pose_landmarks_v2(
                debug_image,
                pose_landmarks
            )

        # Hands ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        # Left Hand
        if left_hand_landmarks is not None:
            debug_image = draw_hand_v2(debug_image, left_hand_landmarks)
            # brect = calc_bounding_rect(debug_image, left_hand_landmarks)

        # Right Hand
        if right_hand_landmarks is not None:
            debug_image = draw_hand_v2(debug_image, right_hand_landmarks)
            # brect = calc_bounding_rect(debug_image, right_hand_landmarks)

        # Steering Wheel ######################################################
        steering_wheel.update(results)
        debug_image = steering_wheel.draw_steering_wheel(debug_image)

        # Displaying FPS ######################################################
        fps_color = (0, 255, 0)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2, cv.LINE_AA)

        # Prediction Logics ###################################################
        # Check if body is detected
        gest_clsf.is_body_detected = gest_clsf.update_body_detect(results)
        print("Is body detected: ", gest_clsf.is_body_detected)

        # "keypoints" is solely used to contain extracted landmarks
        keypoints = extract_keypoints_v3(results)

        # Append keypoint array constantly (until we reach at least 5 frames for prediction)
        pred_sequence.append(keypoints)

        # Take the last 10 frames, if we have 0-9 10 11 then it'd take 2-11 for prediction
        pred_sequence = pred_sequence[-10:]

        ### This section will contain quite a lot of me "yapping" about the whole idea I have
        # in mind for this, but then again it seems pretty OK in my head so I'll give it a shot.
        '''
        - This is how this section works:
        + When there are no predictions, res_idx will have NONE_ACTION_IDX as a default value.
        This is to ensure that "predictions" always have value, so Counter(preds).most_common[0] will always work.
        + When there are predictions, res_idx will contain the result of said predictions.
        + Either way, res_idx will be appended to "predictions" deque.
        + After that, most_common_res_idx will take the most common value in the "predictions" deque, which
        is a queue of predicted res_idx that is constantly updated.
        + Finally, we get our current_action string from the most_common_res_idx.

        - Why this might work:
        + We're using Counter on a constantly upgrading deque.
        + Counter works even when "predictions" deque have no elements, or when "predictions" is full.
        But you can't access the Counter's elements when "predictions" is empty though.
        + The miraculous thing about deque is that when we append a new res_idx into "preds",
        if "preds" is full, it will remove the oldest element in the list (the 0th one) and replace
        it with the new res_idx.

        - Example: We go by <predictions with [lastest output]> <most common> pair
        [ [7] ] = 7
        [ 7 [5] ] = 7 (Counter takes the first value when there are 2 values with the same number of occurences)
        [ 7 5 [5] ] = 5
        [ 5 5 [5] ] = 5
        [ 5 5 [3] ] = 5
        [ 5 3 [3] ] = 3
        '''

        # Default res_idx value, is used when nothing is detected + recognized
        '''
        We actually need this value to use when:
        - nothing is detected 
        - and for Counter().most_common()[0][0] to work, since it requires at least 1 element
        in the list for it to actually take out the first [0] 
        - Counter().most_common() will run even when there's no elements, BUT we won't be able to access
        any elements in it, since there is nothing to access.
        '''
        res_idx = NONE_ACTION_IDX

        # If our sequence has the required length, we predict
        if len(pred_sequence) >= 10:
            res_idx = gest_clsf(pred_sequence[-10:])  # This is our classifier object

        # Append our predicted idx (or default idx) into our "prediction generalization list"
        predictions.append(res_idx)

        # Get the most common res_idx out of the 3 most recent predictions
        # This will ALWAYS work since we'd appended into "predictions". Meaning we can use Counter(preds).most_com[0]
        # IT is technically a list [(element, count)]
        most_common_res_idx = Counter(predictions).most_common(1)[0][0]

        ### DEPRECATED LIST VERSION, move along, people ###
        # most_common_res_idx = Counter(predictions[-3:]).most_common(1)[0][0]
        # predictions = predictions[-10:]     # Now we limit it to 10 recent predictions
        ### DEPRECATED LIST VERSION, move along, people ###

        # Last but not least, we get our action string
        current_action = actions[most_common_res_idx]
        print("Current Action: ", current_action)

        # And we display it
        cv.putText(debug_image, "Gesture:" + current_action, (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2, cv.LINE_AA)

        # Sending our output as UDP Data #######################################
        output_data = process_udp_data(current_action, steering_wheel.current_angle)
        send_data(sock, UDP_IP, UDP_PORT, output_data)

        # Escaping camera loop #################################################
        '''
        Replaced ESC key --> M key
        We use the M key, since we might need the ESC key to pause the game.
        '''
        key = cv.waitKey(1)
        if key == 109:  # 'm' key
            # Close UDP connection
            sock.close()
            break

        # Display our processed image ##########################################
        cv.imshow('10F Keras - MediaPipe Holistic Demo', debug_image)

    ### The end of our camera loop
    cap.release()
    cv.destroyAllWindows()


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


