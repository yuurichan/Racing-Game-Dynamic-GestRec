import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import os

actions = np.array(["Idle", "StaticStraight", "LSteer", "RSteer", "Boost", "Brake", "BrakeHold", "Reverse", "None"])
NONE_ACTION_IDX = 8     # Yeah I made this up, purely for when there's nothing happening - "None" action
full_path = os.path.realpath(__file__)
# print(os.path.dirname(full_path))

# This (luckily) is a generic class used for both 10f and 5f models.
class KerasGestureClassifier(object):
    def __init__(self,
                 model_path=os.path.join(os.path.dirname(full_path),"_1-5f-lstm_model_1.keras"),  # Chose this as default because random
                 threshold=0.5):
        self.model = models.load_model(model_path)
        self.pred_threshold = threshold

        # This is a really bad way to counter the "No landmark BrakeHold - threshold 0.5" problem, I'm implementing it as a test
        self.is_body_detected = False

    def update_body_detect(self, results):
        if results is not None and results.pose_landmarks is not None:
            return True
        return False


    # inp_seq will be in the form of (5, 150) or (10, 150).
    # This will be collected in the main app.
    # (Seq of 5/10 frames, each consisting of 150 landmarks and values)
    # This will then be reshaped as (1, 5/10, 150) - using np.expand_dims,
    # since the training of the model required it to be multiple inputs of
    # (5/10, 150): (x, 5/10, 150) with x being the number of sequences.
    ## Then again, we only needed to check 1 sequence's worth of landmarks,
    ## so (1, 5/10, 150) will suffice.
    def __call__(self,
                 inp_seq):
        # keypoints = extract_keypoints_v3(results)
        # sequence.append(keypoints)
        # sequence = sequence[-10:]

        # if len(sequence) == 5:
        if self.is_body_detected is False:
            return NONE_ACTION_IDX

        # res = self.model.predict(np.expand_dims(inp_seq, axis=0), verbose=0)[0]
        res = self.model.predict(np.expand_dims(inp_seq, axis=0))[0]

        '''
        # Since our model has a softmax layer at the end, the output will be
        # an array of percentages which totals to 1:
        # EX: [0.1 0.3 0.6]
        # When we use res[np.argmax(res)], np.argmax(res) will take the index
        # of the highest element in the list. With res[np.argmax(res)] we get
        # the highest element.
        Reason why we don't use max(res) is because we still need the index
        to get our action string in actions[].
        But yeah the INDEX from np.argmax(res) is basically our index. 
        '''
        # current_action = ""
        if res[np.argmax(res)] > self.pred_threshold:
            current_action_idx = np.argmax(res)
        else:
            current_action_idx = NONE_ACTION_IDX

        return current_action_idx

