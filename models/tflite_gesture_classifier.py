import numpy as np
import tensorflow as tf
import os

actions = np.array(["Idle", "StaticStraight", "LSteer", "RSteer", "Boost", "Brake", "BrakeHold", "Reverse", "None"])
NONE_ACTION_IDX = 8
full_path = os.path.realpath(__file__)
dir_name = os.path.dirname(full_path)
# print(os.path.dirname(full_path))

# This (luckily) is a generic class used for both 10f and 5f models.
class TFLiteGestureClassifier(object):
    def __init__(
            self,
            model_path=os.path.join(dir_name,"tflite","7-lstm_model_7.tflite"),
            threshold=0.5,
            num_threads=1
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.pred_threshold = threshold

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
        if self.is_body_detected is False:
            return NONE_ACTION_IDX

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.expand_dims(np.float32(inp_seq), axis=0))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']

        res = self.interpreter.get_tensor(output_details_tensor_index)

        result_idx = np.argmax(np.squeeze(res))

        # If result is lower than pred_threshold, return action string
        # Else return empty string.
        # This is the same as the keras one, taking the highest element
        # in the list using its own index. If it exceeds the set threshold,
        # return an action string with the same index.
        '''
        np.squeeze(res): 
        [0.0023739  0.00033794 0.9810674  0.00071454 0.00712923 0.00787502 0.00050211]
        
        np.argmax(np.squeeze(res)):
        2 - since it's 0.98, the highest in the group
        '''
        if np.squeeze(res)[result_idx] > self.pred_threshold:
            current_action_idx = result_idx
        else:
            current_action_idx = NONE_ACTION_IDX

        return current_action_idx


