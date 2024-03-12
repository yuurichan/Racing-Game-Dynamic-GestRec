import mediapipe as mp
import cv2
from math import sqrt, acos, degrees


class SteeringWheel:
    '''
    results = holistic.process(image)
    results.left_hand_landmarks --> a bunch of landmark{x, y, z} - landmark objects
    results.left_hand_landmarks.landmark --> list of x,y,z coords
    '''
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistics = mp.solutions.holistic   # Might be unused cuz we'll be getting
                                                    # our landmarks from the main app
        self.image = None
        self.image_width = 960
        self.image_height = 720
        self.hands_landmarks_list = [None, None]      # Getting 9th landmark coords on left and right hands
        print("Current list - count: ", len(self.hands_landmarks_list))
        '''
        When list is empty [], you can't reference the index directly ([0] or [1] resulted in list assignment out of range)
        So I came up with a VERY SCUFFED solution:
        ...that is to initialize my list with 2 elements.
        '''
        self.current_angle = 0
        self.current_hand_distance = 0

    # results = holistic.process(image)
    def update(self, results):
        self.get_hand_landmarks(results)
        self.current_hand_distance = self.get_distance_between_hands()
        # print(self.current_hand_distance)
        self.current_angle = self.get_angle_between_hands()
        self.__set_good_hand_order()
        # So this would've had self.draw_steering_wheel but I decided to make it
        # a seperated method that returns an image like my other visualize methods

    # Get the x y coords of the 9th landmark of each hands
    # They need to be multiplied with the app's screen ratio to get the coord on OpenCV's window
    def get_hand_landmarks(self, results):
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        if left_hand_landmarks is not None and right_hand_landmarks is not None:
            if len(self.hands_landmarks_list) >= 2:
                # Get the preprocessed hand landmark coords
                self.hands_landmarks_list[0] = [min(int(left_hand_landmarks.landmark[9].x * self.image_width), self.image_width - 1),
                                                min(int(left_hand_landmarks.landmark[9].y * self.image_height), self.image_height - 1)]
                self.hands_landmarks_list[1] = [min(int(right_hand_landmarks.landmark[9].x * self.image_width), self.image_width - 1),
                                                min(int(right_hand_landmarks.landmark[9].y * self.image_height), self.image_height - 1)]

            # if len(self.hands_landmarks_list) == 0:
            #     # Left hand first, then right hand
            #     self.hands_landmarks_list.append([min(int(left_hand_landmarks.landmark[9].x * self.image_width), self.image_width - 1),
            #                                     min(int(left_hand_landmarks.landmark[9].y * self.image_height), self.image_height - 1)])
            #     self.hands_landmarks_list.append([min(int(right_hand_landmarks.landmark[9].x * self.image_width), self.image_width - 1),
            #                                     min(int(right_hand_landmarks.landmark[9].y * self.image_height), self.image_height - 1)])
        # If either of the hand landmarks are None (undetectable) then I'll just empty the hand_landmark_list,
        # thus rendering the other funcs unusable
        else:
            self.hands_landmarks_list = [None, None]

    def get_distance_between_hands(self):
        if len(self.hands_landmarks_list) >= 2 and self.hands_landmarks_list[0] is not None and self.hands_landmarks_list[1] is not None:
            pos1 = self.hands_landmarks_list[0]
            pos2 = self.hands_landmarks_list[1]
            sqr_dist = ( pos2[0] - pos1[0] )**2 + ( pos2[1] - pos1[1] )**2
            return sqrt(sqr_dist)
        else:
            return 0

    def get_angle_between_hands(self):
        if len(self.hands_landmarks_list) >= 2 and self.hands_landmarks_list[0] is not None and self.hands_landmarks_list[1] is not None:
            # If distance between hands is less than 100, disable angle + wheel
            # This is also to prevent division by 0, since I can technically make
            # the 9th landmark of both of my hands overlap, thus getting dist=0.
            if self.current_hand_distance < 100:
                return 0

            pos1 = self.hands_landmarks_list[0]
            pos2 = self.hands_landmarks_list[1]
            vec_hands = (pos2[0] - pos1[0] , pos2[1] - pos1[1])
            vec_comp = (0,1)
            magn_vec_hands = sqrt(vec_hands[0]**2 + vec_hands[1]**2)
            magn_vec_comp = sqrt(vec_comp[0]**2 + vec_comp[1]**2 )
            angle = acos( (vec_comp[0]*vec_hands[0] + vec_comp[1]*vec_hands[1]) / (magn_vec_comp * magn_vec_hands) )
            angle = degrees(angle)
            return angle - 90
        else:
            return 0

    # This is to swap the coordinates' order when:
    # + Left Hand's coord is further right than Right Hand's coord - Tay trái nằm phía bên phải của tay phải
    # + Right Hand's coord is further left than Left Hand's coord - Tay phải nằm phía bên trái của tay trái
    def __set_good_hand_order(self):
        if len(self.hands_landmarks_list) >= 2 and self.hands_landmarks_list[0] is not None and self.hands_landmarks_list[1] is not None:
            if (self.hands_landmarks_list[0][0] > self.hands_landmarks_list[1][0]) :
                #Swap temp hand landmarks
                swap_landmarks = self.hands_landmarks_list[0]
                self.hands_landmarks_list[0] = self.hands_landmarks_list[1]
                self.hands_landmarks_list[1] = swap_landmarks

    # Draw steering wheel (I feel like I should return an image here, like my other visualization methods)
    def draw_steering_wheel(self, image):
        # image_width, image_height = image.shape[1], image.shape[0]

        if len(self.hands_landmarks_list) >= 2 and self.hands_landmarks_list[0] is not None and self.hands_landmarks_list[1] is not None:
            if self.current_hand_distance >= 100:
                # Draw debug point to average hand landmarks position (hopefully the hands)
                cv2.circle(image, self.hands_landmarks_list[0], 12, (0, 0, 255), 2)
                cv2.circle(image, self.hands_landmarks_list[1], 12, (255, 0, 0), 2)

                # Draw debug line between the hands
                cv2.line(image, self.hands_landmarks_list[0], self.hands_landmarks_list[1], (0, 255, 0), 2)

                # Draw circle to simulate steering wheel
                cv2.circle(image, (int((self.hands_landmarks_list[0][0] + self.hands_landmarks_list[1][0]) / 2),
                                       int((self.hands_landmarks_list[0][1] + self.hands_landmarks_list[1][1]) / 2)),
                           int(self.get_distance_between_hands() / 2), (128, 128, 128), 2)

        # If there are literally no hand landmarks then Angle will be 0
        cv2.putText(image, "Angle : " + str(round(self.get_angle_between_hands(), 2)), (0+30, 720-30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, 1)

        return image