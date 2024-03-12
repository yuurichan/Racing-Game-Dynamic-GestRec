import os
import numpy as np
import cv2
from glob import glob


def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


def save_frame(video_path, save_dir, gap=1, frames_to_skip=1):
    name = video_path.split("/")[-1].split(".")[0].split("\\")[-1]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    frame_count = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video length: ", length, " frames.")

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        frame_count += 1
        if frame_count <= frames_to_skip:
            continue
        # For 5FPS use
        # if idx not in [0, 1, 2, 3, 4]:
        #     continue
        # if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        #     continue

        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1



if __name__ == "__main__":
    # video_paths = glob("K:/Pycharm/NLN/_holistics/__DATASET/7. StaticRTilt/30fps/*.mp4")

    # video_paths = glob("K:/Pycharm/NLN/_holistics/__DATASET_TEMP/_5FPS_Data/RSteer/*.mp4")
    # save_dir = "K:/Pycharm/NLN/_holistics/__DATASET_TEMP/_5FPS_Data/RSteer/img"
    video_paths = glob("F:/OBS REcs/Thesus Data/10FPS Test 3/*.mp4")
    # video_paths = glob("F:/OBS REcs/Thesus Data/5FPS Test 3/*.mp4")

    # save_dir = r"K:\Pycharm\NLN\_holistics\__DATASET\1. Boost\IMG_30"
    # save_dir = "K:/Pycharm/NLN/_holistics/__DATASET/7. StaticRTilt/IMG_30"

    save_dir = "F:/OBS REcs/Thesus Data/10FPS Test 3/img"
    # save_dir = "F:/OBS REcs/Thesus Data/5FPS Test 3/img"

    # Getting 30 frames in a video containing 46 frames (very funny)
    # print(np.round(np.linspace(0, 45, 30)).astype(int))

    for path in video_paths:
        save_frame(path, save_dir, gap=1, frames_to_skip=0)
        print(path)
        name = path.split("/")[-1].split(".")[0].split("\\")[-1]
        print(name)
        save_path = os.path.join(save_dir, name)
        print(save_path, '\n', '----------')

