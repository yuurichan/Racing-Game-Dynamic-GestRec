# Dynamic Gesture Recognition App

- Real-time Dynamic Gestures Recognition system, utilizing:
  - [Mediapipe Holistic](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md)
  - [LSTM Models](#references)
  - UDP Sockets (for data transferring)
- Is also a controller for [a thesis project that my friend and I are working on](https://github.com/Ghostexvan/FinalProject), using UDP Sockets, Unity and C#.

# Getting Started
(Again, will add requirement.txt and how to run later)
## Prerequisites
(yeah this is where the requirement.txt will settle)

## Included Gestures
(make table here)

## Notes
- Most Notebook files are deprecated, only "Copy2" and "Training" are updated/maintained.
- Do remember to adjust the filepaths in the modules. The filepaths listed in the Notebook files are on my local machine.

## Modules
### Data Collection Module
(In "Copy2" Notebook file)
### Data Processing Module
(In "Training" Notebook file)
### Model Training Module
(In "Training" Notebook file)
### Model Evaluating Module
(In "Training" Notebook file)
### Real-time Gesture Recognition Module
(In "Training" Notebook file)
### Real-time Gesture Recognition App (.py file and .exe file)
(Details soon)

# Dataset
## Details
(Details will be added later)

## Importing Dataset
*(Git wasn't working properly since there were too much files in the Dataset, so I instead compressed most of it in "_Landmark_Data+Backup.rar".)*
- Extract "_5FPS_LandmarkData" and "_10FPS_LandmarkData" into the Dataset folder, it should be like this:
```
Dataset/
┣ _10FPS_LandmarkData/
┗ _5FPS_LandmarkData/
```
- Adjust your datapath in the Data Collection, Data Processing and Model Training modules.

# Models
(Will add structures + evals later)


# Version Control
Everything before v1.5 is tracked locally, I only pushed this current version to github because it's the most stable as of this point.
- **v1.0: "Copy1" Notebook file - the baseline for almost every versions that came after.**
  - Added Data Collection, Data Processing, Model Training and Evaluation, Real-time Gesture Recognition modules in "Copy1" Notebook file.
  - Added 30-frames models. Most models had decent accuracy but the evaluation graphs aren't exactly great, especially with the One-hot version.
  - Initial dataset was (7*100, 30, 150) - 7 Gestures with 100 sequences each; each sequence is 30 frames long; each frame has 150 landmarks.
  - (Dataset itself was yet to be properly processed.)
  - Landmark visualization methods written with https://github.com/Kazuhito00/mediapipe-python-sample as base.

- **v1.1 : "Copy2" + "Training" Notebook files**
  - Adjusted Data Collection module: Users can now collect 10-frames data and 5-frames data at the same time.
  - Re-recorded dataset. Dataset is currently at (7*300, 5/10, 150).
  - Made 7 models for 10-frames dataset; 4 models for 5-frames dataset (mostly for testing). Models have good accuracy but graphs still have a lot of acc/loss spikes.
  - Recognition system on Notebook now performs well, with faster response time using 5f-10f models.
  *(as of this point, all other Notebook files are deprecated; 30-frames models and dataset were backed up and removed from the main app)*

- **v1.2: Dataset Overhaul**
  - Normalized all current data in the dataset: Shifted the x, y coordinates of every landmarks with (Mediapipe's) Pose Landmark's LEFT SHOULDER (11th landmark) as point of origin (0, 0).
  - Added 300 more sequences for each gestures, making the total dataset count at (7*600, 5/10, 150).
  - Tweaked and retrained all models with newly adjusted dataset. Models now yield great acc with low loss, graphs don't have acc/loss spikes - spikes are not too severe.
 
- **v1.3: Reworking the Dynamic Gesture Recognition App**
  - Converted most models (mainly models 1, 4, 6 and 7) into tflite models for low-end devices.
  - Added class files to invoke model predictions easier (you only need to specify the model directory and your desired prediciton threshold).
  - Recognition system is rewritten in a seperated Python file. Available in *4 versions*: 5f-10f Keras and 5f-10f TFLite.
 
- **v1.4: SteeringWheel class**
  - Added SteeringWheel class to display... a basic steering wheel on webcam feed.
  - Added steering wheel Angles for display.

- **v1.5: Bug Fixing + Adding new Gesture**
  - Added "Reverse" gesture label and 600 sequences of data for "Reverse" gesture.
  - Retrained some models for use with newly reworked dataset.
  - Fixed Keras classifier and TFLite classifier classes' "None" class problem (The system recognized something else rather than "None" when there are no landmarks detected).
  - Cleaned + reorganized most of the filebase of the project.

## TO DO LIST FOR UPCOMING VERSION 2.0
- UDP Socket support + Controller support for https://github.com/Ghostexvan/FinalProject.
- Build separate Executable file (.exe) for ease of use.

# References
- [Hand and Pose Landmark Visualization Codebase](https://github.com/Kazuhito00/mediapipe-python-sample/blob/main/sample_holistic.py)
- [Steering Wheel Visualization Codebase](https://github.com/HugoM25/Virtual_Steering_Wheel_Controller)
- Some of the logics for this project and to improve output accuracy are based on:
  - [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/app.py#L158)
  - [nicknochnack/ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage/tree/main)
- Articles:
  - [Light-Weight Deep Learning Techniques with Advanced Processing for Real-Time Hand Gesture Recognition](https://www.mdpi.com/1424-8220/23/1/2)
  - [MediaPipe’s Landmarks with RNN for Dynamic Sign Language Recognition](https://www.mdpi.com/2079-9292/11/19/3228)
  - [An Efficient Patient Activity Recognition using LSTM Network and High-Fidelity Body Pose Tracking](https://www.proquest.com/openview/b4c9be043783ff11e3db57766b5562a5/1?pq-origsite=gscholar&cbl=5444811)
  - [SIGN LANGUAGE RECOGNITION USING LANDMARK DETECTION, GRU and LSTM](https://ajec.smartsociety.org/wp-content/uploads/2023/01/5.pdf)
