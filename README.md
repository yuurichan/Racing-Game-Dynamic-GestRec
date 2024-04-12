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
- Most Notebook files are deprecated, only "Copy2" and "Training" are updated/maintained. You can also take a look at "Copy1_improv" to see the baseline of v1.0. (I'd advise against using it though...)
- Do remember to adjust the filepaths in the modules. The filepaths listed in the Notebook files are on my local machine.
- I might need to note that this warning (I got it from using a venv): `This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.`.
Means that Tensorflow is simply telling you that it can use the operations listed to make things faster. This does not affect the app in any way.
You can choose to not display this by adding this at the start of the app file:
```
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
- Remember to exclude the folder containing the built app in Windows Defender Settings. It usually gives a false positive since apps compiled with PyInstaller don't have cetificates.
- Notes about things to notice in app compiling are found in the README.txt file in the Released files.

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
- (Similarly, if you want to use the "Copy1_improv" file, extract the "_LandmarkData - 30fModified" into the Dataset folder like so:)
```
Dataset/
┗ _LandmarkData - 30fModified/
```
Then again, I wouldn't recommend using it. It's more or less a prototype (baseline for v1.0).

- _**IMPORTANT: Adjust your datapath in the Data Collection, Data Processing and Model Training modules.**_

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
 
- **v2.0: Release App + Bugfixes**
  - Added UDP Socket support for [this project](https://github.com/Ghostexvan/FinalProject). UDP Socket is added to every Python app files (Both Keras/TFLite - 5f/10f).
  - Is now built as an app, with 2 versions: Keras model ; TFLite model.
  - Added TFLite models for 5f and 10f.
  - Bugfix details:
  	+ MOST IMPORTANT: Fixed most of the explanations in my "Copy2" and "Training" Notebook files. Reason being:
  		1. IF you use OpenCV to mirror/flip and image horizontally, then you use Mediapipe Holistics on it, then it will process and get the opposite landmark side for you.
  		2. This won't affect the models whatsoever. Since it was trained using the same mirrored landmarks.
    
  		E.G: Your LEFT Hand will have results.right_hand_landmarks; Your RIGHT SHOULDER will have PoseLandmark.LEFT_SHOULDER (the 11th pose landmark).
  	+ Fixed label list out of bounds for app_keras_10f.py.
  	+ Added note on how to turn off the app on its UI.
   	
  	**_(Steering Wheel section)_**
  	+ Changed Steering wheel's minimal distance (for calculating angle + for displaying) from 100 --> 10.
  	+ Fixed Steering wheel angle display. Mostly to prevent divide by 0 when calculating the angle between 2 hands.
  	+ Adjustments to Steering wheel angle calculations: Caps your steering wheel angle to -90°/90° when the steering wheel angle exceed either of them.
  	(The Old Steering wheel class (the one prior to v2.0) is still available for use, check the __init__ file in "utils" folder)
  	
  	**_(PyInstaller section)_**
  	+ Fixed a major problem regarding app compiled from PyInstaller: "NoneType" object has no attribute "write".
  		1. Problem details: "App is built with [noconsole / windowed] and tensorflow's logging is naively assuming that sys.stdout and sys.stderr are available, 
   but in fact they are NoneType."
  		2. Fix: (is specified at the start of every Python app files) - Sending output to a dummy stream.
 
- **v2.1: Hotfix**
  - Fixed SteeringWheel class not being able to get both hand landmarks correctly, which affected how the steering wheel is displayed on the app. This problem only occurs when cap_width and cap_height are small.
      + _(E.G: Putting video capture size at **(950, 720)** would return us an image size of **(960, 720)**. But **(720, 540)** would return us an image size of **(640, 480)**, which is smaller than our initial size.)_
  - SteeringWheel class will take the image size directly from the app to calculate the hand landmark coordinates, instead of having to rely on a constant cap_width and cap_height defined by the user.
  - Removed image_width and image_height attributes in SteeringWheel class (reasons listed above).
  - Rebuilt both Keras and TFLite to implement the SteeringWheel class update.
  - Added a TFLite app version for smaller monitors.

## ~~TO DO LIST FOR UPCOMING VERSION 2.0~~ VERSION ~~2.0~~ 2.1.1 IS RELEASED
- ~~UDP Socket support + Controller support for https://github.com/Ghostexvan/FinalProject.~~
- ~~Build separate Executable file (.exe) for ease of use.~~
- ~~If no problem occurs, v2.0 will be the final version for this app as a whole.~~
- We have gotten to v2.1.1 with a couple of bug fixes here and there. This should be final (I hope I don't jinx it).

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
