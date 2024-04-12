So the main project still revolves around 10frames/5frames models.
This 30frames model is used to compare with the ones in the "_(deprecated) 30f models" folder.
My thesis requires me to make comparisons between them, since I have to make them look as if I'm improving my models bit by bit.
(but in reality I'm doing this in reverse, I went from old, unoptimized 30f models --> 10f/5f models --> improved 30f models, purely for comparisons)
You can test this one out if you want, just change the model path in the "keras_gesture_classifier.py" file, copy one of the <app_keras>.py files and change the
input into 30frames instead of 5 or 10 frames - Lines 540, 583 and 584.

That should be all from me.