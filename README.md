# faceDetectionAndRecognition
This program can automatically detect a face from an input image and recognize the face.

There are three steps for achieving the goal: face detection --> training --> face recognition.  

## Requirement
- openCV: 2.4.13.2

## Training data set
Run the python script to produce a csv file which includes pairs of image and lable:

``` 
python creteCSV.py
```

## Compile and run
```
g++ $(pkg-config --cflags --libs opencv) *.cpp
./a.out
```
