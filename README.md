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
## Credit
 - The database of faces are from [AT&T Laboratories Cambridge](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
 - Some idea and part of the code are from [Face Recognition with OpenCV, OpenCV 2.4.13.2 documentation](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#id27)
