import cv2 as opencv
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
buzzer = mixer.Sound('buzzer.wav')

face = opencv.CascadeClassifier('haarcascade_files/haarcascade_frontalface_alt.xml')
leye = opencv.CascadeClassifier('haarcascade_files/haarcascade_lefteye_2splits.xml')
reye = opencv.CascadeClassifier('haarcascade_files/haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('cnnCat2.h5')

path = os.getcwd()
cap = opencv.VideoCapture(0)  #0-webcam, displays what is showing on the webcam
font = opencv.FONT_HERSHEY_COMPLEX_SMALL  #sets the font
count=0
score=0
thickness=2 
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()  #ret - true/false whether it was able to read the file or not, frame - one frame object

    height,width = frame.shape[:2] 

    gray = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)  #converting video to grayscale

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    opencv.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=opencv.FILLED ) #draws a rectangle on the image for score and label

    for (x,y,w,h) in faces:
        opencv.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = opencv.cvtColor(r_eye,opencv.COLOR_BGR2GRAY)
        r_eye = opencv.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = opencv.cvtColor(l_eye,opencv.COLOR_BGR2GRAY)  
        l_eye = opencv.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        opencv.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        opencv.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)
    
        
    if(score<0):
        score=0   
    opencv.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)
    if(score<=7):
        buzzer.stop()
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        opencv.imwrite(os.path.join(path,'image.jpg'),frame)  #saves one frame on the device
        try:
            buzzer.play()
            
        except:  # isplcap.read()aying = False
            pass
        if(thickness<16):
            thickness= thickness+2
        else:
            thickness=thickness-2
            if(thickness<2):
                thickness=2
        opencv.rectangle(frame,(0,0),(width,height),(0,0,255),thickness) #draws a rectangle
    
    opencv.imshow('frame',frame)  #displays on the screen what is being captured by the webcam
    if opencv.waitKey(1) & 0xFF == ord('q'):  #when q is pressed then capturing will terminate
        break
        
cap.release()
opencv.destroyAllWindows()
