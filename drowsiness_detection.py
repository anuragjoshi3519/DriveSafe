import cv2 as opencv
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

def drive(start_score = 17, stop_score = 10):
    
    mixer.init()
    buzzer = mixer.Sound('buzzer.wav')

    face = opencv.CascadeClassifier('haarcascade_files/haarcascade_frontalface_alt.xml')
    leye = opencv.CascadeClassifier('haarcascade_files/haarcascade_lefteye_2splits.xml')
    reye = opencv.CascadeClassifier('haarcascade_files/haarcascade_righteye_2splits.xml')

    model = load_model('cnnCat2.h5')

    path = os.getcwd()
    cap = opencv.VideoCapture(0)  #0-webcam, displays what is showing on the webcam
    font = opencv.FONT_HERSHEY_COMPLEX_SMALL  #sets the font
    count=0
    score=0
    thickness=2

    lbl = None

    while(True):
        rpred=0
        lpred=0

        ret, frame = cap.read()  #ret - true/false whether it was able to read the file or not, frame - one frame object
        
        if ret==False:
            continue
        
        height,width = frame.shape[:2] 

        gray = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)  #converting video to grayscale

        faces = list(face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)))

        if len(faces)!=0:

            faces.sort(key = lambda f:f[2]*f[3])
            x,y,w,h = faces[-1]

            opencv.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

            cropped_face = frame[y:y+h,x:x+w]

            left_eye = leye.detectMultiScale(cropped_face)
            right_eye =  reye.detectMultiScale(cropped_face)

            opencv.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=opencv.FILLED ) #draws a rectangle on the image for score and label

            if len(right_eye)!=0:
                x,y,w,h = right_eye[0]
                r_eye=cropped_face[y:y+h,x:x+w]
                count=count+1
                r_eye = opencv.cvtColor(r_eye,opencv.COLOR_BGR2GRAY)
                r_eye = opencv.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = np.argmax(model.predict(r_eye), axis=-1)
                if(rpred==1):
                    lbl='Open' 
                if(rpred==0):
                    lbl='Closed'

            if len(left_eye)!=0:
                x,y,w,h = left_eye[0]
                l_eye=cropped_face[y:y+h,x:x+w]
                count=count+1
                l_eye = opencv.cvtColor(l_eye,opencv.COLOR_BGR2GRAY)  
                l_eye = opencv.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = np.argmax(model.predict(l_eye), axis=-1)
                if(lpred==1):
                    lbl='Open'   
                if(lpred==0):
                    lbl='Closed'

        if(rpred==0 and lpred==0):
            score += 1
            opencv.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            if score > start_score*2:
                score -= 2
            else:
                score -= 1
            opencv.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)


        if(score<0):
            score=0
        
        opencv.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,opencv.LINE_AA)
        
        if(score<=stop_score):
            buzzer.stop()
        if(score>=start_score):
            #person is feeling sleepy so we beep the alarm
#             opencv.imwrite(os.path.join(path,'image.jpg'),frame)  #saves one frame on the device
            try:
                buzzer.play()
            except:
                pass
            if(thickness<16):
                thickness= thickness+2
            else:
                thickness=thickness-2
                if(thickness<2):
                    thickness=2
            opencv.rectangle(frame,(0,0),(width,height),(0,0,255),thickness) #draws a rectangle
        
        if opencv.waitKey(1) & 0xFF == ord('q'):  #when q is pressed then capturing will terminate
            break
            
        ret, buffer = opencv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        # opencv.imshow('frame',frame)  #displays on the screen what is being captured by the webcam

    cap.release()
    opencv.destroyAllWindows()
    

if __name__ == "__main__":
    drive()
