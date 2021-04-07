# Gesture Volume Control (works on Mac only)

import cv2 
import time 
import numpy as np 
import base 
import math 
import osascript as osa

detector = base.HandDetector() 
cap = base.InitCam() 

def findLen(img, lms): 
    length = 0
    if len(lms) > 8: 
        x1,y1 = lms[4][1],lms[4][2]
        x2,y2 = lms[8][1],lms[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(img, (x1,y1), 9, (255,255,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 9, (255,255,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,255,0), 3)
        cv2.circle(img, (cx,cy), 9, (255,255,0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        if (length < 50):
            cv2.circle(img, (cx,cy), 11, (0,0,255), cv2.FILLED)
    return length

def tuneVol(volume): 
    osa.run("set volume output volume "+str(volume))

volume = 0
bar = 400
perc = 0
while True: 
    success, img = cap.read() 

    detector.findHands(img)
    detector.UpdateFps(img)

    lms = detector.findPosition(img, shouldDraw=False)
    length = findLen(img, lms)

    # Hand range 40-400, device volume range 0-100
    volume = np.interp(length, [40,400], [0,100])
    tuneVol(volume)

    cv2.rectangle(img, (40,150), (85,400), (255,0,0), 2)
    bar = np.interp(length, [40,400], [400,150])
    cv2.rectangle(img, (40,int(bar)), (85,400), (255,0,0), cv2.FILLED)
    perc = np.interp(length, [40,400], [0,100])
    cv2.putText(img, f'{int(perc)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)

    cv2.imshow("Gesture Volume Control", img) 
    cv2.waitKey(1)

