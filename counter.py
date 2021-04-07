# Finger Counter 

import cv2
import time
import os 
import base

cap = base.InitCam()
detector = base.HandDetector()

fingersFolder = "fingers"
fingerList = os.listdir(fingersFolder)
fingerList.sort()
overlays = []
for imPath in fingerList: 
    image = cv2.imread(f'{fingersFolder}/{imPath}')
    overlays.append(image) 

tips = [4,8,12,16,20]

while True: 
    success, img = cap.read() 
    detector.UpdateFps(img)
    img = detector.findHands(img)
    lms = detector.findPosition(img)
    if (len(lms)>20):
        fingers = []

        # Left Thump
        if (lms[tips[0]][1]<lms[tips[0]-1][1]): 
            fingers.append(1)
        else: 
            fingers.append(0)

        # Pointies 
        for tip in range(1,5): 
            if (lms[tips[tip]][2]<lms[tips[tip]-2][2]): 
                fingers.append(1)
            else: 
                fingers.append(0)
        
        fingersOpen = fingers.count(1)
        print(f'{fingersOpen} fingers open')

        if (fingersOpen in [3,4]):
            h,w,c = overlays[fingersOpen-2].shape
            img[:h, :w] = overlays[fingersOpen-2]

    cv2.imshow("Finger counter", img)
    cv2.waitKey(1)