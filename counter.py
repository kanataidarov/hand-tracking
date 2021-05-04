# Finger Counter 

import cv2
import time
import os 
import hand

def main(): 
    detector = hand.HandDetector()
    cap = detector.InitCam()
    
    fingersFolder = "fingers"
    fingerList = os.listdir(fingersFolder)
    fingerList.sort()
    overlays = []
    for imPath in fingerList: 
        image = cv2.imread(f'{fingersFolder}/{imPath}')
        overlays.append(image) 

    while True: 
        _, img = cap.read() 
        detector.UpdateFps(img)

        lms = detector.FindHandLandmarks(img)
        fingers = detector.FingersUp(lms)
        fingersOpen = fingers.count(1)
        print(f'{fingersOpen} fingers open')

        if (fingersOpen in [3,4]):
            h, w, _ = overlays[fingersOpen-2].shape
            img[:h, :w] = overlays[fingersOpen-2]

        cv2.imshow("Finger counter", img)
        detector.ReleaseCam(cap)

if __name__ == "__main__": 
    main() 