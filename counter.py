# Finger Counter 

import cv2
import glob 
import os 
import hand

fingersFolder = "img/fingers"

def main(): 
    detector = hand.HandDetector()
    cap = detector.InitCam()
    fingerList = sorted(glob.glob(f'{fingersFolder}/*.png'), 
        key=lambda x: int(os.path.basename(x).split('.')[0]))
    overlays = {}
    i: int = 1
    for imPath in fingerList: 
        overlays[i] = cv2.imread(imPath)
        i += 1
    
    while True: 
        _, img = cap.read() 
        detector.UpdateFps(img)

        lms = detector.FindHandLandmarks(img)
        fingers = detector.FingersUp(lms)
        fingersOpen = fingers.count(1)

        if (fingersOpen in [1,2,3,4,5]):
            print(f'{fingersOpen} fingers open')
            h, w, _ = overlays[fingersOpen].shape
            img[:h, :w] = overlays[fingersOpen]

        cv2.imshow("Finger counter", img)
        detector.ReleaseCam(cap)

if __name__ == "__main__": 
    main() 