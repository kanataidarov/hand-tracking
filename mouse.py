import cv2 
import numpy as np 
import hand 
import pyautogui

#################################################################
### Constants ###################################################
#################################################################
wCam =      800     # shrink output frame width 
hCam =      550     # shrink output frame height 
frameR =    150     # movement detection boundaries reducted 
indFgr =    8       # index of index finger in mediapipe 
midFgr =    12      # index of middle finger in mediapipe 
smooth =    7       # smoothening mouse cursor 
#################################################################

def main():
    wScr, hScr = pyautogui.size()

    detector = hand.HandDetector()
    cap = detector.InitCam(wCam, hCam)

    cLocX = cLocY = pLocX = pLocY = 0 
    while True: 
        _, img = cap.read()
        img = cv2.resize(img, (wCam,hCam))
        detector.UpdateFps(img)

        # Hand landmarks and boundaries
        lms = detector.FindHandLandmarks(img, handNo=0)
        detector.FindBoundaries(img, lms, shouldDraw=True)

        # Tip of the index and middle fingers
        if len(lms) > 20: 
            _,xInd,yInd = lms[8]
            _,xMid,yMid = lms[12]

        # Fingers that are up 
        fingers = detector.FingersUp(lms)
        cv2.rectangle(img, (frameR,frameR), (wCam-frameR, hCam-frameR), detector.COLORS['PURPLE'])

        # Track movement only when Index finger Up
        if fingers==[0,1,0,0,0]: 
            # Convert coords to screen wide 
            xMou = np.interp(xInd, (frameR,wCam-frameR), (0,wScr))
            yMou = np.interp(yInd, (frameR,hCam-frameR), (0,hScr))
            # Smoothen mouse cursor 
            cLocX = pLocX + (xMou - pLocX) / smooth 
            cLocY = pLocY + (yMou - pLocY) / smooth 
            # Move Mouse 
            pyautogui.moveTo(cLocX, cLocY)
            cv2.circle(img, (xInd,yInd), 9, detector.COLORS['PURPLE'], cv2.FILLED)
            # Update smoothened mouse cursor 
            pLocX = cLocX 
            pLocY = cLocY

        # Track movement when Index and Middle fingers up 
        if fingers==[0,1,1,0,0]:
            # Find distance between index and Middle fingers 
            length = detector.FindDist(img, lms, first=8, second=12, shouldDraw=True) 
            # Click mouse if distance below threshold 
            if (length < 36): 
                cv2.circle(img, (xInd,yInd), 9, detector.COLORS['RED'], cv2.FILLED)
                cv2.circle(img, (xMid,yMid), 9, detector.COLORS['RED'], cv2.FILLED)
                pyautogui.click() 

        cv2.imshow("Virtual Mouse", img)
        detector.ReleaseCam(cap)

if __name__ == "__main__": 
    main() 
