# On-screen painter with finger 

import cv2 
import mediapipe as mp 
import os 
import hand 
import numpy as np 

#################################################################
### Constants ###################################################
#################################################################
headersFolder =     "img/headers"
wCam =              1280
hCam =              720
headerHeight =      125
brushThickness =    9 
eraserThickness =   45
#################################################################

def main(): 
    # import header images 
    headers = []
    for headerPath in sorted(os.listdir(headersFolder)): 
        headerImg = cv2.imread(f'{headersFolder}/{headerPath}')
        headers.append(headerImg)

    # init camera 
    detector = hand.HandDetector(detectConf=.75)
    cap = detector.InitCam(wCam, hCam)
    header = headers[0] 
    drawClr = detector.COLORS['PURPLE']
    xprev = yprev = 0 

    canvas = np.zeros((hCam, wCam, 3), np.uint8)
    while True: 
        # read frame and flip it horizontally 
        _, img = cap.read()
        img = cv2.flip(img, 1)

        # find hand landmarks 
        lms = detector.FindHandLandmarks(img, shouldDraw=False)
        # find up fingers 
        fingers = detector.FingersUp(lms)
        if (len(lms)>0): 
            x1, y1 = lms[8][1:]
            x2, y2 = lms[12][1:]
            # Select mode - two fingers are up 
            if (fingers == [0,1,1,0,0]):
                xprev = yprev = 0 
                if y1 < headerHeight: 
                    # check for click 
                    if 250 < x1 < 450: 
                        header = headers[1] 
                        drawClr = detector.COLORS['PURPLE']
                    elif 550 < x1 < 750: 
                        header = headers[2]
                        drawClr = detector.COLORS['BLUE']
                    elif 800 < x1 < 950: 
                        header = headers[3]
                        drawClr = detector.COLORS['GREEN']
                    elif 1050 < x1 < 1200: 
                        header = headers[0]
                        drawClr = detector.COLORS['BLACK']
                cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawClr, cv2.FILLED) 
            # Draw mode - index finger is up 
            if (fingers == [0,1,0,0,0]):
                cv2.circle(img, (x1,y1), brushThickness, drawClr, cv2.FILLED) 
                if (xprev<=0 and yprev<=0): 
                    xprev, yprev = x1, y1
                thickness = eraserThickness if drawClr==detector.COLORS['BLACK'] else brushThickness 
                cv2.line(img, (xprev,yprev), (x1,y1), drawClr, thickness)
                cv2.line(canvas, (xprev,yprev), (x1,y1), drawClr, thickness)
                xprev, yprev = x1, y1 

        # mask canvas on main img 
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, canvas) 
        
        # set header 
        img[:headerHeight, :wCam] = header 

        # show frame and try to release it 
        detector.UpdateFps(img)
        cv2.imshow("On-screen painter", img)
        detector.ReleaseCam(cap)


if __name__ == "__main__": 
    main() 