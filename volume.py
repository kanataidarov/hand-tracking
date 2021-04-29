# Gesture Volume Control (Mac only)

import cv2 
import time 
import numpy as np 
import base 
import math 
import osascript as osa

class VolumeControl(base.HandDetector):
    def __init__(self, volume = 0, bar = 400, perc = 0, 
                mode=False, maxHands=1, detectConf=.5, trackConf=.5):
        super().__init__(mode, maxHands, detectConf, trackConf)
        self.volume = volume 
        self.bar = bar 
        self.perc = perc 

    # Finds distance between thumb and index fingers
    def findLen(self, img, lms, littleDown=False): 
        length = 0
        if len(lms) > 8: 
            # Filter based on size 
            x1,y1 = lms[4][1],lms[4][2]
            x2,y2 = lms[8][1],lms[8][2]
            cx,cy = (x1+x2)//2,(y1+y2)//2

            cv2.circle(img, (x1,y1), 9, (255,255,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 9, (255,255,0), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255,255,0), 3)

            length = math.hypot(x2-x1, y2-y1)
            if littleDown:
                cv2.circle(img, (cx,cy), 9, (0,0,255), cv2.FILLED)
            else:
                cv2.circle(img, (cx,cy), 9, (255,255,0), cv2.FILLED)
            
        return length

    # changes volume by perc
    def tuneVol(self, perc): 
        osa.run("set volume output volume "+str(perc))

    # get current system volume 
    def receiveVol(self): 
        return osa.run("get output volume of (get volume settings)")[1]

    # Checks which fingers up 
    def fingersUp(self, lms): 
        fingers = []
        tips = [4,8,12,16,20]

        if (len(lms)>20):
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
            
        return fingers

    # Finds hand boundaries
    def findBoundaries(self, img, lms, shouldDraw=False): 
        margin = 10
        if len(lms)>0: 
            fLms = list(zip(*lms))
            xMin = min(fLms[1])
            xMax = max(fLms[1])
            yMin = min(fLms[2])
            yMax = max(fLms[2])
            
            if shouldDraw: 
                cv2.rectangle(img, (xMin-margin,yMin-margin), (xMax+margin,yMax+margin), (50,255,255), 1)
            
            return (xMax-xMin)*(yMax-yMin)//100

        return 0

def main():
    maxArea = 1050
    minArea = 550
    smooth = 5

    detector = VolumeControl() 
    cap = base.InitCam() 

    length = 0
    perc = 0

    while True: 
        _, img = cap.read() 
        img = cv2.resize(img, (800, 530))
        detector.UpdateFps(img)

        # Find hand 
        detector.findHands(img)
        lms = detector.findPosition(img, shouldDraw=False)

        length = detector.findLen(img, lms, littleDown=False)
        perc = np.interp(length, [40,400], [0,100])

        # Hand range 40-400, device volume range 0-100
        area = detector.findBoundaries(img, lms, shouldDraw=True)
        if minArea < area < maxArea and not detector.fingersUp(lms)[4]: 
            detector.findLen(img, lms, littleDown=True)
            detector.tuneVol(perc)
            volSetClr = (0,0,255)
        else: 
            volSetClr = (255,0,0)

        # HUD 
        cv2.rectangle(img, (40,150), (85,400), (255,0,0), 2)
        bar = np.interp(length, [40,400], [400,150])
        cv2.rectangle(img, (40,int(bar)), (85,400), (255,0,0), cv2.FILLED)
        perc = smooth * round(perc/smooth) 
        cv2.putText(img, f'{int(perc)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
        cv2.putText(img, f'VOL: {int(detector.receiveVol())}', (650, 50), cv2.FONT_HERSHEY_COMPLEX, 1, volSetClr, 2)
        
        # Display
        cv2.imshow("Gesture Volume Control", img) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__": 
    main() 