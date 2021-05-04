# Gesture Volume Control (Mac only)

import cv2 
import time 
import numpy as np 
import hand 
import osascript as osa

class VolumeControl(hand.HandDetector):
    def __init__(self, volume = 0, bar = 400, perc = 0, 
                mode=False, maxHands=1, detectConf=.5, trackConf=.5):
        super().__init__(mode, maxHands, detectConf, trackConf)
        self.volume = volume 
        self.bar = bar 
        self.perc = perc 

    # changes volume by perc
    def TuneVol(self, perc): 
        osa.run("set volume output volume "+str(perc))

    # get current system volume 
    def ReceiveVol(self): 
        return osa.run("get output volume of (get volume settings)")[1]


def main():
    maxArea = 1050
    minArea = 550
    smooth = 5

    detector = VolumeControl() 
    cap = detector.InitCam() 

    length = 0
    perc = 0

    while True: 
        _, img = cap.read() 
        img = cv2.resize(img, (800, 530))
        detector.UpdateFps(img)

        # Find hand landmarks  
        lms = detector.FindHandLandmarks(img)

        # Finds distance between thumb and index fingers
        length = detector.FindDist(img, lms, littleDown=False)
        perc = np.interp(length, [40,400], [0,100])

        # Hand range 40-400, device volume range 0-100
        area = detector.FindBoundaries(img, lms, shouldDraw=True)
        if minArea < area < maxArea and not detector.FingersUp(lms)[4]: 
            detector.FindDist(img, lms, littleDown=True)
            detector.TuneVol(perc)
            volSetClr = detector.COLORS['GREEN']
        else: 
            volSetClr = detector.COLORS['BLUE']

        # HUD 
        cv2.rectangle(img, (40,150), (85,400), detector.COLORS['BLUE'], 2)
        bar = np.interp(length, [40,400], [400,150])
        cv2.rectangle(img, (40,int(bar)), (85,400), detector.COLORS['BLUE'], cv2.FILLED)
        perc = smooth * round(perc/smooth) 
        cv2.putText(img, f'{int(perc)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, detector.COLORS['BLUE'], 3)
        cv2.putText(img, f'VOL: {int(detector.ReceiveVol())}', (650, 50), cv2.FONT_HERSHEY_COMPLEX, 1, volSetClr, 2)
        
        # Display
        cv2.imshow("Gesture Volume Control", img) 
        detector.ReleaseCam(cap)

if __name__ == "__main__": 
    main() 