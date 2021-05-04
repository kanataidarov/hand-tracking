import cv2
import mediapipe as mp 
import time 
import math 
import base 

# Base class for Hand Tracking 
class HandDetector(base.Base): 

  def __init__(self, mode=False, maxHands=1, detectConf=.5, trackConf=.5):
    super().__init__(mode, maxHands, detectConf, trackConf)

    self.mpHands = mp.solutions.hands 
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConf, self.trackConf)
    self.mpDraw = mp.solutions.drawing_utils

  # Finds landmarks on the first hand 
  def FindHandLandmarks(self, img, handNo=0, shouldDraw=True):
    lms = []
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = self.hands.process(imgRgb)
    if results.multi_hand_landmarks: 
      myHand = results.multi_hand_landmarks[handNo]
      for idx, lm in enumerate(myHand.landmark): 
        hImg, wImg, _ = img.shape
        cx, cy = int(lm.x * wImg), int(lm.y * hImg)
        lms.append([idx, cx, cy])

        if shouldDraw:
          self.mpDraw.draw_landmarks(img, myHand, self.mpHands.HAND_CONNECTIONS)

    return lms 

  # Checks which fingers up 
  def FingersUp(self, lms): 
    fingers = []
    tips = [4,8,12,16,20] # Thumb and pointies 

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
  def FindBoundaries(self, img, lms, shouldDraw=False): 
    margin = 10
    if len(lms)>0: 
      fLms = list(zip(*lms))
      xMin = min(fLms[1])
      xMax = max(fLms[1])
      yMin = min(fLms[2])
      yMax = max(fLms[2])
      
      if shouldDraw: 
          cv2.rectangle(img, (xMin-margin,yMin-margin), (xMax+margin,yMax+margin), self.COLORS['YELLOW'], 1)
      
      return (xMax-xMin)*(yMax-yMin)//100

    return 0

  # Finds distance between first and second fingers 
  def FindDist(self, img, lms, first=4, second=8, littleDown=False, shouldDraw=True): 
        length = 0
        if len(lms) > 20: 
            # Filter based on size 
            x1,y1 = lms[first][1],lms[first][2]
            x2,y2 = lms[second][1],lms[second][2]
            cx,cy = (x1+x2)//2,(y1+y2)//2

            cv2.circle(img, (x1,y1), 9, self.COLORS['CYAN'], cv2.FILLED)
            cv2.circle(img, (x2,y2), 9, self.COLORS['CYAN'], cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), self.COLORS['CYAN'], 3)

            length = math.hypot(x2-x1, y2-y1)
            if shouldDraw: 
              if littleDown:
                  cv2.circle(img, (cx,cy), 9, self.COLORS['RED'], cv2.FILLED)
              else:
                  cv2.circle(img, (cx,cy), 9, self.COLORS['CYAN'], cv2.FILLED)
            
        return length


def main(): 
  detector = HandDetector()
  cap = detector.InitCam()

  while True: 
    success, img = cap.read() 
    detector.UpdateFps(img)
    lms = detector.FindHandLandmarks(img)
    fingers = detector.FingersUp(lms)
    area = detector.FindBoundaries(img, lms, shouldDraw=True)

    cv2.imshow("Hand Tracker", img)
    detector.ReleaseCam(cap)

if __name__ == "__main__": 
  main() 
