import cv2
import mediapipe as mp 
import time 

class HandDetector(): 
  def __init__(self, mode=False, maxHands=2, detectConf=.5, trackConf=.5):
    self.cTime = 0
    self.pTime = 0

    self.mode = mode 
    self.maxHands = maxHands
    self.detectConf = detectConf
    self.trackConf = trackConf 

    self.mpHands = mp.solutions.hands 
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConf, self.trackConf)
    self.mpDraw = mp.solutions.drawing_utils

  def findHands(self, img, shouldDraw=True): 
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRgb)

    if self.results.multi_hand_landmarks: 
      for handLms in self.results.multi_hand_landmarks: 
        if shouldDraw: 
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
    
    return img

  def findPosition(self, img, handNo=0, shouldDraw=True):
    lms = []
    if self.results.multi_hand_landmarks: 
      myHand = self.results.multi_hand_landmarks[handNo]
      for id, lm in enumerate(myHand.landmark): 
        h,w,c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        lms.append([id, cx, cy])

        if shouldDraw:
          cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED) 

    return lms 

  def updateFps(self, img): 
    self.cTime = time.time()
    fps = 1/(self.cTime-self.pTime)
    self.pTime = self.cTime
    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

def initCam(): 
  wCam, hCam = 800, 600 
  cap = cv2.VideoCapture(0)
  cap.set(3, wCam)
  cap.set(4, hCam) 
  return cap 

def main(): 
  cap = initCam()
  detector = HandDetector()

  while True: 
    success, img = cap.read() 
    img = detector.findHands(img)
    lms = detector.findPosition(img)
    detector.updateFps(img)

    cv2.imshow("Hand Tracker", img)
    cv2.waitKey(1)

if __name__ == "__main__": 
  main() 
