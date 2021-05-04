import cv2
import time 

# Base class for CV 
class Base(): 

  def __init__(self, mode=False, maxHands=2, detectConf=.5, trackConf=.5):
    self.COLORS = {'PURPLE':(255,0,255), 'GREEN':(0,255,0), 'RED':(0,0,255), 
                  'CYAN':(255,255,0), 'BLUE':(255,0,0), 'YELLOW':(50,255,255)}

    self.pTime = 0
    self.mode = mode 
    self.maxHands = maxHands
    self.detectConf = detectConf
    self.trackConf = trackConf 

  # Show FPS counter 
  def UpdateFps(self, img): 
    cTime = time.time()
    fps = 1/(cTime-self.pTime)
    self.pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, self.COLORS['BLUE'], 3)

  # Initializes video capture device 
  def InitCam(self, wCam = 800, hCam = 600): 
    cap = cv2.VideoCapture(0)
    cap.set(3, int(wCam))
    cap.set(4, int(hCam)) 
    return cap 

  # Releases video capture device 
  def ReleaseCam(self, cap): 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()


def main(): 
  base = Base()
  cap = base.InitCam()

  while True: 
    success, img = cap.read() 
    base.UpdateFps(img)

    cv2.imshow("Base Detection", img)
    base.ReleaseCam(cap)

if __name__ == "__main__": 
  main() 
