import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, maxHands = 2, model_comp=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_comp = model_comp
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_comp,self.detectionCon,self.trackCon,)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
            imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRgb)
            # print(results.multi_hand_landmarks)
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlists=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
               # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlists.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 20, (255,0,255),cv2.FILLED)

        return lmlists



def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=False)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()