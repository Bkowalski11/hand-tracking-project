import cv2
import mediapipe as mp
import time

class HandTracker():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )
        self.drawing = mp.solutions.drawing_utils
    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            for point in hand_landmarks.landmark:
                print(point.x, point.y, point.z)
        return frame
    def find_position(self, frame, hand_no=0):
      lm_list = []
      if self.results.multi_hand_landmarks:
        my_hand = self.results.multi_hand_landmarks[hand_no]
        for id, lm in enumerate(my_hand.landmark):
          h, w = frame.shape
          cx, cy = int(lm.x * w), int(lm.y * h)
          lm_list.append([id, cx, cy])
        return lm_list
tracker = HandTracker()
cap = cv2.VideoCapture(0)
p_time = 0
while True:
    ret, frame = cap.read()
    c_time = time.time()
    time_difference = c_time - p_time
    fps = 1/time_difference
    p_time = c_time
    if time_difference > 0:
        fps = 1 / time_difference
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = tracker.find_hands(frame)
    cv2.putText(
      frame,
      f"FPS: {int(fps)}",
      (10, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
  )
    cv2.imshow('Hand Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
