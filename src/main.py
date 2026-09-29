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
        self.results = None
    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    def find_position(self, frame, hand_no=0):
        lm_list = []
        if not self.results or not self.results.multi_hand_landmarks:
            return lm_list
        if hand_no >= len(self.results.multi_hand_landmarks):
            return lm_list

        hand_landmarks = self.results.multi_hand_landmarks[hand_no]
        h, w, _ = frame.shape
        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            lm_list.append([landmark_id, cx, cy])
        return lm_list
tracker = HandTracker()
cap = cv2.VideoCapture(0)
p_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    c_time = time.time()
    time_difference = c_time - p_time
    fps = 1 / time_difference if time_difference > 0 else 0
    p_time = c_time
    frame = cv2.flip(frame, 1)
    frame = tracker.find_hands(frame)
    lm_list = tracker.find_position(frame)
    finger_tips = [8, 12, 16, 20]
    fingers_raised = 0
    if lm_list:
        fingers_raised = sum(
            1
            for tip_id in finger_tips
            if lm_list[tip_id][2] < lm_list[tip_id - 2][2]
        )
        hand_label = tracker.results.multi_handedness[0].classification[0].label
        thumb_tip_x = lm_list[4][1]
        thumb_joint_x = lm_list[3][1]
        if (hand_label == "Right" and thumb_tip_x < thumb_joint_x) or (
            hand_label == "Left" and thumb_tip_x > thumb_joint_x
        ):
            fingers_raised += 1
    cv2.putText(
        frame,
        f"Raised fingers: {fingers_raised}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 90),
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
