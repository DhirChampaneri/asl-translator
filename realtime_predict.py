import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import joblib
from collections import deque
from features import landmarks_to_feature

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load trained model (A–Z + SPACE + END)
clf = joblib.load("asl_letters_model.joblib")
CLASS_NAMES = clf.classes_

# 🔧 TUNING KNOBS
CONF_THRESH = 0.65
BUFFER_SIZE = 8
MIN_COUNT = 6
COOLDOWN_START = 12


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pred_buffer = deque(maxlen=BUFFER_SIZE)
    current_text = ""

    cooldown_frames = 0
    last_label = None  # last label we actually applied (A–Z, SPACE, END)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            display_label = "-"
            display_conf = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    feat = landmarks_to_feature(hand_landmarks, w, h)

                    probs = clf.predict_proba([feat])[0]
                    best_idx = probs.argmax()
                    best_label = CLASS_NAMES[best_idx]
                    best_conf = float(probs[best_idx])

                    display_label = best_label
                    display_conf = best_conf

                    if best_conf >= CONF_THRESH:
                        pred_buffer.append(best_label)

            # Decide when to "lock in" a label
            if len(pred_buffer) == BUFFER_SIZE:
                common = max(set(pred_buffer), key=pred_buffer.count)
                count = pred_buffer.count(common)

                if count >= MIN_COUNT and cooldown_frames == 0:
                    applied = False

                    if common == "SPACE":
                        if last_label != "SPACE":
                            current_text += " "
                            last_label = "SPACE"
                            applied = True
                            print("[ADD SPACE]")
                        else:
                            print("[IGNORE REPEAT SPACE]")

                    elif common == "END":
                        if last_label != "END":
                            # Add a period + space to end the sentence
                            current_text = current_text.rstrip() + ". "
                            last_label = "END"
                            applied = True
                            print("[ADD END SENTENCE]")
                        else:
                            print("[IGNORE REPEAT END]")

                    else:
                        # Normal letter A–Z
                        if common != last_label:
                            current_text += common
                            last_label = common
                            applied = True
                            print(f"[ADD LETTER]: {common}")
                        else:
                            print(f"[IGNORE REPEAT LETTER]: {common}")

                    if applied:
                        cooldown_frames = COOLDOWN_START

                pred_buffer.clear()

            if cooldown_frames > 0:
                cooldown_frames -= 1

            # ---------- UI ----------
            cv2.putText(
                frame,
                f"Pred: {display_label} ({display_conf:.2f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Text: {current_text}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Right-hand A–Z + SPACE + END  |  ESC: quit",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            cv2.imshow("ASL Realtime (Alphabet + Space/End)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
