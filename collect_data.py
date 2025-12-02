import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
from features import landmarks_to_feature

mp_hands = mp.solutions.hands


def collect_for_label(label, num_samples=150):
    """
    Collect num_samples examples for a single label (letter).
    Right hand only for now.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None

    X = []
    y = []

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        print(f"\n=== Collecting data for '{label}' (RIGHT HAND) ===")
        print("Show the ASL sign for this letter and press 's' to start collecting.")
        print("Press ESC to stop early.\n")
        started = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            status_text = f"Label: {label} | Collected: {len(X)}/{num_samples}"
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    if started and len(X) < num_samples:
                        feat = landmarks_to_feature(hand_landmarks, w, h)
                        X.append(feat)
                        y.append(label)

            cv2.imshow("Collect Data (Right Hand Alphabet)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                started = True
                print("Started collecting... Hold the sign steady for a few seconds.")

            if key == 27:  # ESC
                print("Stopped by user.")
                break

            if len(X) >= num_samples:
                print(f"Done collecting {num_samples} samples for '{label}'.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(X) == 0:
        return None, None

    return np.array(X), np.array(y)


if __name__ == "__main__":
    all_X = []
    all_y = []

    # FULL ALPHABET (RIGHT HAND ONLY)
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "END"]

    for label in labels:
        X, y = collect_for_label(label, num_samples=150)
        if X is None:
            print(f"Skipping label '{label}' due to no data.")
            continue
        all_X.append(X)
        all_y.append(y)

    if all_X:
        X_final = np.vstack(all_X)
        y_final = np.concatenate(all_y)

        np.save("X.npy", X_final)
        np.save("y.npy", y_final)
        print("\nSaved X.npy and y.npy")
        print("Shapes:", X_final.shape, y_final.shape)
    else:
        print("No data collected.")
