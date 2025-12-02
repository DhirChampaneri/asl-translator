import numpy as np
import mediapipe as mp

# Just in case you need the MediaPipe structures elsewhere
mp_hands = mp.solutions.hands


def landmarks_to_feature(hand_landmarks, img_width, img_height):
    """
    Convert MediaPipe hand landmarks into a normalized feature vector.

    - Uses x, y, z for each of the 21 landmarks (63 values).
    - Coordinates are made:
        * relative to the wrist (landmark 0)
        * scaled by overall hand size so distance/scale changes matter less.
    """

    # Collect raw (x, y, z) in image space
    xs = []
    ys = []
    zs = []

    for lm in hand_landmarks.landmark:
        x = lm.x * img_width
        y = lm.y * img_height
        z = lm.z * img_width  # z is unitless, but scale roughly like x
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # Use wrist (landmark 0) as reference point
    base_x = xs[0]
    base_y = ys[0]
    base_z = zs[0]

    xs_rel = xs - base_x
    ys_rel = ys - base_y
    zs_rel = zs - base_z

    # Hand size = max distance from wrist in xy-plane
    dists = np.sqrt(xs_rel**2 + ys_rel**2)
    hand_size = np.max(dists) + 1e-6  # avoid division by zero

    xs_norm = xs_rel / hand_size
    ys_norm = ys_rel / hand_size
    zs_norm = zs_rel / hand_size

    # Stack into a single 1D feature vector: [x0, y0, z0, x1, y1, z1, ...]
    features = np.stack([xs_norm, ys_norm, zs_norm], axis=1).reshape(-1)

    return features
