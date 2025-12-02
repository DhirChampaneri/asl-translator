import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def main():
    X = np.load("X.npy")
    y = np.load("y.npy")

    # Ensure labels are plain strings
    y = y.astype(str)

    print("Data shape:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # IMPORTANT:
    # No early stopping, no validation split, no score calls during training
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=700,
        random_state=42,
    )

    print("Training MLPClassifier on full alphabet...")
    clf.fit(X_train, y_train)

    # Manual accuracy calculation (to avoid np.isnan)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)

    joblib.dump(clf, "asl_letters_model.joblib")
    print("Saved model to asl_letters_model.joblib")


if __name__ == "__main__":
    main()
