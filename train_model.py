"""Train and save a Random Forest sign classifier from normalized CSV features."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [*[f"x_{i}" for i in range(21)], *[f"y_{i}" for i in range(21)], *[f"z_{i}" for i in range(21)]]

class ModelTrainer:
    def __init__(self, dataset=Path("dataset/csv/hand_landmarks.csv"), model_dir=Path("models")):
        self.dataset, self.model_dir = Path(dataset), Path(model_dir)

    def load_data(self):
        if not self.dataset.exists(): raise FileNotFoundError(f"Dataset not found: {self.dataset}")
        frame = pd.read_csv(self.dataset)
        required = {"label", *FEATURE_COLUMNS}
        if not required.issubset(frame.columns): raise ValueError("CSV has missing label or feature columns.")
        frame = frame.dropna(subset=["label", *FEATURE_COLUMNS]).drop_duplicates(subset=["label", *FEATURE_COLUMNS])
        if len(frame) < 2 or frame["label"].nunique() < 2: raise ValueError("Need at least two labels and valid samples to train.")
        return frame.sample(frac=1, random_state=42).reset_index(drop=True)

    def train(self):
        frame = self.load_data(); x = frame[FEATURE_COLUMNS].astype(float).to_numpy(); encoder = LabelEncoder(); y = encoder.fit_transform(frame.label)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(x_train, y_train); predicted = model.predict(x_test)
        print("Accuracy:", round(accuracy_score(y_test, predicted), 4)); print(classification_report(y_test, predicted, target_names=encoder.classes_, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, predicted))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_dir / "gesture_model.pkl"); joblib.dump(encoder, self.model_dir / "label_encoder.pkl")
        print("Saved model and label encoder in", self.model_dir)
        return model, encoder

if __name__ == "__main__":
    try: ModelTrainer().train()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error: print("ERROR:", error)
